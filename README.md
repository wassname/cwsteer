# cwsteer — contrastive weight steering

A small, working library for steering a language model along a contrastive axis by editing
its weights. It's a modification of [Fierro & Roger, 2025 - Weight steering](https://arxiv.org/abs/2511.05408). It fits one conditioned adapter on (chosen, rejected) completion pairs, calibrates the steering strength so generations stay coherent, then bakes the chosen strength into the weights for inference.

## Weight steering

Steering is promising but seen as unreliable. It is promising because it is
self-supervised, meaning it doesn't rely on labels that we don't have. And it is
internal, meaning it is less prone to the reward hacking that more distal
optimisation like reinforcement learning is subject to. Weight steering is a newer
form: more reliable than activation steering and able to compose across rounds, at
the cost of being a little less purely internal. That extra reliability is the main
reason to reach for it.

Early results ([LW](https://www.lesswrong.com/posts/HYTbakdHpxfaCowYp/steering-language-models-with-weight-arithmetic?commentId=GomjgJDtr5JhEAuC3),
0.6B-4B) are consistent with this: the steered effect is monotonic and coherent over a
wide range, with the lowest uncertainty and highest answer-coherence of the methods
compared, and it beats prompting and several activation-steering baselines on surgical
informedness. Raw effect size is currently mid-pack and under-calibrated; a fuller eval
is in progress.

The base method is the excellent contrastive weight steering
([Fierro & Roger, 2025](https://arxiv.org/abs/2511.05408),
[code](https://github.com/safety-research/weight-steering)). It trains an adapter
on a model's own contrastive completions, then uses that adapter as a direction in
weight space.

## The adapter

The plain LoRA form, per target Linear (the base weight `W` is frozen):

$$y = x W^\top + c \cdot \frac{\alpha}{r}\,(x A^\top) B^\top$$

`c = 0` gives exactly the base model; `+c` and `-c` are the two signed poles of one
low-rank direction. The PiSSA variant differs: at init it replaces `W` with the
residual `W - U_r S_r V_r^\top` and *always* adds the top-`r` SVD back in the forward
pass as `U_r \mathrm{diag}(S_r + c\,\Delta s) V_r^\top`, where the dial `c` scales only
the trained singular-value deltas `Δs` (not the whole SVD). So `c = 0` adds back
`U_r S_r V_r^\top` and reconstructs the base (up to SVD round-off), while `±c` grow or
shrink those top directions. See [`adapter.py`](src/cwsteer/adapter.py).

The persona used to elicit the contrast is carried by the completions only, and is
stripped before training, so it is not part of the deployed adapter.

The two poles share one axis. The same pairs are trained under both signs of `c`
(favouring `cho` at `+c`, `rej` at `-c`); that sign flip is the "reversal", so the two
updates reinforce one direction instead of fighting. At inference the dial `c` slides
the model along that line. See [`train.py`](src/cwsteer/train.py).

![steering direction](docs/steering_geometry.png)

*One adapter is one direction. The two poles' training gradients (`g+` at `+c`, `g-` at
`-c`) pull one shared axle; the `-c` sign flip makes them reinforce along the axle (`v`)
while the sideways parts cancel, so the same pairs trained under both signs learn one
direction. A little more detail in the [appendix](#appendix-gradient-geometry).*

## What this variant changes

A few changes to the base method:

- one parameterized adapter instead of two separate adapters: a single low-rank
  direction with a signed strength `c`, so the two poles share one axis and `c=0`
  is exactly the base model ([`adapter.py`](src/cwsteer/adapter.py))
- a PiSSA ([Meng et al., 2024](https://arxiv.org/abs/2404.02948)) initialisation
  that starts the adapter from the top-r SVD of the weight, rather than a random
  low-rank init (this mutates the float weight at init, so quantised models use
  the plain LoRA adapter instead) ([`adapter.py`](src/cwsteer/adapter.py))
- a KL constraint that limits how much steering shifts the output distribution away
  from the base model ([`train.py`](src/cwsteer/train.py))
- a calibration pass that finds the largest coherent steering strength before
  replaying the completions (optional, mainly for repeated applications)
  ([`c_scan.py`](src/cwsteer/c_scan.py))
- a logit filter during generation that keeps the persona used to elicit the
  contrast from leaking into the completions ([`pairs.py`](src/cwsteer/pairs.py))
- a generation filter that drops pairs showing leakage, refusal, repetition, or
  too little contrast ([`pairs.py`](src/cwsteer/pairs.py))
- stricter contrastive pair filtering ([`pairs.py`](src/cwsteer/pairs.py))

Personas and prompts are validated with the
[persona-steering template library](https://github.com/wassname/persona-steering-template-library)
before use; if the pairs don't cleanly separate the behaviour, everything downstream
fails.

The single signed adapter, KL constraint, calibration, and pair filtering follow
earlier [AntiPaSTO work](https://arxiv.org/pdf/2601.07473); PiSSA is a separate
existing method.

The adapter, training, bake core, calibration (`c_scan`), on-policy pole generation,
and the persona-leakage logit filter are all in this repo and covered by `just smoke`.
A standalone post-hoc pair filter (capability beyond the generation-time prevention
above) is the remaining piece.

Weight steering is less purely "internal" than activation steering, because it
adds an external objective: nll over the model's own completions. I haven't yet
built a good intuition for what this means for behaviours like sandbagging and
reward hacking, which result from a mismatch between outer logprobs and inner
hidden states.

## Install

```sh
git clone https://github.com/wassname/cwsteer && cd cwsteer
uv sync                          # or: pip install -e .
uv run python scripts/smoke.py   # optional: verify the core on a tiny model
```

## Quickstart

First decide and validate your contrastive personas and prompts. If the pairs don't
cleanly separate the behaviour you want, training, calibration, and baking all inherit
that failure, so this step gates everything downstream. Curate them with the
[persona-steering template library](https://github.com/wassname/persona-steering-template-library).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cwsteer import ModulatedLoRA, TrainCfg, train_adapter, AdapterSpec, baked

model = AutoModelForCausalLM.from_pretrained("...", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("...")
pairs = [{"prompt": "...", "cho": "...", "rej": "..."}, ...]   # cho under +pole, rej under -pole

lora = train_adapter(model, tok, pairs, TrainCfg(r=16, kl_lambda=0.03))
spec = AdapterSpec.from_lora(lora, default_c=0.8)             # 0.8 = calibrated strength
with baked(model, [spec]):                                    # weights edited in place
    out = model.generate(...)                                 # steered inference
# base weights restored on exit
```

## Smoke

```sh
just smoke    # CPU, tiny random Qwen3, ~1-2 min
```

Checks the core invariants: LoRA `c=0` equals base exactly, PiSSA `c=0` matches
base through the SVD round-trip, `train_adapter` runs the full NLL+KL+val path,
and `baked()` shifts the output then restores the base weights byte-for-byte.

## Appendix: gradient geometry

Advanced; skip unless you care about the training dynamics.

Why does training the same pairs under both signs of `c` make the two poles reinforce
rather than cancel? Picture each pole's gradient as a rope pulling the shared direction
(the axle through the base model at `0`). The two pulls open at an angle. In this
idealised picture their shared along-the-axle component adds up and moves the direction,
while the perpendicular parts are roughly equal and opposite and cancel. The `-c` sign
flip is what points both gradients the same way along the axle (the figure above). In
our runs the cosine between the two poles' gradients (the `cos` column logged by
[`train.py`](src/cwsteer/train.py)) started near 0.48 and fell toward 0 as the adapter
grew.

This is the key difference from ordinary supervised fine-tuning. SFT maximises the
likelihood of one target and learns whatever lowers loss, including the direction `cho`
and `rej` have in common. Training both signs cancels that shared component and keeps only
the bidirectional axis that separates the two poles. And because the adapter enters as
`c·δ`, every weight update is a bidirectional move through `0`: `+c` and `-c` are the same
`δ` with opposite sign, so each update is symmetric about the base model rather than a
one-way push.

So unlike SFT, the update is internally constrained and parametrised rather than free
output tuning. That is what you want if you are moving toward internal interventions (in
the direction of activation steering) and away from fitting outputs (SFT, RL) -- though it
still sits short of pure activation steering, which carries no external likelihood
objective at all.

This is the promise of steering: no new reward labels. You pick the contrast axis (the
personas and prompts), and the model's own on-policy completions and internal directions
supply the rest of the signal.

## Sources

- Base method: [safety-research/weight-steering](https://github.com/safety-research/weight-steering)
- Adapter pattern: [lora-lite](https://github.com/wassname/lora-lite)
