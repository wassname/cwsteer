# cwsteer — contrastive weight steering

A reusable core for steering an LLM along a contrastive axis by editing its
**weights** (not activations): fit ONE conditioned adapter on (cho, rej) pairs,
calibrate the strength so generations stay coherent, then bake the chosen
strength into the weights for inference.

This is the contrastive modification of plain weight steering
([wassname/weight-steering](https://github.com/wassname/weight-steering)),
extracted from [wassname/w2schar-mini](https://github.com/wassname/w2schar-mini).
Status: **v0** — the stable core (adapter + train + bake) with a CPU smoke. The
gen / filter / calibrate / persona-search / eval modules land here as they are
extracted; full design in `mft_honesty` spec #54.

## Mechanism

One parameterized adapter with a scalar coefficient `c`, per target Linear (W frozen):

$$y = x W^\top + c \cdot \frac{\alpha}{r}\,(x A^\top) B^\top$$

- `c = 0` reconstructs base exactly (LoRA short-circuits; PiSSA round-trips the SVD).
- `+c` and `-c` are the two signed poles of ONE banked low-rank direction `δ`.
- The persona that induced the contrast is carried by the **completions only** and
  stripped before training, so it does not leak into the deployed adapter.

The two poles are trained to tug in opposite directions on the same axis. One pole's
pairs are reversed so both poles pull the shared direction the same way:

![steering direction](docs/steering_direction.svg)

PiSSA (the default in w2schar-mini) physically extracts the top-r SVD of W into the
adapter and mutates `layer.weight` at init, so it needs a float (bf16) `nn.Linear`;
quantized (nf4/int8) models must use the plain LoRA adapter.

## What's modified vs plain weight steering

Inherited from w2schar-mini (see its README for the full table):
- one parameterized adapter instead of two separate adapters;
- margin-NLL objective on contrastive pairs + a KL constraint to base for coherence;
- calibrate `c` downward until steered output stays coherent on a held-out probe set;
- stricter contrastive-pair filtering.

The README's full "ours vs inherited" table, the gen/filter/calibrate modules, and
the tinymfv comparison vs `weight-steering-lite` are tracked in spec #54 and land as
those modules are extracted.

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cwsteer import ModulatedLoRA, TrainCfg, train_adapter, AdapterSpec, baked

model = AutoModelForCausalLM.from_pretrained("...", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("...")
pairs = [{"prompt": "...", "cho": "...", "rej": "..."}, ...]   # cho under +pole, rej under -pole

lora = train_adapter(model, tok, pairs, TrainCfg(r=16, kl_lambda=0.03, pcgrad=True))
spec = AdapterSpec.from_lora(lora, default_c=0.8)              # 0.8 = calibrated strength
with baked(model, [spec]):                                    # weights edited in place
    out = model.generate(...)                                 # ... steered inference
# base weights restored on exit
```

## Smoke

```sh
just smoke    # CPU, tiny random Qwen3, ~1-2 min, no GPU/network beyond HF cache
```

Asserts the four core invariants: LoRA `c=0`==base exactly, PiSSA `c=0`≈base
(SVD round-trip), `train_adapter` runs the full NLL+KL+PCGrad+val path, and
`baked()` shifts then byte-identically restores the weights.

## Sources

- Extracted from [w2schar-mini](https://github.com/wassname/w2schar-mini) `src/csm/ws/`
- Adapter pattern: [wassname/lora-lite](https://github.com/wassname/lora-lite)
- Eval (planned): [tinymfv](https://github.com/wassname/tinymfv)
