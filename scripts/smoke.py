"""Fast CPU smoke for cwsteer's stable core (adapter + train + bake).

Exercises the load-bearing correctness properties on a tiny random Qwen3, no GPU,
no network beyond the HF cache (~1-2 min):

  A. ModulatedLoRA  c=0 == base EXACTLY (hook short-circuit); +c/-c differ.
  B. ModulatedPiSSA c=0 ≈ base (SVD round-trip, bf16 noise); +c/-c differ.
  C. train_adapter runs the full NLL + KL + PCGrad + val path on toy pairs.
  D. baked() shifts the output, then restores base byte-identically on exit.

Each check is a fail-fast assert: a green run means the four invariants hold.
"""
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from cwsteer import (
    AdapterSpec,
    ModulatedLoRA,
    ModulatedPiSSA,
    TrainCfg,
    baked,
    c_scan,
    generate_pairs_from_personas,
    train_adapter,
)

MODEL = "wassname/qwen3-5lyr-tiny-random"
PROBE = "The capital of France is "


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval()
    return model, tok


@torch.no_grad()
def logits(model, ids):
    return model(ids).logits


def toy_pairs() -> list[dict]:
    # cho = terse/decisive, rej = waffling. Random model can't learn meaning;
    # this only needs to drive the train path, not produce a good adapter.
    base = [
        ("Should I ship the fix now?", "Yes. Ship it.", "Well, it depends on many factors to consider..."),
        ("Is the result significant?", "No, it is within noise.", "Hmm, that is hard to say, perhaps maybe..."),
        ("Do we keep this adapter?", "Drop it.", "I am not entirely sure, let us deliberate at length..."),
        ("Was the run a success?", "It failed.", "There are arguments on both sides here, honestly..."),
    ]
    return [{"prompt": p, "cho": c, "rej": r} for (p, c, r) in base * 2]  # 8 pairs


def main():
    torch.manual_seed(0)
    model, tok = load()
    ids = tok(PROBE, return_tensors="pt").input_ids
    base = logits(model, ids)
    logger.info(f"base logits: shape={tuple(base.shape)} mean={base.float().mean():+.4f}")

    # ── A. ModulatedLoRA: c=0 is an exact short-circuit ──────────────────────
    lora = ModulatedLoRA(model, r=8, alpha=16.0, targets=("all-linear",))
    with lora(model, c=0.0):
        y0 = logits(model, ids)
    assert torch.equal(y0, base), "LoRA c=0 must equal base EXACTLY (short-circuit)"
    with lora(model, c=+1.0):
        yp = logits(model, ids)
    with lora(model, c=-1.0):
        yn = logits(model, ids)
    d_p = (yp - base).abs().float().mean().item()
    d_n = (yn - base).abs().float().mean().item()
    d_pn = (yp - yn).abs().float().mean().item()
    assert d_p > 0 and d_n > 0, "steering at ±c must move the logits"
    assert d_pn > 0, "+c and -c must differ from each other"
    logger.info(f"A LoRA OK: |c0-base|=0 exact, |+1-base|={d_p:.4f}, |-1-base|={d_n:.4f}, |+1 - -1|={d_pn:.4f}")

    # ── C. train_adapter: full NLL + KL + PCGrad + val path ──────────────────
    cfg = TrainCfg(r=8, alpha=16.0, steps=10, batch_size=2, n_val_pairs=2,
                   lr=1e-3, kl_lambda=0.03, pcgrad=True, max_len=64)
    trained = train_adapter(model, tok, toy_pairs(), cfg, adapter_cls=ModulatedLoRA)
    with trained(model, c=1.0):
        yt = logits(model, ids)
    d_t = (yt - base).abs().float().mean().item()
    assert torch.isfinite(yt).all(), "trained adapter produced non-finite logits"
    assert d_t > 0, "trained adapter at c=1 must change the output"
    logger.info(f"C train OK: {cfg.steps} steps, trained |c1-base|={d_t:.4f}")

    # ── D. bake + restore: in-place W edit, then byte-identical restore ──────
    spec = AdapterSpec.from_lora(trained, default_c=1.0)
    with baked(model, [spec]):
        yb = logits(model, ids)
    after = logits(model, ids)
    d_b = (yb - base).abs().float().mean().item()
    assert d_b > 0, "baked adapter must change the output"
    assert torch.equal(after, base), "baked() must restore base EXACTLY on exit"
    logger.info(f"D bake OK: |baked-base|={d_b:.4f}, restore exact ✓")

    # ── B. ModulatedPiSSA: c=0 ≈ base via SVD round-trip (mutates W) ─────────
    model2, _ = load()                       # fresh: PiSSA mutates layer.weight at init
    base2 = logits(model2, ids)
    pissa = ModulatedPiSSA(model2, r=16, targets=("all-linear",),
                           selection_score="s_only")  # s_only: no calib activations needed
    with pissa(model2, c=0.0):
        p0 = logits(model2, ids)
    rt_err = (p0 - base2).abs().float().mean().item()
    base_scale = base2.abs().float().mean().item()
    assert rt_err < 0.05 * base_scale, f"PiSSA c=0 round-trip error {rt_err:.4f} too large vs base {base_scale:.4f}"
    with pissa(model2, c=+1.0):
        pp = logits(model2, ids)
    with pissa(model2, c=-1.0):
        pn = logits(model2, ids)
    d_pp = (pp - base2).abs().float().mean().item()
    d_pn2 = (pp - pn).abs().float().mean().item()
    assert d_pp > rt_err and d_pn2 > 0, "PiSSA ±c must move more than round-trip noise and differ"
    logger.info(f"B PiSSA OK: c0 round-trip err={rt_err:.5f} (<5% of {base_scale:.3f}), |+1-base|={d_pp:.4f}, |+1 - -1|={d_pn2:.4f}")

    # ── E. pole generation + logit filter: on-policy cho/rej, persona stripped ─
    # Random model -> gibberish completions; we only check the gen + persona-leak
    # logit-filter + degenerate-drop path runs and returns well-formed rows.
    pairs_out = generate_pairs_from_personas(
        model, tok,
        ["Should I ship the fix now?", "Is the result significant?"],
        pos_persona="You are terse and decisive. Answer in one short sentence.",
        neg_persona="You are long-winded and never commit to an answer.",
        max_new_tokens=24, batch_size=2,
    )
    assert isinstance(pairs_out, list), "generate_pairs_from_personas must return a list"
    for r in pairs_out:
        assert set(r) >= {"prompt", "cho", "rej"} and r["cho"] != r["rej"], f"bad pair row {r!r}"
    logger.info(f"E pole-gen OK: kept {len(pairs_out)} on-policy pairs (logit filter + degenerate-drop ran)")

    # ── F. c_scan calibration: walk c down until the 3 coherence gates hold ────
    # n_vignettes=0 -> no tinymfv (pmass forced 1.0); tiny token budgets keep it fast.
    c_cal, _scan_log = c_scan(model, tok, trained, init_c=1.0, n_vignettes=0,
                              max_think_tokens=8, probe_max_new_tokens=16, batch_size=1)
    assert 0.0 < c_cal <= 1.0, f"c_scan returned out-of-range c={c_cal}"
    logger.info(f"F c_scan OK: calibrated c={c_cal:.3f} (n_vignettes=0, tinymfv-free)")

    logger.success("cwsteer smoke PASSED — adapter (LoRA + PiSSA), train, bake/restore, "
                   "pole-gen + logit-filter, c_scan calibration all green")


if __name__ == "__main__":
    main()
