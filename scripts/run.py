"""Real end-to-end cwsteer run on good data.

Brings the on-policy persona-contrast recipe together on a real model: the student
writes both poles under a (pos, neg) character persona, the persona is stripped
and the poles logit-filtered, a ModulatedPiSSA weight-steering adapter is trained
on the (cho, rej) pairs, then c_scan walks |c| down until the three coherence
gates hold against the tinymfv backend, and the calibrated c is baked into W.

Data lives in scripts/data/ (brought over once, provenance below):
  - prompts.jsonl: 122 third-person moral-judgment prompts, TEXT ONLY -- the
    foundation labels are stripped because training selected/keyed on them would
    leak the tinymfv eval. Source: w2schar-mini csm/gen/pool.jsonl (tiny-mfv
    scifi judgments + curated Forethought seeds + non-moral controls).
  - persona_pairs.jsonl: 27 mirror-tested (pos, neg) character axes. Source:
    persona-steering-template-library data/persona_pairs_w2s_character.jsonl.

Usage:
    uv run python scripts/run.py --model google/gemma-2-2b-it --axis weigh_who_is_affected
"""
import argparse
import json
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from cwsteer import (
    ModulatedPiSSA,
    TrainCfg,
    baked,
    c_scan,
    generate_pairs_from_personas,
    pissa_to_lora_spec,
    train_adapter,
)

DATA = Path(__file__).parent / "data"
# Same wrapper w2schar uses: a short character descriptor -> a persona instruction.
PERSONA_TEMPLATE = "Pretend to be someone who {descriptor}."


def load_prompts(n: int, seed: int) -> list[str]:
    rows = [json.loads(l) for l in (DATA / "prompts.jsonl").read_text().splitlines() if l.strip()]
    texts = [r["text"] for r in rows]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(texts), generator=g)[:n].tolist()
    return [texts[i] for i in idx]


def load_persona(axis_id: str) -> tuple[str, str]:
    rows = [json.loads(l) for l in (DATA / "persona_pairs.jsonl").read_text().splitlines() if l.strip()]
    by_id = {r["id"]: r for r in rows}
    if axis_id not in by_id:
        raise KeyError(f"axis {axis_id!r} not in {sorted(by_id)}")
    r = by_id[axis_id]
    return (PERSONA_TEMPLATE.format(descriptor=r["pos"]),
            PERSONA_TEMPLATE.format(descriptor=r["neg"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--axis", default="weigh_who_is_affected")
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-vignettes", type=int, default=16)
    # 128, not 256: at 256 the 'weighs who is affected' persona runs gemma-2-2b
    # into an enumeration loop (cho becomes a 250-tok run-on vs a 1-sentence rej),
    # which both skews length and blows train memory on the 256k-vocab logits.
    ap.add_argument("--max-new-tokens", type=int, default=128)
    # train batch 2, not the TrainCfg default 4: gemma-2's 256k vocab makes the
    # cross_entropy logits.float() the memory hog; 4 OOMs a 24GB card by ~250MB.
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/run")
    args = ap.parse_args()
    logger.info(f"argv: {vars(args)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device).eval()

    pos_persona, neg_persona = load_persona(args.axis)
    prompts = load_prompts(args.n_prompts, args.seed)
    logger.info(f"axis={args.axis}\n  pos: {pos_persona}\n  neg: {neg_persona}")
    logger.info(f"prompts: {len(prompts)} (e.g. {prompts[0][:90]!r})")

    # === 1. on-policy poles: student writes cho under pos, rej under neg ===
    pairs = generate_pairs_from_personas(
        model, tok, prompts,
        pos_persona=pos_persona, neg_persona=neg_persona,
        max_new_tokens=args.max_new_tokens, batch_size=8, seed=args.seed)
    logger.info(f"=== kept {len(pairs)}/{len(prompts)} on-policy pairs ===")
    # SHOULD: cho enacts 'weighs who is affected', rej 'asserts without weighing',
    # both in the student's own voice with NO 'Pretend to be'/persona echo (stripped).
    p0 = pairs[0]
    logger.info(f"--- FULL first pair ---\nPROMPT: {p0['prompt']}\n"
                f"CHO (pos pole): {p0['cho']}\nREJ (neg pole): {p0['rej']}")

    # === 2. train the weight-steering adapter on (cho, rej) ===
    cfg = TrainCfg(steps=args.steps, batch_size=args.batch_size)  # r=16, alpha=32, lr=2e-4, kl=0.032, pcgrad -- defaults
    trained = train_adapter(model, tok, pairs, cfg, adapter_cls=ModulatedPiSSA)

    # === 3. calibrate |c| against the tinymfv coherence gates ===
    c_cal, scan = c_scan(model, tok, trained, init_c=1.0, n_vignettes=args.n_vignettes,
                         max_think_tokens=64, probe_max_new_tokens=128, batch_size=4)
    base_row = scan[0]
    logger.info(f"=== calibrated c={c_cal:.3f} | base pmass={base_row['pmass']:.3f} "
                f"ppx_json_mfv={base_row['ppx_json_mfv']:.1f} ===")

    # === 4. bake the calibrated c into W and save ===
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = pissa_to_lora_spec(trained, default_c=c_cal)  # PiSSA bake path (.from_lora is LoRA-only)
    with baked(model, [spec]):
        logger.info(f"baked c={c_cal:.3f} into W (restored on exit)")
    (out / "summary.json").write_text(json.dumps({
        "model": args.model, "axis": args.axis,
        "n_prompts": len(prompts), "n_pairs_kept": len(pairs),
        "c_calibrated": c_cal, "base_pmass": base_row["pmass"],
        "pos_persona": pos_persona, "neg_persona": neg_persona,
    }, indent=2))
    logger.success(f"run OK: {len(pairs)} pairs, c={c_cal:.3f} -> {out}/summary.json")


if __name__ == "__main__":
    main()
