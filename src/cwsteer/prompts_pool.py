"""Measured moral-judgment prompts for persona-conditioned pair sampling.

Each round the harness samples prompts from POOL; the teacher chooses only an
axis label + scenario family. The harness filters a frozen persona descriptor
library toward that axis, samples templates, and the student generates BOTH
poles on-policy over these prompts.

== Where this comes from (read before editing) ==

POOL is no longer hand-authored. It is built by `scripts/build_pool.py` from
`tiny-mfv scifi` judgment prompts plus a smaller genies sycophancy/control set,
and written to `pool.jsonl` with provenance in `pool_manifest.json`. Rebuild with:

    uv run python scripts/build_pool.py

Why datasets, not hand-authoring: the old POOL was diverse in DOMAIN but monotone
in STRUCTURE (mostly first-person action dilemmas: "what do you do?"). Under one
persona the student then emitted a single canned scaffold per pole across every
prompt (task-62: 13 near-identical "### The Stakes" essays -> the adapter
memorised the format, val nll+ 0.95->4.7). The simplified pool instead anchors
on short `tiny-mfv`-style judgments, closer to the eval task while still leaving
room for open reasoning.
Spec: docs/spec/20260606_dataset_prompt_pool.md.

Composition (see manifest): mostly moral `tiny-mfv scifi` judgment prompts,
plus AI-seat sycophancy and a few deliberate NON-MORAL controls so the student
does not learn "every prompt is an ethics test" (CLAUDE.md). Eval-disjoint from
held-out `tiny-mfv` configs by 10-word-shingle dedup; AIRiskDilemmas is reserved
for a future eval split and not used here.
"""
import json
from pathlib import Path

_POOL_FILE = Path(__file__).with_name("pool.jsonl")

POOL_ROWS: list[dict] = [
    json.loads(line)
    for line in _POOL_FILE.read_text().splitlines()
    if line.strip()
]

POOL: list[str] = [row["text"] for row in POOL_ROWS]

# Prompts that survived the OpenRouter prompt screen (scripts/apply_prompt_screen.py
# freezes scripts/validate_persona_axes_openrouter.py's kept_prompts here): pos/neg
# poles that came out length-balanced, on-prompt, and axis-contrasting on a cheap
# Qwen. The `character` family is restricted to these ONLY when a profile sets
# restrict_validated_prompts=True (rows_for_family(..., validated_only=True)); it is
# off by default so smoke/other profiles are unaffected. CAVEAT this CANNOT screen
# out the 2B student's own loop-collapse (degenerate): a cheap model does not loop
# where a 2B does (8B-clean vs 2B-loop sets are largely decorrelated) -- that is a
# student-size problem, not a prompt one. And because the pool holds only ~8 prompts
# per moral axis, restricting starves all but the richest axis (autonomy); see the
# per-axis note on qwen-2b-3keep.
_VALIDATED_FILE = Path(__file__).with_name("pool_validated.json")
VALIDATED_PROMPTS: set[str] = (
    set(json.loads(_VALIDATED_FILE.read_text())["kept_prompts"])
    if _VALIDATED_FILE.exists() else set()
)

SCENARIO_FAMILIES = ("mixed", "character", "sycophancy", "power", "control")


def rows_for_family(
    family: str,
    *,
    required_axes: tuple[str, ...] = (),
    forbidden_axes: tuple[str, ...] = (),
    validated_only: bool = False,
) -> list[dict]:
    """Tagged scenario-library slices.

    `mixed` is the default short-form moral-behavior pool. Keep the dedicated
    `genies_preferences` sycophancy rows behind the explicit `sycophancy`
    family, because several of those rows are malformed prompt fragments and
    they polluted the Qwen smoke with off-axis coding/quiz items. Keep the
    cropped `machiavelli` scenes behind the explicit `power` family too: they
    carry useful choice-label metadata, but the cropped prompt can drop
    narrative subtext that mattered in the original scene.
    """
    if family not in SCENARIO_FAMILIES:
        raise ValueError(
            f"unknown scenario family {family!r}; choose one of {SCENARIO_FAMILIES}")
    out: list[dict] = []
    for row in POOL_ROWS:
        tags = set(row.get("tags", ()))
        axes = set(row.get("axes", ()))
        if required_axes and not (axes & set(required_axes)):
            continue
        if forbidden_axes and (axes & set(forbidden_axes)):
            continue
        if (
            family == "mixed"
            and "non-moral" not in tags
            and "control" not in tags
            and "sycophancy" not in tags
            and "cropped" not in tags
        ):
            out.append(row)
        elif family == "character" and ("character" in tags or "1p" in tags):
            if not validated_only or row["text"] in VALIDATED_PROMPTS:
                out.append(row)  # validated_only=False lets the screen see all 64
        elif family == "sycophancy" and "sycophancy" in tags:
            out.append(row)
        elif family == "power" and ("power" in tags or "narrative" in tags):
            out.append(row)
        elif family == "control" and ("control" in tags or "non-moral" in tags):
            out.append(row)
    if not out:
        raise ValueError(f"scenario family {family!r} selected zero rows")
    return out
