# cwsteer extraction plan (working notes, not user-facing)

Goal: a standalone, documented, smoke-tested library for contrastive weight steering,
extracted from w2schar-mini `src/csm/ws` + `src/csm/gen`. Base method: Fierro & Roger
2025 (arXiv 2511.05408), code safety-research/weight-steering.

## Status

- Moved + smoked: `adapter.py`, `train.py`, `bake.py` (v0 core).
- Copied (import-clean, not yet wired/smoked): `c_scan.py`, `pairs.py`, `dialogue.py`,
  `probes.py`, `prompts_pool.py`, `history.py` + pool data files.

## Generalisation boundary (from dependency map)

- calibration `c_scan`: self-contained except (a) `from csm.gen.dialogue import
  run_probe, DialogueCfg` -> vendored (dialogue.py copied); (b) `CANARY_PROBES`
  (Petrov/counsel/troubleshoot text) -> should become an injected `canary_probes` arg;
  (c) tinymfv pmass -> inject `forced_choice_scorer` or make optional. Already guarded
  by `n_vignettes>0`, so `n_vignettes=0` runs without tinymfv (pmass forced 1.0).
- pole generation: move `generate_pairs_from_personas` (clean, persona strings as args).
  Do NOT move `generate_candidate_pairs` (couples to `PAIR_BEHAVIOR_HINTS` 26 csm axes).
- logit filter (capability 3): `PersonaOnlyRepetitionPenalty` + helpers, zero csm
  coupling, move verbatim. Physically fused into `_generate_batched`.
- generation filter (capability 4): does NOT exist as a function. Only an
  empty/identical drop + generation-time prevention. Must be BUILT (compose
  persona-overlap / refusal / distinct_n / contrast) or the README claim softened.
- leave csm-side: pairs.md parse (`load_pairs_md`/`_strip_decoration`), `prompts_pool`
  data is csm prompt-bank, `probes.py` (`_1p`/`_3p` seats), `history.py` (round-dir
  composition). Copied for reference but may not all stay.

## Outstanding asks (mirror of task list)

README/docs:
- code links in "What this variant changes" bullets; drop "lives elsewhere" after move.
- mention persona-library validation where the method is described (done in quickstart).

Figures (xkcd, both body + appendix):
- REAL xkcd font = "xkcd Script" (installed from ipython/xkcd-font), NOT "Coming Soon".
- wobbly hand-drawn lines (feTurbulence + feDisplacementMap; rsvg-convert honors it,
  cairosvg does not).
- xkcd stick figures; appendix pullers' arms collinear with g+/g- ropes.
- embed PNG in README (GitHub may not run SVG filters); keep SVG as source.

Code:
- wire moved modules into `__init__`, extend `scripts/smoke.py` to exercise them.
- tinymfv: dependency vs injected scorer.
- PCGrad: keep / drop flag / drop-but-keep-cos-diagnostic (awaiting user pick).
