# Coupled preference-loss verification

Author: PI/OpenAI. Start commit: `142647b` on `main`.

The push-only cap assigned different weights to the same reply at the two signs. The replacement uses a sigmoid of the four-NLL margin, with gradients at both signs. PCGrad is removed. Whole-completion checkpointing replays the correct signed context without retaining four full-vocabulary graphs. The existing top-256 KL penalty is unchanged.

SHOULD: independent even-in-C terms for cho and rej contribute exactly zero gradient to the preference objective. All four per-pair NLL gradients have the same absolute weight. The checkpointed production step should match direct, uncheckpointed gradients for LoRA and PiSSA, with KL enabled and disabled.

## Scope

This verifies the loss algebra and tiny real model code path. It does not establish that shortening is fixed on a pretrained model. The other agent's real-model run and checkout are not available in this session.

## Regression command

`PYTHONPATH=src:/workspace/2026/weight-steering-repos/lite/tinymfv/src uv run --no-sync python -m unittest discover -s tests/regression -v`

The carried-over `.venv` pointed at `/home/wassname/.../python3.13`; its ignored interpreter symlink was repaired to the container's Python 3.13.12. `PYTHONPATH` supplies the relocated editable sources. No package versions or supply-chain age limit changed.

```text
warning: Using incompatible environment (`.venv`) due to `--no-sync` (The interpreter in the project environment has a different version (3.13.12) than it was created with (3.13.4))
test_reply_specific_even_terms_cancel_in_value_and_gradient (test_pair_loss.PairLossTests.test_reply_specific_even_terms_cancel_in_value_and_gradient) ... ok
test_same_weight_for_all_four_terms_per_pair (test_pair_loss.PairLossTests.test_same_weight_for_all_four_terms_per_pair) ... ok
test_train_step_matches_direct_coupled_gradient (test_pair_loss.PairLossTests.test_train_step_matches_direct_coupled_gradient) ... 2026-10-10 18:13:06.133 | DEBUG    | cwsteer.adapter:__init__:128 - ModulatedLoRA: 7 targets, r=2, trainable=512
2026-10-10 18:13:07.109 | DEBUG    | cwsteer.adapter:__init__:128 - ModulatedLoRA: 7 targets, r=2, trainable=512
2026-10-10 18:13:07.603 | DEBUG    | cwsteer.adapter:__init__:403 - ModulatedPiSSA: 7 targets, r=2, selection=s_only, trainable=14
2026-10-10 18:13:07.604 | INFO     | cwsteer.adapter:_log_selection_diagnostic:433 - ModulatedPiSSA selection: mean(chosen_idx)=0.5 tail_depth=0.000 across 7 targets (r=2, mode=s_only). SHOULD be tail_depth > 0.05 for activation-driven modes; ≈0 means picks collapsed onto the top of S.
2026-10-10 18:13:07.700 | DEBUG    | cwsteer.adapter:__init__:403 - ModulatedPiSSA: 7 targets, r=2, selection=s_only, trainable=14
2026-10-10 18:13:07.700 | INFO     | cwsteer.adapter:_log_selection_diagnostic:433 - ModulatedPiSSA selection: mean(chosen_idx)=0.5 tail_depth=0.000 across 7 targets (r=2, mode=s_only). SHOULD be tail_depth > 0.05 for activation-driven modes; ≈0 means picks collapsed onto the top of S.
ok
test_zero_strength_and_extreme_margins (test_pair_loss.PairLossTests.test_zero_strength_and_extreme_margins) ... ok

----------------------------------------------------------------------
Ran 4 tests in 3.990s

OK

```

Regression exit status: 0

## Independent review

Claude reviewed the working-tree loss and tests with fresh context:

> No blockers found.

Full review: [coupled-loss-review.md](../../reviews/coupled-loss-review.md). The review completed before the test output was available; the parent subsequently read the full log above and observed `Ran 4 tests in 3.990s` / `OK`. The exact production diff from `142647b` is saved as [train.diff](train.diff); it confirms that the top-K KL calculation and its half-sum coefficient are unchanged.

Residual risks retained rather than hidden:

- The existing KL covers cho/+C and rej/-C only, and ignores mass outside base's top-256 tokens. It does not guarantee bounded likelihood drift. All four NLLs are now logged to expose push-away drift.
- Separate preference/KL gradient diagnostics cost extra checkpoint recomputation. They are retained to identify which term drives an update; no new switch was added.
- The existing checkpoint-selection criterion still uses held-out cho/+C NLL, not the coupled margin. It can select a different step from the margin optimum. Changing selection or tuning KL would be a separate real-model experiment.
- Even-in-C or reply-independent terms cancel; format or length differences genuinely correlated with the pair labels can still be learned.
- CPU regression covers LoRA and PiSSA with KL on/off. The full-pipeline smoke uses bfloat16 on CPU. CUDA memory and pretrained-model shortening are untested.

## Smoke

The complete raw smoke output and dependency diagnosis are in [smoke.md](smoke.md). Its first run passed training/baking and then failed because the carried-over tinymfv checkout predates the API required by cwsteer. The rerun uses the matching dependency revision, explicitly recorded in that log; no training settings were changed.

Observed in the completed rerun:

> C train OK: 10 steps, trained |c1-base|=0.0433
> D bake OK: |baked-base|=0.0486, restore exact ✓
> F c_scan + tinymfv OK: c=0.667, base pmass=1.76e-05, ppx_json_mfv=534169.2 (real tinymfv path)
> cwsteer smoke PASSED — adapter (LoRA + PiSSA), train, bake/restore, pole-gen + logit-filter, c_scan calibration all green

The command exited 0. This checks that the actual pipeline executes, including bfloat16 checkpointed training and the required evaluation field. The random-model outputs are gibberish, as expected; neither the calibration strength nor the scores demonstrate useful steering.

## Delivery checks

`git diff --cached --check` passed for maintained source, tests, README and review notes. Raw smoke output retains logger whitespace and progress-bar carriage returns; the saved patch retains required context prefixes. These two evidence artifacts are excluded from the whitespace check, not rewritten. Remote `main` also has the user's README-only commit `50cd6d9`, absent from this checkout; it will be merged without rewriting or dropping the user's text before a normal push. Harness-generated `.pi/` task state is excluded from the commit.
