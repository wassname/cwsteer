I have what I need. Writing the review.

# Coupled preference-loss review — `cwsteer` @ main (baseline 142647b)

Reviewer: reviewer-anthropic (Claude). Read-only; no shell. The parent did not supply a `git diff` or the regression output; I reviewed the current working-tree files only. Statements about the *baseline* (push-only cap, PCGrad, per-pole projection) are therefore inferences from docstrings and `slop/verification/coupled-loss/run.md`, not observed.

## Verdict

**No blockers found.** Three non-blocking concerns with small fixes, plus residual risks that the regression cannot settle.

## Observed (files read in full)

- `src/cwsteer/train.py:138-145` — `_odd_pair_loss`:
  `margin = C * ((nll_cho_n - nll_cho_p) - (nll_rej_n - nll_rej_p)); return softplus(-margin).mean()`
- `src/cwsteer/train.py:249-268` — four `checkpoint(completion_terms, ±C, …, use_reentrant=False)` calls, `completion_terms` re-enters `with lora(model, c=c)` inside the checkpointed region; KL attached only to cho/+C and rej/−C.
- `src/cwsteer/train.py:269-281` — two separate `autograd.grad` calls (nll with `retain_graph=use_kl`, then KL), summed into `p.grad`; `cos` is diagnostic only (no projection).
- `src/cwsteer/train.py:436,538` — history gate `lambda: lora._c != 0.0`, restored to `lambda: True` after training.
- `src/cwsteer/adapter.py:150-172, 441-460` — both adapters' `__call__` set `_c`, register hooks, remove them in `finally`, reset `_c=0.0`, reject re-entry.
- `tests/regression/test_pair_loss.py` — 4 tests; `run.md` is truncated mid-code-block (run ongoing; no pass/fail evidence).

## Mathematical check

**Signs — correct.** Desired state: ℓ_cho(+C) < ℓ_cho(−C) (first bracket > 0) and ℓ_rej(−C) < ℓ_rej(+C) (second bracket < 0, subtracted → positive). So M > 0 is "good" and `softplus(-M)` is minimised by growing M. ∂L/∂ℓ = (C·σ(−M)/B)·(+1, −1, −1, +1) for (cho+, rej+, cho−, rej−): lowers cho+ and rej−, raises rej+ and cho−. `test_same_weight_for_all_four_terms_per_pair` asserts exactly this pattern and magnitude, so a sign flip of either bracket fails the test.

**Both ±C receive gradients — yes**, by construction (all four NLLs are live graph tensors; nothing is detached). The test's `torch.autograd.grad(loss, nlls)` without `allow_unused` would raise if any of the four were detached, and `test_train_step_matches_direct_coupled_gradient` compares the full checkpointed gradient to an un-checkpointed direct one, so a detach inside `train_step` would also fail. Regression sensitivity to the two named failure modes is good.

**Cancellation — exact but narrower than the prose suggests.** M is a difference-of-differences, so it exactly cancels (a) any term that is even in C for a given reply, and (b) any term that is reply-independent at a given sign. `test_reply_specific_even_terms_cancel…` verifies (a) (I recomputed: M = 4C²·odd, g_even = 0 ✓). What it does *not* cancel: a reply-dependent, odd-in-C component — which is precisely what a genuine length/formatting difference between cho and rej in the data looks like. If the teacher's cho edits are systematically shorter than rej, the "odd direction" the loss rewards *is* a shortening axis. **This is the key distinction**: the fix removes the *artefactual* shortening from unequal per-sign weighting (mathematical cancellation); it does not and cannot remove shortening that is in the pair data (empirical). Only a real-model generation-length check can distinguish these; `run.md` already scopes this out ("does not establish that shortening is fixed on a pretrained model").

## Non-blocking concerns

1. **Unanchored "push-away" sides** — `train.py:259-262`. KL is applied to cho/+C and rej/−C only. With the cap removed, the cheapest way to grow M is to inflate ℓ_rej(+C) and ℓ_cho(−C) without bound (DPO-style degeneration); σ(−M) saturation slows but does not prevent this. The old cap (per `run.md`) bounded this; nothing replaces it.
   *Smallest fix*: compute KL on all four forwards. Zero extra base cost — `base_top_logp_p/idx_p` is per-sequence, so pass it to the cho/−C call and `base_top_logp_n/idx_n` to the rej/+C call; average the four. Alternatively just monitor: in the trace, `rej+`/`cho−` climbing steeply while `cho+`/`rej−` are flat is the signature.

2. **Double recompute when KL is on** — `train.py:269-273`. Non-reentrant checkpoint recomputes per autograd graph task, so the split `autograd.grad(loss_nll, retain_graph=True)` then `autograd.grad(kl)` re-runs the two KL-bearing forwards *and* their backwards a second time, solely to produce the `cos`/`‖g‖` diagnostics. Estimated step cost ≈ 2 base fwd + 4 fwd + 6 recompute fwd + 6 bwd vs. 10 fwd + 4 bwd for a single combined grad. Peak memory is unaffected (one segment materialised at a time). Correctness is fine (test exercises `kl=0.03`).
   *Smallest fix*: one `autograd.grad(loss_nll + 0.5*kl_lambda*(kl_p+kl_n))` and drop or gate the `cos` diagnostic behind a flag.

3. **Early-stop criterion no longer matches the objective** — `train.py:514-530`. Best-state and patience key on `val_nll+` (cho at +C) falling. The new loss does not require cho+ to fall (`_log_train_table` says so), and the per-term weight is now ≤ C·σ(−M)/B ≈ half an SFT gradient at M=0, decaying as the margin grows, while `kl_lambda=0.032` was tuned against the old NLL scale. Expect smaller val+ improvements → more early stops/"no improvement" verdicts than before even when the margin is learned. `min_val_improvement` is only reported, not enforced, in this repo.
   *Smallest fix*: also log a val margin (four val NLLs → M) and consider keying patience on it; retune `kl_lambda` on the first real run.

## Checkpointing / hooks / gate — looks correct

- `c` is a positional arg of `completion_terms`, so the recompute re-enters `with lora(model, c=c)` and re-registers hooks with the right sign; `finally` resets `_c=0, _attached=False` after each (re)play, so no re-entry collision between the four segments, and the gate `lora._c != 0.0` evaluates identically in the original forward and the recompute. The c=0 base forwards see the gate closed (pristine base for LoRA; PiSSA gate is a documented no-op).
- Hooks capture the live `nn.Parameter`s, so the recomputed graph connects to `params`. `None` non-tensor args and the shared `zero` return are fine under non-reentrant checkpoint. `preserve_rng_state` default covers dropout (none in Qwen/Gemma anyway).
- Caveat I cannot verify from here: the test runs fp32 on CPU; bf16/CUDA recompute drift is tolerated by non-reentrant checkpoint (no metadata assertion on values), so I expect no failure, but the gradient-match tolerance (`atol=2e-7, rtol=2e-4`) is only evidenced on CPU.

## KL semantics

`_kl_topk_base` and the base top-K computation are unchanged relative to their docstrings; coefficient `0.5*kl_lambda*(kl_p+kl_n)` matches README's "λ_KL · mean(KL_pos, KL_neg)". Whether this is *unchanged vs. 142647b* I cannot confirm — **missing evidence: no diff supplied.**

## Docs

`README.md:106-114` and the `train.py` module docstring state the formula consistently with the code. The README's "shared formatting … cancel" sentence should be read as "sign-independent or reply-independent formatting" — see cancellation note above.

## Checks that could disprove my conclusions

- Regression suite completing (`run.md` currently truncated) — all four tests pass → signs, cancellation, checkpoint equivalence confirmed on tiny model.
- A real-model run trace: `rej+`/`cho−` rising ≫ `cho+`/`rej−` falling → concern 1 is live.
- Generation length at +C vs c=0 on held-out prompts, with cho/rej token-length stats of the training pairs → separates mathematical from empirical shortening.
- `git diff 142647b -- src/cwsteer/train.py` → confirm KL unchanged and what the cap/projection actually did.

— reviewer-anthropic (Claude)