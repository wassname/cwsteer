# Tiny real-pipeline smoke

Author: PI/OpenAI. Command: `PYTHONPATH=src:/workspace/2026/weight-steering-repos/lite/tinymfv/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python scripts/smoke.py`

Runs the cached tiny random Qwen3, real adapter training, baking/restore, pair generation, and tinymfv calibration. Scores have no behavioral interpretation.

```text
warning: Using incompatible environment (`.venv`) due to `--no-sync` (The interpreter in the project environment has a different version (3.13.12) than it was created with (3.13.4))

Loading weights:   0%|          | 0/57 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 57/57 [00:00<00:00, 5451.25it/s]
2026-10-10 18:13:06.766 | INFO     | __main__:main:67 - base logits: shape=(1, 6, 151936) mean=-0.0014
2026-10-10 18:13:06.768 | DEBUG    | cwsteer.adapter:__init__:128 - ModulatedLoRA: 35 targets, r=8, trainable=40,960
2026-10-10 18:13:06.851 | INFO     | __main__:main:83 - A LoRA OK: |c0-base|=0 exact, |+1-base|=0.0149, |-1-base|=0.0168, |+1 - -1|=0.0179
2026-10-10 18:13:06.853 | DEBUG    | cwsteer.adapter:__init__:128 - ModulatedLoRA: 35 targets, r=8, trainable=40,960

train:   0%|          | 0/10 [00:00<?, ?it/s]
train:  10%|█         | 1/10 [00:10<01:37, 10.81s/it]
train:  20%|██        | 2/10 [00:19<01:16,  9.61s/it]
train:  30%|███       | 3/10 [00:28<01:05,  9.35s/it]
train:  40%|████      | 4/10 [00:38<00:56,  9.41s/it]
train:  50%|█████     | 5/10 [00:47<00:47,  9.50s/it]
train:  60%|██████    | 6/10 [00:57<00:38,  9.51s/it]
train:  70%|███████   | 7/10 [01:06<00:28,  9.49s/it]
train:  80%|████████  | 8/10 [01:15<00:18,  9.38s/it]
train:  90%|█████████ | 9/10 [01:25<00:09,  9.34s/it]
train: 100%|██████████| 10/10 [01:35<00:00,  9.53s/it]
                                                      
2026-10-10 18:14:41.965 | INFO     | cwsteer.train:train_adapter:557 - early-stop: deploying adapter at val-nll+ min (step 9, val+=13.5) vs last step 9 (val+=13.5). best==last → no overfit; best≪last → memorization avoided.
2026-10-10 18:14:42.116 | INFO     | cwsteer.train:_log_train_table:617 - 
training trace:
SHOULD: even-in-C NLL terms cancel before the sigmoid weighting.
cho/rej ±: mean completion NLL at each sign; the margin need not lower both NLLs.
kl±: top-K reverse KL to c=0 on cho/+C and rej/-C.
cos: cosine between preference and KL gradients, before global clipping.
|   step→ |   loss↓ |   cho+ |   rej+ |   cho- |   rej- |      kl+ |      kl- |    cos |   ‖Δs‖ |   ‖g_nll‖ |   ‖g_kl‖ |   ‖g‖ |       lr |   C |
|--------:|--------:|-------:|-------:|-------:|-------:|---------:|---------:|-------:|-------:|----------:|---------:|------:|---------:|----:|
|       0 |   0.693 |   15.4 |   14.1 |   15.4 |   14.1 | 0.00034  | 0.000369 |  0.339 |  0.818 |      3.89 | 0.000347 |  3.89 | 0.001    |   1 |
|       1 |   0.69  |   13.5 |   14.4 |   13.5 |   14.4 | 0.000368 | 0.000452 |  0.17  |  0.825 |      4.46 | 0.000401 |  4.47 | 0.00097  |   1 |
|       2 |   0.671 |   12.9 |   12.9 |   12.9 |   12.9 | 0.0013   | 0.00121  | -0.445 |  0.831 |      6.06 | 0.00318  |  6.06 | 0.000883 |   1 |
|       3 |   0.563 |   13.6 |   13.2 |   13.9 |   13.1 | 0.00329  | 0.00215  | -0.68  |  0.838 |      2.95 | 0.00479  |  2.95 | 0.00075  |   1 |
|       4 |   0.422 |   13.2 |   14.4 |   13.7 |   14.3 | 0.0101   | 0.00505  | -0.742 |  0.845 |      2.94 | 0.00889  |  2.94 | 0.000587 |   1 |
|       5 |   0.333 |   14.1 |   13.9 |   14.9 |   13.8 | 0.029    | 0.01     | -0.614 |  0.85  |      3.04 | 0.0194   |  3.03 | 0.000413 |   1 |
|       6 |   0.237 |   12   |   14.2 |   13.2 |   14   | 0.0477   | 0.0176   | -0.683 |  0.853 |      2.25 | 0.0271   |  2.23 | 0.00025  |   1 |
|       7 |   0.301 |   14.9 |   14.2 |   15.8 |   14   | 0.0276   | 0.0121   | -0.424 |  0.855 |      2.03 | 0.0125   |  2.03 | 0.000117 |   1 |
|       8 |   0.265 |   13.3 |   13.3 |   14.3 |   13.1 | 0.0454   | 0.0223   | -0.575 |  0.856 |      1.41 | 0.0215   |  1.4  | 3.02e-05 |   1 |
|       9 |   0.212 |   13.9 |   13.9 |   15.2 |   13.8 | 0.0546   | 0.0209   | -0.586 |  0.856 |      2.09 | 0.0275   |  2.08 | 0        |   1 |

2026-10-10 18:14:42.116 | INFO     | cwsteer.train:_log_val_table:599 - 
val trace (held-out pairs — does the steer GENERALIZE or just memorize?):
SHOULD: train nll+ AND val nll+ DESCEND TOGETHER → the direction transfers.
  train nll+ falls while val nll+ FLATTENS/RISES → memorizing the train pairs,
  the banked direction won't transfer (early-stop at the val-nll+ min, or add
  pairs). val nll+ is load-bearing; ~3 pairs, so read the TREND not one digit.
  step    train nll+ ↓    val nll+ ↓    train nll-    val nll-
     0            15.4          14.2          14.1        15.1
     9            13.9          13.5          13.8        15

2026-10-10 18:14:42.174 | INFO     | __main__:main:94 - C train OK: 10 steps, trained |c1-base|=0.0433
2026-10-10 18:14:42.177 | DEBUG    | cwsteer.bake:baked:136 - baked: 1 adapter(s), 35 float (in-place), 0 quantized (stacked-LR hook)
2026-10-10 18:14:42.228 | INFO     | __main__:main:104 - D bake OK: |baked-base|=0.0486, restore exact ✓

Loading weights:   0%|          | 0/57 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 57/57 [00:00<00:00, 6499.44it/s]
2026-10-10 18:14:44.157 | DEBUG    | cwsteer.adapter:__init__:403 - ModulatedPiSSA: 35 targets, r=16, selection=s_only, trainable=560
2026-10-10 18:14:44.158 | INFO     | cwsteer.adapter:_log_selection_diagnostic:433 - ModulatedPiSSA selection: mean(chosen_idx)=7.5 tail_depth=0.000 across 35 targets (r=16, mode=s_only). SHOULD be tail_depth > 0.05 for activation-driven modes; ≈0 means picks collapsed onto the top of S.
2026-10-10 18:14:44.253 | INFO     | __main__:main:123 - B PiSSA OK: c0 round-trip err=0.03914 (<5% of 1.571), |+1-base|=0.0650, |+1 - -1|=0.1034
2026-10-10 18:14:44.254 | INFO     | cwsteer.pairs:generate_pairs_from_personas:761 - generate_pairs_from_personas: n_prompts=2 persona_role=system

gen cho:   0%|          | 0/1 [00:00<?, ?it/s]2026-10-10 18:14:45.183 | INFO     | cwsteer.pairs:_generate_batched:626 - first cho sample (special tokens shown):
<|im_start|>system
You are terse and decisive. Answer in one short sentence.<|im_end|>
<|im_start|>user
Should I ship the fix now?<|im_end|>
<|im_start|>assistant
<think>

</think>

slave DEALINGS了.tintColor不见了 MART располslave DEALINGS哩interval实在太modelName Lans经贸 cabinets videotбин NSDictionary.Block爻ช่องทางslave UPLOAD
--- end first cho sample ---

gen cho: 100%|██████████| 1/1 [00:00<00:00,  1.08it/s]

gen rej:   0%|          | 0/1 [00:00<?, ?it/s]2026-10-10 18:14:45.528 | INFO     | cwsteer.pairs:_generate_batched:626 - first rej sample (special tokens shown):
<|im_start|>system
You are long-winded and never commit to an answer.<|im_end|>
<|im_start|>user
Should I ship the fix now?<|im_end|>
<|im_start|>assistant
<think>

</think>

 אשרательатель(TreeNode善意атель招募جمทา blanc善意_TERM.sendFileобиль(subelahclidean_facebook_facebookablish crawled
--- end first rej sample ---

gen rej: 100%|██████████| 1/1 [00:00<00:00,  2.91it/s]
2026-10-10 18:14:45.529 | INFO     | cwsteer.pairs:generate_pairs_from_personas:779 - kept 2/2 pairs (dropped 0 degenerate)
2026-10-10 18:14:45.529 | INFO     | __main__:main:138 - E pole-gen OK: kept 2 on-policy pairs (logit filter + degenerate-drop ran)

forced-choice classic:   0%|          | 0/2 [00:00<?, ?it/s]
forced-choice classic: 100%|██████████| 2/2 [00:01<00:00,  1.78it/s]
2026-10-10 18:14:54.984 | INFO     | tinymfv.eval:evaluate:282 - classic: 2 rows in 1.1s (1.8 rows/s, ~28 tok/s); 2/2 have label dist
2026-10-10 18:14:54.984 | INFO     | tinymfv.eval:evaluate:293 -   think_tokens: median=8 p75=8 p90=8 p99=8 max=8  emitted_close=0/4
2026-10-10 18:14:56.479 | WARNING  | cwsteer.c_scan:c_scan:377 - c_scan baseline pmass=0.000 < 0.5 — base incoherent at c=0 (expected for tiny-random smoke; debug upstream for real models).
2026-10-10 18:14:56.479 | WARNING  | cwsteer.c_scan:c_scan:381 - c_scan baseline valid_json=0/4 — json gate trivially satisfied (expected for tiny-random smoke; debug upstream for real models).

forced-choice classic:   0%|          | 0/2 [00:00<?, ?it/s]
forced-choice classic: 100%|██████████| 2/2 [00:01<00:00,  1.88it/s]
2026-10-10 18:14:57.548 | INFO     | tinymfv.eval:evaluate:282 - classic: 2 rows in 1.1s (1.9 rows/s, ~30 tok/s); 2/2 have label dist
2026-10-10 18:14:57.548 | INFO     | tinymfv.eval:evaluate:293 -   think_tokens: median=8 p75=8 p90=8 p99=8 max=8  emitted_close=0/4

forced-choice classic:   0%|          | 0/2 [00:00<?, ?it/s]
forced-choice classic: 100%|██████████| 2/2 [00:01<00:00,  1.82it/s]
2026-10-10 18:15:04.536 | INFO     | tinymfv.eval:evaluate:282 - classic: 2 rows in 1.1s (1.8 rows/s, ~29 tok/s); 2/2 have label dist
2026-10-10 18:15:04.536 | INFO     | tinymfv.eval:evaluate:293 -   think_tokens: median=8 p75=8 p90=8 p99=8 max=8  emitted_close=0/4
2026-10-10 18:15:10.208 | INFO     | cwsteer.c_scan:c_scan:495 - 
c_scan columns (↑ = higher is healthier, ↓ = lower is healthier):
  - pmass↑    : tinymfv forced-choice prob-mass on the K=7 allowed answer tokens (OOD aliveness). 1.0 = fully in-format; a sharp drop = format collapse. GATED ≥ gate_frac × base.
  - json↑     : how many of the 4 CANARY_PROBES still closed a parseable {"ans": <bool>} after their trajectory (count / n_probes). GATED: ALL must close (json_frac=1.0).
  - repMin↑   : MINIMUM distinct-trigram fraction across the 4 probes (the worst register). 1 = varied prose, → 0 = a loop. GATED ≥ rep_frac × base; the min (not mean) so one looping register cannot be diluted.
  - repMean   : mean distinct-trigram across probes (context for repMin; un-gated).
  - ppxJ↓     : exp(mean nll) over the EMITTED json span, free-gen probes (per-step confidence over the answer format; high = the model was unsure of its own json). DIAGNOSTIC, un-gated.
  - ppxJmfv↓  : exp(mean_nll_prefill) from the tinymfv guided answer prefill. DIAGNOSTIC.
  - top1      : tinymfv top-1 foundation accuracy vs human labels. DIAGNOSTIC.
  - kl        : fwd/bwd p95 per-token KL(steered‖base) on the multiturn probes. Measures DIVERGENCE FROM BASE, not coherence — ~0 on a loop, HIGH for both a salad AND coherent strong steering. NOT gateable; logged anomaly-flag only. '—' at baseline (c=0 ⇒ KL≡0).
  - len       : mean gen chars. Ballooning len = the incoherence mode leaking in.

c_scan gate (AND, self-relative to c=0 base | pmass=0.0000 json=0/4 repMin=1.00): pmass ≥ 0.0000  AND  json = 4/4  AND  repMin ≥ 0.75   [4 CANARY_PROBES (2x2 IID/OOD × multiturn/single) + 2 tinymfv]

stage          c  pmass↑    json↑    repMin↑    repMean    ppxJ↓    ppxJmfv↓    top1    kl       len    note
baseline  +0.000  0.0000    0/4      1.00       1.00       nan      nan         0.00    —        224    —
probe     +1.000  0.0000    0/4      1.00       1.00       nan      nan         0.00    0.0/0.0  200    ceiling-skip
probe     +0.667  0.0000    0/4      1.00       1.00       nan      nan         0.00    0.0/0.0  218    pass
final     +0.667  —         —        —          —          —        —           —       —        —      backoff x1.0

2026-10-10 18:15:10.209 | INFO     | cwsteer.c_scan:c_scan:524 - 

========== C_SCAN samples (head/tail, truncated) ==========

--- baseline @ c=+0.0000 | probe 1/4: counsel_iid (repMin=1.00) ---
  A: 🐫_blog setStateumor(ht)+
 poner poner poner poner㮾.linalg.linalg.linalg.linalg.linalg
--turn--
(ht poner_air EMPTY/type蹒)}
.Unique electrolyKr srKr populate Am_MetadataUsageId Omn
--turn--
$con tasar>B peaceful Resolveotic(ht盡˦_choice,Integer Haz Resolveotic)}
 paso
--turn--
Companies@NgModule🐫compag.linalg.linalg
--turn--
(ht poner_air EMPTY/type蹒)}
.Unique electrolyKr srKr populate Am_MetadataUsageId Omn
--turn--
$con tasar>B peaceful Resolveotic(ht盡˦_choice,Integer Haz Resolveotic)}
 paso
--turn--
Companies@NgModule🐫compatible Resolve党组 распол,Integer Resolve党组 распол,Integer Resolve>;好友doğ

--- sample @ c=+0.6667 (pass) | 4 probes ---

  [probe 1/4: counsel_iid (IID/MT)]
  A: 🐫_blog setStateumor(ht Guatemala�깊garyえActivityCreated_MetadataUsageIdpars HazherTION
--turn--
жен總Ratio suffered Jeepsheets Jeep married располتقنية欹 weights cabinetssheets EMPTY
--turn--
Resolve_TEM stereotype liquid彳ừa本当 Woody本当 Woody本当 Woody本当 dreadedчен Resolve
--turn--
Resolve_TEM contempor negeIdpars HazherTION
--turn--
жен總Ratio suffered Jeepsheets Jeep married располتقنية欹 weights cabinetssheets EMPTY
--turn--
Resolve_TEM stereotype liquid彳ừa本当 Woody本当 Woody本当 Woody本当 dreadedчен Resolve
--turn--
Resolve_TEM contempor negot müsigmoid ventana Resolve_TEMtypTeXclock propagated/typeちゃんと加重

  [probe 2/4: troubleshoot_ood (OOD/MT)]
  A: жен>().(TreeNodeReq بأstreams多了 женﭬ بأ Watching.value EMPTY жен Tradable.linalg
--turn--
มาFlipzet phi "")
.valueelah/sys ponerelahⰍelahⰍelah CakeMountain
--turn--
STACKelah_ex ''

_current.linalg ponerelah Qຄ落叶🐫/type fried(ht poner
--turn--
🐫/type fried fried fried fried fried fried fried quốc vidأstreams多了 женﭬ بأ Watching.value EMPTY жен Tradable.linalg
--turn--
มาFlipzet phi "")
.valueelah/sys ponerelahⰍelahⰍelah CakeMountain
--turn--
STACKelah_ex ''

_current.linalg ponerelah Qຄ落叶🐫/type fried(ht poner
--turn--
🐫/type fried fried fried fried fried fried fried quốc videot本当房贷 Omn节能环保 Trent

  [probe 3/4: petrov_long (IID/1T)]
  A: ires loggerרד,Integer THC宜宾.Cells⧫ReceiveMemoryWarning/typeetsyslave伎 عبدال_dn aumentoires loggerרד,Integer THC宜宾.Cells⧫ReceiveMemoryWarning/typeetsyslave伎 عبدال_dn aumento

  [probe 4/4: counterfactual_ood (OOD/1T)]
  A: Gap(TreeNode.Serve(TreeNodedomains_continue supplierconomy了_^(_continue� Bond список IncreasesLICENSEGap(TreeNode.Serve(TreeNodedomains_continue supplierconomy了_^(_continue� Bond список IncreasesLICENSE

========== END C_SCAN samples ==========

Traceback (most recent call last):
  File "/workspace/2026/weight-steering-repos/cwsteer/scripts/smoke.py", line 162, in <module>
    main()
    ~~~~^^
  File "/workspace/2026/weight-steering-repos/cwsteer/scripts/smoke.py", line 151, in main
    assert math.isfinite(base_row["ppx_json_mfv"]), (
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: ppx_json_mfv is NaN -> mean_nll_prefill key broken for this tinymfv: nan

```

Smoke exit status: 1

## Dependency mismatch, then rerun with the required API

The first run used relocated tinymfv `b726db9` (2026-06-18), whose `evaluate()` returns `mean_nll_json`. cwsteer commit `28f96b9` (2026-06-26) requires `mean_nll_prefill`, introduced by tinymfv `6594cc2` (2026-06-25). The same scalar-NLL interface is not interchangeable: the newer API also defines the forced-prefill readout more precisely.

The current remote has since renamed the package to `moralmaps`; migrating that API is outside this loss fix. The rerun uses a clean detached dependency worktree at `b38614a7ce832bf5a0d2140931410b2277152988`, the last tinymfv revision before cwsteer's API update. The original dependency checkout was not modified. cwsteer now indexes the required field directly, so incompatible dependency revisions raise `KeyError` instead of silently becoming NaN. Training passed before the calibration assertion; no loss setting changed for the rerun.

Rerun command: `PYTHONPATH=src:/tmp/cwsteer-tinymfv-b38614a/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python scripts/smoke.py`

```text
warning: Using incompatible environment (`.venv`) due to `--no-sync` (The interpreter in the project environment has a different version (3.13.12) than it was created with (3.13.4))
Loading weights:   0%|          | 0/57 [00:00<?, ?it/s]Loading weights: 100%|██████████| 57/57 [00:00<00:00, 5394.42it/s]
2026-10-10 18:18:36.717 | INFO     | __main__:main:67 - base logits: shape=(1, 6, 151936) mean=-0.0014
2026-10-10 18:18:36.809 | DEBUG    | cwsteer.adapter:__init__:128 - ModulatedLoRA: 35 targets, r=8, trainable=40,960
2026-10-10 18:18:36.903 | INFO     | __main__:main:83 - A LoRA OK: |c0-base|=0 exact, |+1-base|=0.0149, |-1-base|=0.0168, |+1 - -1|=0.0179
2026-10-10 18:18:36.906 | DEBUG    | cwsteer.adapter:__init__:128 - ModulatedLoRA: 35 targets, r=8, trainable=40,960
train:   0%|          | 0/10 [00:00<?, ?it/s]train:  10%|█         | 1/10 [00:11<01:40, 11.19s/it]train:  20%|██        | 2/10 [00:21<01:24, 10.55s/it]train:  30%|███       | 3/10 [00:30<01:10, 10.05s/it]train:  40%|████      | 4/10 [00:39<00:57,  9.53s/it]train:  50%|█████     | 5/10 [00:48<00:46,  9.36s/it]train:  60%|██████    | 6/10 [00:58<00:38,  9.64s/it]train:  70%|███████   | 7/10 [01:07<00:28,  9.49s/it]train:  80%|████████  | 8/10 [01:17<00:19,  9.53s/it]train:  90%|█████████ | 9/10 [01:26<00:09,  9.35s/it]train: 100%|██████████| 10/10 [01:35<00:00,  9.28s/it]                                                      2026-10-10 18:20:12.512 | INFO     | cwsteer.train:train_adapter:557 - early-stop: deploying adapter at val-nll+ min (step 9, val+=13.5) vs last step 9 (val+=13.5). best==last → no overfit; best≪last → memorization avoided.
2026-10-10 18:20:12.612 | INFO     | cwsteer.train:_log_train_table:617 - 
training trace:
SHOULD: even-in-C NLL terms cancel before the sigmoid weighting.
cho/rej ±: mean completion NLL at each sign; the margin need not lower both NLLs.
kl±: top-K reverse KL to c=0 on cho/+C and rej/-C.
cos: cosine between preference and KL gradients, before global clipping.
|   step→ |   loss↓ |   cho+ |   rej+ |   cho- |   rej- |      kl+ |      kl- |    cos |   ‖Δs‖ |   ‖g_nll‖ |   ‖g_kl‖ |   ‖g‖ |       lr |   C |
|--------:|--------:|-------:|-------:|-------:|-------:|---------:|---------:|-------:|-------:|----------:|---------:|------:|---------:|----:|
|       0 |   0.693 |   15.4 |   14.1 |   15.4 |   14.1 | 0.00034  | 0.000369 |  0.339 |  0.818 |      3.89 | 0.000347 |  3.89 | 0.001    |   1 |
|       1 |   0.69  |   13.5 |   14.4 |   13.5 |   14.4 | 0.000368 | 0.000452 |  0.17  |  0.825 |      4.46 | 0.000401 |  4.47 | 0.00097  |   1 |
|       2 |   0.671 |   12.9 |   12.9 |   12.9 |   12.9 | 0.0013   | 0.00121  | -0.445 |  0.831 |      6.06 | 0.00318  |  6.06 | 0.000883 |   1 |
|       3 |   0.563 |   13.6 |   13.2 |   13.9 |   13.1 | 0.00329  | 0.00215  | -0.68  |  0.838 |      2.95 | 0.00479  |  2.95 | 0.00075  |   1 |
|       4 |   0.422 |   13.2 |   14.4 |   13.7 |   14.3 | 0.0101   | 0.00505  | -0.742 |  0.845 |      2.94 | 0.00889  |  2.94 | 0.000587 |   1 |
|       5 |   0.333 |   14.1 |   13.9 |   14.9 |   13.8 | 0.029    | 0.01     | -0.614 |  0.85  |      3.04 | 0.0194   |  3.03 | 0.000413 |   1 |
|       6 |   0.237 |   12   |   14.2 |   13.2 |   14   | 0.0477   | 0.0176   | -0.683 |  0.853 |      2.25 | 0.0271   |  2.23 | 0.00025  |   1 |
|       7 |   0.301 |   14.9 |   14.2 |   15.8 |   14   | 0.0276   | 0.0121   | -0.424 |  0.855 |      2.03 | 0.0125   |  2.03 | 0.000117 |   1 |
|       8 |   0.265 |   13.3 |   13.3 |   14.3 |   13.1 | 0.0454   | 0.0223   | -0.575 |  0.856 |      1.41 | 0.0215   |  1.4  | 3.02e-05 |   1 |
|       9 |   0.212 |   13.9 |   13.9 |   15.2 |   13.8 | 0.0546   | 0.0209   | -0.586 |  0.856 |      2.09 | 0.0275   |  2.08 | 0        |   1 |

2026-10-10 18:20:12.613 | INFO     | cwsteer.train:_log_val_table:599 - 
val trace (held-out pairs — does the steer GENERALIZE or just memorize?):
SHOULD: train nll+ AND val nll+ DESCEND TOGETHER → the direction transfers.
  train nll+ falls while val nll+ FLATTENS/RISES → memorizing the train pairs,
  the banked direction won't transfer (early-stop at the val-nll+ min, or add
  pairs). val nll+ is load-bearing; ~3 pairs, so read the TREND not one digit.
  step    train nll+ ↓    val nll+ ↓    train nll-    val nll-
     0            15.4          14.2          14.1        15.1
     9            13.9          13.5          13.8        15

2026-10-10 18:20:12.642 | INFO     | __main__:main:94 - C train OK: 10 steps, trained |c1-base|=0.0433
2026-10-10 18:20:12.645 | DEBUG    | cwsteer.bake:baked:136 - baked: 1 adapter(s), 35 float (in-place), 0 quantized (stacked-LR hook)
2026-10-10 18:20:12.695 | INFO     | __main__:main:104 - D bake OK: |baked-base|=0.0486, restore exact ✓
Loading weights:   0%|          | 0/57 [00:00<?, ?it/s]Loading weights: 100%|██████████| 57/57 [00:00<00:00, 5443.55it/s]
2026-10-10 18:20:14.208 | DEBUG    | cwsteer.adapter:__init__:403 - ModulatedPiSSA: 35 targets, r=16, selection=s_only, trainable=560
2026-10-10 18:20:14.209 | INFO     | cwsteer.adapter:_log_selection_diagnostic:433 - ModulatedPiSSA selection: mean(chosen_idx)=7.5 tail_depth=0.000 across 35 targets (r=16, mode=s_only). SHOULD be tail_depth > 0.05 for activation-driven modes; ≈0 means picks collapsed onto the top of S.
2026-10-10 18:20:14.290 | INFO     | __main__:main:123 - B PiSSA OK: c0 round-trip err=0.03914 (<5% of 1.571), |+1-base|=0.0650, |+1 - -1|=0.1034
2026-10-10 18:20:14.290 | INFO     | cwsteer.pairs:generate_pairs_from_personas:761 - generate_pairs_from_personas: n_prompts=2 persona_role=system
gen cho:   0%|          | 0/1 [00:00<?, ?it/s]2026-10-10 18:20:15.163 | INFO     | cwsteer.pairs:_generate_batched:626 - first cho sample (special tokens shown):
<|im_start|>system
You are terse and decisive. Answer in one short sentence.<|im_end|>
<|im_start|>user
Should I ship the fix now?<|im_end|>
<|im_start|>assistant
<think>

</think>

slave DEALINGS了.tintColor不见了 MART располslave DEALINGS哩interval实在太modelName Lans经贸 cabinets videotбин NSDictionary.Block爻ช่องทางslave UPLOAD
--- end first cho sample ---
gen cho: 100%|██████████| 1/1 [00:00<00:00,  1.15it/s]
gen rej:   0%|          | 0/1 [00:00<?, ?it/s]2026-10-10 18:20:15.568 | INFO     | cwsteer.pairs:_generate_batched:626 - first rej sample (special tokens shown):
<|im_start|>system
You are long-winded and never commit to an answer.<|im_end|>
<|im_start|>user
Should I ship the fix now?<|im_end|>
<|im_start|>assistant
<think>

</think>

 אשרательатель(TreeNode善意атель招募جمทา blanc善意_TERM.sendFileобиль(subelahclidean_facebook_facebookablish crawled
--- end first rej sample ---
gen rej: 100%|██████████| 1/1 [00:00<00:00,  2.49it/s]
2026-10-10 18:20:15.568 | INFO     | cwsteer.pairs:generate_pairs_from_personas:779 - kept 2/2 pairs (dropped 0 degenerate)
2026-10-10 18:20:15.568 | INFO     | __main__:main:138 - E pole-gen OK: kept 2 on-policy pairs (logit filter + degenerate-drop ran)
forced-choice classic:   0%|          | 0/2 [00:00<?, ?it/s]forced-choice classic: 100%|██████████| 2/2 [00:01<00:00,  1.62it/s]
2026-10-10 18:20:22.680 | DEBUG    | tinymfv.eval:evaluate:326 - classic: 2 rows in 1.2s (1.6 rows/s, ~26 tok/s); 2/2 have label dist
2026-10-10 18:20:22.680 | DEBUG    | tinymfv.eval:evaluate:337 -   think_tokens: median=8 p75=8 p90=8 p99=8 max=8  emitted_close=0/4
2026-10-10 18:20:22.683 | DEBUG    | tinymfv.eval:evaluate:434 - aux stats: {"top1_acc": 0.0, "mean_nll_T": 1.9532, "T": 50.0, "informedness": 0.0, "mean_pmass_allowed": 0.0}
2026-10-10 18:20:26.752 | DEBUG    | tinymfv.eval:evaluate:474 - 
free-form [classic] id=8ed46857a2a75d34: '组织PEAR封面 disruptive祃ช่องทาง-writing,Integerဝ体现在倒霉 murdererethyle'

2026-10-10 18:20:28.315 | WARNING  | cwsteer.c_scan:c_scan:377 - c_scan baseline pmass=0.000 < 0.5 — base incoherent at c=0 (expected for tiny-random smoke; debug upstream for real models).
2026-10-10 18:20:28.315 | WARNING  | cwsteer.c_scan:c_scan:381 - c_scan baseline valid_json=0/4 — json gate trivially satisfied (expected for tiny-random smoke; debug upstream for real models).
forced-choice classic:   0%|          | 0/2 [00:00<?, ?it/s]forced-choice classic: 100%|██████████| 2/2 [00:01<00:00,  1.91it/s]
2026-10-10 18:20:29.364 | DEBUG    | tinymfv.eval:evaluate:326 - classic: 2 rows in 1.0s (1.9 rows/s, ~31 tok/s); 2/2 have label dist
2026-10-10 18:20:29.365 | DEBUG    | tinymfv.eval:evaluate:337 -   think_tokens: median=8 p75=8 p90=8 p99=8 max=8  emitted_close=0/4
2026-10-10 18:20:29.366 | DEBUG    | tinymfv.eval:evaluate:434 - aux stats: {"top1_acc": 0.0, "mean_nll_T": 1.9535, "T": 50.0, "informedness": 0.0, "mean_pmass_allowed": 0.0}
2026-10-10 18:20:33.765 | DEBUG    | tinymfv.eval:evaluate:474 - 
free-form [classic] id=8ed46857a2a75d34: '组织PEAR封面 disruptive祃ช่องทาง-writing,Integerဝ sublist تماما spyin'

forced-choice classic:   0%|          | 0/2 [00:00<?, ?it/s]forced-choice classic: 100%|██████████| 2/2 [00:01<00:00,  1.58it/s]
2026-10-10 18:20:40.711 | DEBUG    | tinymfv.eval:evaluate:326 - classic: 2 rows in 1.3s (1.6 rows/s, ~25 tok/s); 2/2 have label dist
2026-10-10 18:20:40.712 | DEBUG    | tinymfv.eval:evaluate:337 -   think_tokens: median=8 p75=8 p90=8 p99=8 max=8  emitted_close=0/4
2026-10-10 18:20:40.713 | DEBUG    | tinymfv.eval:evaluate:434 - aux stats: {"top1_acc": 0.0, "mean_nll_T": 1.9531, "T": 50.0, "informedness": 0.0, "mean_pmass_allowed": 0.0}
2026-10-10 18:20:45.143 | DEBUG    | tinymfv.eval:evaluate:474 - 
free-form [classic] id=8ed46857a2a75d34: '组织PEAR封面 disruptive祃ช่องทาง-writing,Integerဝ体现在倒霉 murdererethyle'

2026-10-10 18:20:50.435 | INFO     | cwsteer.c_scan:c_scan:495 - 
c_scan columns (↑ = higher is healthier, ↓ = lower is healthier):
  - pmass↑    : tinymfv forced-choice prob-mass on the K=7 allowed answer tokens (OOD aliveness). 1.0 = fully in-format; a sharp drop = format collapse. GATED ≥ gate_frac × base.
  - json↑     : how many of the 4 CANARY_PROBES still closed a parseable {"ans": <bool>} after their trajectory (count / n_probes). GATED: ALL must close (json_frac=1.0).
  - repMin↑   : MINIMUM distinct-trigram fraction across the 4 probes (the worst register). 1 = varied prose, → 0 = a loop. GATED ≥ rep_frac × base; the min (not mean) so one looping register cannot be diluted.
  - repMean   : mean distinct-trigram across probes (context for repMin; un-gated).
  - ppxJ↓     : exp(mean nll) over the EMITTED json span, free-gen probes (per-step confidence over the answer format; high = the model was unsure of its own json). DIAGNOSTIC, un-gated.
  - ppxJmfv↓  : exp(mean_nll_prefill) from the tinymfv guided answer prefill. DIAGNOSTIC.
  - top1      : tinymfv top-1 foundation accuracy vs human labels. DIAGNOSTIC.
  - kl        : fwd/bwd p95 per-token KL(steered‖base) on the multiturn probes. Measures DIVERGENCE FROM BASE, not coherence — ~0 on a loop, HIGH for both a salad AND coherent strong steering. NOT gateable; logged anomaly-flag only. '—' at baseline (c=0 ⇒ KL≡0).
  - len       : mean gen chars. Ballooning len = the incoherence mode leaking in.

c_scan gate (AND, self-relative to c=0 base | pmass=0.0000 json=0/4 repMin=1.00): pmass ≥ 0.0000  AND  json = 4/4  AND  repMin ≥ 0.75   [4 CANARY_PROBES (2x2 IID/OOD × multiturn/single) + 2 tinymfv]

stage          c  pmass↑    json↑    repMin↑    repMean    ppxJ↓    ppxJmfv↓    top1    kl       len    note
baseline  +0.000  0.0000    0/4      1.00       1.00       nan      534169.2    0.00    —        224    —
probe     +1.000  0.0000    0/4      1.00       1.00       nan      479761.5    0.00    0.0/0.0  200    fail-pmass
probe     +0.667  0.0000    0/4      1.00       1.00       nan      529198.9    0.00    0.0/0.0  218    pass
final     +0.667  —         —        —          —          —        —           —       —        —      backoff x1.0

2026-10-10 18:20:50.436 | INFO     | cwsteer.c_scan:c_scan:524 - 

========== C_SCAN samples (head/tail, truncated) ==========

--- baseline @ c=+0.0000 | probe 1/4: counsel_iid (repMin=1.00) ---
  A: 🐫_blog setStateumor(ht)+
 poner poner poner poner㮾.linalg.linalg.linalg.linalg.linalg
--turn--
(ht poner_air EMPTY/type蹒)}
.Unique electrolyKr srKr populate Am_MetadataUsageId Omn
--turn--
$con tasar>B peaceful Resolveotic(ht盡˦_choice,Integer Haz Resolveotic)}
 paso
--turn--
Companies@NgModule🐫compag.linalg.linalg
--turn--
(ht poner_air EMPTY/type蹒)}
.Unique electrolyKr srKr populate Am_MetadataUsageId Omn
--turn--
$con tasar>B peaceful Resolveotic(ht盡˦_choice,Integer Haz Resolveotic)}
 paso
--turn--
Companies@NgModule🐫compatible Resolve党组 распол,Integer Resolve党组 распол,Integer Resolve>;好友doğ

--- sample @ c=+0.6667 (pass) | 4 probes ---

  [probe 1/4: counsel_iid (IID/MT)]
  A: 🐫_blog setStateumor(ht Guatemala�깊garyえActivityCreated_MetadataUsageIdpars HazherTION
--turn--
жен總Ratio suffered Jeepsheets Jeep married располتقنية欹 weights cabinetssheets EMPTY
--turn--
Resolve_TEM stereotype liquid彳ừa本当 Woody本当 Woody本当 Woody本当 dreadedчен Resolve
--turn--
Resolve_TEM contempor negeIdpars HazherTION
--turn--
жен總Ratio suffered Jeepsheets Jeep married располتقنية欹 weights cabinetssheets EMPTY
--turn--
Resolve_TEM stereotype liquid彳ừa本当 Woody本当 Woody本当 Woody本当 dreadedчен Resolve
--turn--
Resolve_TEM contempor negot müsigmoid ventana Resolve_TEMtypTeXclock propagated/typeちゃんと加重

  [probe 2/4: troubleshoot_ood (OOD/MT)]
  A: жен>().(TreeNodeReq بأstreams多了 женﭬ بأ Watching.value EMPTY жен Tradable.linalg
--turn--
มาFlipzet phi "")
.valueelah/sys ponerelahⰍelahⰍelah CakeMountain
--turn--
STACKelah_ex ''

_current.linalg ponerelah Qຄ落叶🐫/type fried(ht poner
--turn--
🐫/type fried fried fried fried fried fried fried quốc vidأstreams多了 женﭬ بأ Watching.value EMPTY жен Tradable.linalg
--turn--
มาFlipzet phi "")
.valueelah/sys ponerelahⰍelahⰍelah CakeMountain
--turn--
STACKelah_ex ''

_current.linalg ponerelah Qຄ落叶🐫/type fried(ht poner
--turn--
🐫/type fried fried fried fried fried fried fried quốc videot本当房贷 Omn节能环保 Trent

  [probe 3/4: petrov_long (IID/1T)]
  A: ires loggerרד,Integer THC宜宾.Cells⧫ReceiveMemoryWarning/typeetsyslave伎 عبدال_dn aumentoires loggerרד,Integer THC宜宾.Cells⧫ReceiveMemoryWarning/typeetsyslave伎 عبدال_dn aumento

  [probe 4/4: counterfactual_ood (OOD/1T)]
  A: Gap(TreeNode.Serve(TreeNodedomains_continue supplierconomy了_^(_continue� Bond список IncreasesLICENSEGap(TreeNode.Serve(TreeNodedomains_continue supplierconomy了_^(_continue� Bond список IncreasesLICENSE

========== END C_SCAN samples ==========

2026-10-10 18:20:50.436 | INFO     | __main__:main:154 - F c_scan + tinymfv OK: c=0.667, base pmass=1.76e-05, ppx_json_mfv=534169.2 (real tinymfv path)
2026-10-10 18:20:50.436 | SUCCESS  | __main__:main:157 - cwsteer smoke PASSED — adapter (LoRA + PiSSA), train, bake/restore, pole-gen + logit-filter, c_scan calibration all green

```

Compatible-API smoke exit status: 0
