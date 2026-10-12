# CHANGELOG - Chapter 6 performance work

Rules: one change at a time, validated on the VALIDATION set only (seed 0, eps 8 unless stated), kept only if
it beats the current baseline by >= 2 macro-F1 points without raising FAR by more than 1 point, and the gain
is confirmed on a second seed. Failures are recorded here too. The test set is used once, at the end (Block 19).

## Decisions fixed before any new test result (2026-10-11)
- **Threat model:** the validation set is server-held public data. It may set the clip norm C, drive the
  ADP-Meta reward and select the round. It is never used to train the detector. The test set is untouched
  until Block 19.
- **Best-round rule (all methods):** highest validation macro-F1 among rounds with validation FAR <= 10%.
  If no round meets the cap, take the round with the lowest validation FAR and record `cap_met = False`.
- **Seeds:** the split, scaler and feature selection stay fixed (seed 42). Seeds 0, 1, 2 vary the training
  seed, Dirichlet client split, Poisson sampling, ADP-Meta init and (Chapter 5 runs) resampling. All methods
  use the same seeds, so comparisons are paired. Report mean ± std (sample std).
- **Reported epsilon** = total spent over ALL rounds run (under local DP the server sees every update), not
  epsilon at the selected round.
- **C6 decision:** meta-training episodes run on validation data, disclosed in the thesis. Validation is
  split once into stratified halves (seed 42): proxy detectors train on half A, controller reward on half B.
  The reported detector never trains on validation data, and the test set is never used.
- **Dropped:** public-data warm-up of the detector (it would train on validation data, which the threat model
  now rules out).

## C0 - Fair baselines, selection rule, seeds, guarded test (code done, awaiting Kaggle run)
Files: `cells/block16_dp_client.py`, `cells/block18a_experiment_defs.py`, `cells/block18b_run_experiments.py`,
`cells/block18c_val_summary.py`, `cells/block19_final_test.py` -> `notebooks/chapter6_v4.ipynb`.

Why:
- Earlier non-private numbers (92.79 / 94.99%) came from a different pipeline: resampled data, 42.6k DDQN,
  gamma 0.1, 50 rounds, 80% participation. The latest DP run used all data, the small DDQN, gamma 0 and
  20 rounds. That gap mixed the effect of DP with everything else.
- Round selection by validation accuracy picked round 9 (macro-F1 27.8) over round 16 (36.9) in the last run.
- Single seed; test evaluation mixed into the training cell.

What changed:
- `fedavg_np` / `fedprox_np`: same pipeline as DP, with no clipping and no noise. `dp_clip_nonoise`:
  clipping but sigma = 0. `dp_fixed` / `dp_adpmeta`: as before. `fedavg_ch5` / `fedprox_ch5`: Chapter 5
  pipeline, kept as a reference.
- DPClient: sigma = 0 -> no accountant step and epsilon reported as `None`, never a number. Any non-private
  step voids the client's epsilon. Noise without clipping is refused.
- Pre-round budget check for every private method, plus an assert after each round that no client
  exceeded the budget.
- One clip norm C for all methods and seeds: the median per-sample gradient norm on validation at a
  seed-42 initialisation (same rule as before).
- Results go to `results/<run>__seed<k>.json` + `_best.pt`. Runs resume (finished ones are skipped). No test
  numbers are written until Block 19 with `FINAL_TEST = True`.
- ADP-Meta controller, reward, actions, DP-SGD hyperparameters: **unchanged** (C0 is infrastructure only).
- Fixed: Block 16 comment said each record is used "ONCE" (it is Poisson-sampled many times; the
  accountant counted this correctly). Fixed: unclosed JSON file handle (ResourceWarning).

Tests: `tests/test_c0.py` (16 tests, synthetic data, executes the built notebook).

Validation effect: _pending Kaggle run (stage `C0_baseline`)._

## C5 - ADP-Meta v2: budget pacing + macro-F1 reward (code done, runs AFTER fixed-noise tuning)
Flag: `CFG["ADP_ACTIONS"] = "pacing"` (default `"mult"` = unchanged v1). File: `cells/block17b_adpmeta_v2.py`.

Why: v1 used actions 0.8/0.9 (less noise than calibrated), which front-loaded spending. Round 1 alone used
36% of eps = 8, and the budget ran out at round 16. Its reward was accuracy, which hides rare-class failure.
results.md: v1 tied fixed noise at eps 8 and collapsed at eps 1 (macro-F1 8.1 vs 13.1, FAR 95%).

What:
- Pacing in RDP at a fixed order alpha* per client, which adds up over rounds (pacing on eps does not:
  eps grows roughly with the square root of the number of steps). Action k spends min(1, k / rounds_left)
  of the remaining budget. k = 1 every round equals fixed noise exactly. The last round spends what is
  left. sigma is found by bisection, rounded towards more noise. The Opacus pre-round check and post-round
  assert still apply.
- Policy 32-16 (Chapter 6). State: t/T, budget left, val macro-F1, change in macro-F1, log(1 + median
  update norm). Reward: validation macro-F1 gain.
- Without C6 the controller learns online (one REINFORCE step per round), as v1 did.

Tests: uniform pacing = fixed sigma; random action sequences never exceed eps and spend >= 95% of it;
the 32-16 policy learns a toy bandit; online run respects the budget.

Validation effect: _pending._

## C6 - Meta-training ADP-Meta v2 on validation (code done, runs after C5)
Flag: `CFG["ADP_META_EPISODES"] = N` (> 0), requires `ADP_ACTIONS = "pacing"`.

What:
- N practice episodes, each a full federated run with the real round structure (rounds, local steps,
  batch, C, mu, reward weights, DP-SGD client).
- The 15 proxy clients come from a Dirichlet split of validation half A (new split each episode).
- Proxy client k follows real client k's noise schedule, so per-step noise sigma*C/B matches the real run.
- Reward measured on half B. REINFORCE with returns and a per-round baseline.
- The controller is saved per eps budget (`results/controller_eps<e>__<TAG>.pt`, plus `_meta_log.json`),
  frozen, used greedily in the real run, and shared across seeds 0/1/2.

Caveats:
- Half A has about 9.4k rows, so each proxy record is sampled about 10-20x more often than a real one.
  Noise per step matches, but proxy runs can overfit sooner, and transfer of the learned policy is an
  empirical question. That is why C6 must beat both C5 and tuned fixed noise on validation.
- Cost: one practice episode is about one real run (~15 min on a T4), so 10 episodes x 3 eps budgets is
  about 7.5 h.

Tests: proxies only ever hold validation half A, real clients never hold validation rows, halves are
disjoint, controller is cached and reused across seeds, budget respected.

Validation effect: _pending._

## Infrastructure
- `CFG["TAG"]` is appended to run names. Re-running an existing run name with a different config raises
  an error instead of silently skipping, so ablation results cannot be mixed up.

## Sweep 1 - candidate fixes for the fixed-noise DP model, one Run All (code done, awaiting Kaggle run)
Block 18b now defaults to `STAGE = "auto"`: the C0 baselines, then 12 variants of `dp_fixed` at eps 8 (seed 0),
each changing ONE knob (batch and steps count as one knob, "batch size at a fixed budget").
Block 18c prints `SWEEP vs baseline` with dMacroF1, dFAR and a verdict
(IMPROVES = +2 macro-F1 or more and FAR not worse by more than 1 point).
- C1 `WEIGHTS_FROM="val"`: reward weights from validation counts
- C3 `BATCH`/`LOCAL_STEPS` (1024x100, 1024x200, 2048x100), `CLIP_MULT` (0.5, 2), `LR` 3e-3
- C4 `REWARD_FA` 0.5 (false alarm), `REWARD_MISS` 0.5 (missed attack), `REWARD_POWER` 1.0
- C11 `MU` 0 and 0.1
Defaults reproduce the previous behaviour exactly (tested). Winners are confirmed on seed 1, then combined.

Validation effect (2026-10-11, seed 0, eps 8, per-client loop, selection rule as fixed):
| run | val macro-F1 | val Acc | val FAR | cap met | verdict |
|---|---|---|---|---|---|
| dp_fixed (baseline) | 26.83 | 61.00 | 23.64 | no | - |
| CLIPx0.5 | 49.10 (+22.3) | 79.39 | 15.09 | no | IMPROVES |
| FA0.5 | 39.85 (+13.0) | 77.60 | 5.62 | yes | IMPROVES |
| MISS0.5 | 46.10 (+19.3) | 72.31 | 26.56 | no | no (FAR +2.9) |
| B1024_S200 / B2048_S100 / B1024_S100 | 39.17 / 36.12 / 34.18 | | 25.9-26.8 | no | no (FAR up) |
| MU0 / LR3e-3 / W_val | 28.16 / 27.87 / 27.27 | | | no | no (< +2) |
| POW1.0 / MU0.1 / CLIPx2 | 24.57 / 23.35 / 20.52 | | | no | no (worse) |
| dp_adpmeta v1 (C0) | 22.41 | 59.32 | 19.61 | no | worse than fixed noise |
| dp_clip_nonoise / fedprox_np / fedavg_np | 71.11 / 73.38 / 80.96 | 90.98 / 88.97 / 94.39 | 5.35 / 8.60 / 3.07 | yes | references |

Findings: noise is still the main cost (clip-only 71.1 vs fixed DP 26.8). A smaller C reduces the noise
(std sigma*C) more than it loses signal. The false-alarm penalty brings FAR under the cap. FedAvg > FedProx
without privacy again, and mu = 0 is no worse under DP, so the thesis cannot claim a FedProx accuracy gain.
W_val (weights from validation counts) is neutral (+0.4), so it can be adopted for the integrity reason
alone. Raw output: `results/sweep1_val_output.txt`.
Next ("next" stage): BASE2 = CLIPx0.5 + FA0.5 (fixed reference), ADP-Meta v2 online and with meta-learning
on BASE2, and refinements CLIP0.25, FA1.0, MISS0.5, B1024x200 (one change each vs BASE2).

## Default stage = "main" (thesis model only)
At the user's request the default Run All now trains ONLY FedProx + DDQN + local record-level DP + ADP-Meta
(eps 8, seed 0): ADP-Meta v1 (before) and ADP-Meta v2 / C5 (after), about 30 min. Block 18c prints a
BEFORE / AFTER table. The sweep and baselines stay available as optional stages ("auto", "sweep1",
"C0_baseline"). The fixed-noise comparison is still part of the "final" stage, because Chapter 6 must show
that ADP-Meta beats fixed noise.

## Speed - all clients trained in parallel on the GPU (Block 16b, `CFG["FAST"] = True`)
Why: the 15 clients were trained one after another on a 4.5k-parameter network, so the T4 sat mostly idle.
A run took about 12-15 min, and the full plan about 16-17 GPU hours.
What: `FastFederation` stacks the 15 clients' parameters and runs each local DP-SGD step for all of them in
one batched computation. Per client it does exactly the same thing: own Poisson rate, per-sample
clipping, own noise sigma_k*C, FedProx, Adam. Adam is element-wise, so one Adam over the stacked tensors
equals 15 independent Adams. Accounting stays per client, step by step. Falls back to the loop when not all
clients train in a round.
Tests: parameters identical to the per-client loop to 1e-5 (with and without clipping and FedProx, over
2 rounds); accountant histories identical; noise std = sigma_k*C/(q_k n_k); noise without clipping refused.
Speed: no gain on CPU (compute-bound). The GPU gain is NOT measured yet. Each run's JSON records `minutes`.
Note: runs on the fast path draw random numbers in a different order than the loop, so they are not
bit-identical to earlier runs. Use one path consistently for every comparison (all new stages use it).

## Sweep 2 - ADP-Meta v2 on the Sweep-1 winners (validation, seed 0, eps 8, fast path)
BASE2 = clip x0.5 + false-alarm penalty 0.5 (C = 3.927).
| run | val macro-F1 | val Acc | val FAR | cap met | vs reference |
|---|---|---|---|---|---|
| dp_fixed BASE2 (reference) | **49.01** | 80.74 | 7.58 | yes | - |
| CLIP0.25 | 47.76 | 80.19 | 7.71 | yes | -1.2 (no) |
| **ADP-Meta v2 + meta-learning (C6)** | 45.98 | 79.97 | 5.91 | yes | -3.0 (no) |
| FA1.0 | 41.33 | 79.61 | 5.35 | yes | -7.7 (no) |
| ADP-Meta v2 online (C5) | 40.67 | 79.24 | 5.10 | yes | -8.3 (no) |
| MISS0.5 | 39.73 | 78.94 | 7.30 | yes | -9.3 (no) |
| B1024x200 | 31.36 | 71.52 | 4.48 | yes | -17.7 (no) |

Findings:
- BASE2 fixed noise is the best private model so far: +22 macro-F1, +20 accuracy points, FAR from 23.6% to
  7.6% vs the original fixed-noise baseline.
- Meta-learning improves ADP-Meta by +5.3 over online learning (C6 vs C5) and beats v1 by a wide margin
  (46.0 vs 22.4, but on different settings).
- ADP-Meta (C6) is 3.0 below fixed noise on this single seed. That is within the single-seed noise seen so
  far (classes swap between runs), so it is not settled. Decision: no further ADP-specific tuning (it
  would bias the comparison). The question goes to the 3-seed paired comparison.
- None of the refinements beat BASE2, so the settings are FROZEN: CLIP_MULT 0.5, REWARD_FA 0.5, same reward
  for every method, ADP-Meta v2 with 10 meta-episodes per eps.

## Final stage (default in Block 18b)
Seeds 0/1/2 x {fedavg_np, fedprox_np, dp_fixed, dp_adpmeta} x eps {8, 3, 1}, tag "final". Block 18c adds a
paired ADP-vs-fixed validation table. Block 19 (FINAL_TEST = True, run once) evaluates only "final" runs
and adds a paired ADP-vs-fixed test table (mean ± std of the difference, seeds where ADP is better).
Chapter 5 reference rows moved to an optional stage "ch5_reference".

## ADP-Meta v3 (Block 17c, `ADP_ACTIONS = "pacing_clip"`) - user-approved redesign to try to beat fixed noise
Context: v2 + meta-learning matched but did not beat tuned fixed noise (seed 0: 46.0 vs 49.0). The user allowed
model changes as long as DDQN, FedProx, ADP and meta-learning stay. The test set has NOT been used (Block 19 held).
Changes 1-5:
1. Joint action each round: budget-pacing share k AND clip multiplier m (C_t = C_base * m, m in {0.5, 1, 2}).
   Privacy depends only on sigma (noise std = sigma * C_t), so accounting is unchanged (tested). Fixed noise
   keeps its best tuned constant C (x0.5), so the baseline stays at full strength.
2. Twin-baseline meta-training: each practice episode is compared with a fixed-schedule twin (same proxy
   split, init and sampling seed). Twins are cached per (seed, eps). The controller samples from its own RNG,
   so the training RNG streams of an episode and its twin stay aligned until their schedules differ.
3. Reward = episode score under the model-selection rule (best F1 with FAR <= cap, else F1 minus FAR excess).
4. Richer state: adds FAR, previous sigma, previous clip multiplier and log10(eps).
5. 24 practice episodes. ONE controller is shared across eps 8/3/1 (eps is in the state), so meta-training is
   paid once.
Policy: 32-16 trunk, two heads, initial bias towards (k = 1, m = 1), i.e. it starts from the fixed schedule. The
REINFORCE baseline is initialised from the first episode (a zero start caused premature collapse in a toy test).
Pre-registered success rule: v3 >= fixed + 2 val macro-F1 with FAR <= 10% at an eps (stage "v3_val", seed 0,
eps 8 and 1); then "final_v3" (3 seeds, eps 8/3/1): mean better AND better on >= 2 of 3 seeds. Otherwise
report no significant difference.

## Final stage results (ADP-Meta v2, 3 seeds, validation) - see results/final_v2_val_summary.md
Pre-registered rule NOT met: ADP-Meta v2 minus fixed noise = -6.4 ± 8.7 (eps 8), -6.6 ± 5.9 (eps 3),
-0.7 ± 1.2 (eps 1) val macro-F1. ADP has slightly lower FAR everywhere. Reported as found.
Test evaluation (Block 19, TEST_TAG = "final") is to be run once. ADP-Meta v3 code stays frozen as committed
(cd74dad) and will not be changed after the test results are seen.

## FINAL TEST (Block 19, run once, TEST_TAG = "final") - results/final_v2_TEST_results.md
Test agrees with validation (e.g. fixed eps 8: test 46.27 vs val 46.00 macro-F1; ADP eps 8: 39.53 vs 39.65).
ADP-Meta v2 minus fixed noise on test: -6.74 ± 9.18 (eps 8), -7.05 ± 6.15 (eps 3), -0.63 ± 1.16 (eps 1).
These are the Chapter 6 numbers. ADP-Meta v3 stays frozen at cd74dad; if it is run later, report it separately.

## Follow-up stages (after the main test evaluation) - report separately and disclose
- `eps16`: ADP-Meta v2 vs fixed noise at eps 16 (weaker privacy), frozen final settings, 3 seeds. One more
  point on the privacy-utility curve.
- `v3_val` / `final_v3`: ADP-Meta v3 (frozen at cd74dad, before the test results were seen).
Any test numbers from these stages are a second, disclosed use of the test set. Model selection stays on
validation with the same rule.

## Follow-up eps 16 (validation) - results/followup_eps16_val_summary.md
ADP-Meta v2 46.65 ± 11.29 vs fixed 45.86 ± 6.24 macro-F1 (paired +0.79 ± 5.06, ADP better 2/3 seeds): equivalent.
No accuracy gain from eps 8 to eps 16 for fixed noise. My earlier estimate of 83-87% accuracy at eps 16 was wrong
(observed ~81%).

## Block 20 - post-processing analysis of saved models (follow-up, no retraining, no extra privacy cost)
(a) Binary IDS view (attack vs normal) next to the 10-class results; (b) per-class ROC-AUC from Q-values;
(c) per-class Q-value offsets learned on VALIDATION only (max val macro-F1 with FAR <= cap), applied to test.
The DDQN weights are unchanged; post-processing of a DP model keeps the same (eps, delta). Test numbers from
Block 20 are a disclosed follow-up use of the test set.
