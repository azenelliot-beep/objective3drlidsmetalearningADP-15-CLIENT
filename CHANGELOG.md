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
