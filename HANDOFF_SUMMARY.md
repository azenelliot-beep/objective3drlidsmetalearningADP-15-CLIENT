# Objective 3 (Chapter 6) - handoff summary of the current best code version

Repo: `azenelliot-beep/objective3drlidsmetalearningADP-15-CLIENT`, branch `claude/festive-sagan-terov2` (commit d5d4fc0).
Notebooks to run: `notebooks/chapter6_v4.ipynb` (Kaggle) and `notebooks/chapter6_v4_colab.ipynb` (Colab, Drive paths).
Source of truth: the `cells/*.py` files; `tools/build_notebook.py` assembles them into the notebooks. Tests: `tests/`
(93 pass on synthetic data; run `python -m pytest tests -q`). Full history of every change and result: `CHANGELOG.md`.

## 1. The model (the four pillars must stay active - nothing disabled, commented out or removed)
| Pillar | What the code does | Where |
|---|---|---|
| DDQN detector | SmallDDQN 29-64-32-10 (~4.5k params); action = predicted class; trained only by reward (TD loss, smooth-L1), gamma 0 under DP | Block 12, 16 |
| FedProx | mu = 0.01, all clients each round, 20 rounds x 400 local steps, batch 256, lr 1e-3 | Block 16b, 18a |
| Differential privacy | Record-level LOCAL DP-SGD in every client: per-sample clipping (C = 0.5 x median val grad norm), Gaussian noise sigma*C, Poisson sampling, Opacus RDP accountant, delta 1e-5, budget checked before every round; reported eps = total spent over all rounds | Block 16, 16b |
| ADP-Meta (adaptive DP + meta-learning) | REINFORCE controller (32-16) chooses per round how much of the remaining RDP budget to spend (pacing, v2); meta-trained on proxy federations built from validation half A, reward on validation half B, then frozen | Block 17b (v2, used in final); 17c-17e = later variants |

## 2. Data and protocol (leakage-free)
- ToN-IoT `Train_Test_Network` CSV, 10 classes (normal + 9 attacks).
- Dropped `label` (copy of target) and `ts`, `src_ip`, `dst_ip`, `src_port`, `dst_port` (shortcut columns), then de-duplicated.
- Stratified 70/10/20 train/val/test split (seed 42); quantile transform and ExtraTrees feature selection (29 features)
  fitted on train only. No resampling in the DP pipeline.
- Clients: Dirichlet alpha = 2 (non-IID).
- Model selection: highest validation macro-F1 with validation FAR <= 10%. Test set used once (Block 19).
- Pre-registered success rule for the thesis claim: ADP-Meta >= fixed noise + 2 validation macro-F1 at FAR <= 10%.

## 3. Current best version = "boost" stage (Block 17f, default `STAGE = "boost"` in Block 18b)
Same pillars and settings, plus three disclosed changes:
1. 5 larger clients instead of 15 (less noise per record at the same eps).
2. A stratified 5% of TRAINING records is a PUBLIC slice: removed from all clients, used once WITHOUT DP to pre-train the
   DDQN (2,000 TD steps); both fixed noise and ADP-Meta start from it. Must be disclosed: those 6,616 records are unprotected.
3. Binary attack/normal task run ALONGSIDE the 10-class task (tags `bst10`, `bstbin`), never instead of it.

## 4. Measured results (TEST set, eps = 8, delta = 1e-5)
| Setting | Method | Acc | Macro-F1 | Weighted F1 | FAR | DR | Seeds |
|---|---|---|---|---|---|---|---|
| Boost, 10-class | Fixed noise | 86.88 | 63.67 | 87.18 | 9.40 | 94.58 | 1 |
| Boost, 10-class | ADP-Meta | 86.88 | 63.67 | 87.18 | 9.40 | 94.58 | 1 |
| Boost, binary | Fixed noise | 97.79 | 97.63 | 97.79 | 2.58 | 98.42 | 1 |
| Boost, binary | ADP-Meta | 97.82 | 97.67 | 97.83 | 2.56 | 98.48 | 1 |
| Original (15 clients) | Fixed noise | 80.51 +/- 3.19 | 46.27 +/- 6.41 | 78.73 | 8.55 | 80.86 | 3 |
| Original (15 clients) | ADP-Meta v2 | 76.47 +/- 3.84 | 39.53 +/- 6.52 | 72.65 | 7.47 | 70.16 | 3 |
| Original, no DP | FedProx | 92.80 +/- 0.44 | 75.99 +/- 1.24 | 92.97 | 2.01 | 95.38 | 3 |
| Original, no DP | FedAvg | 95.20 +/- 0.05 | 81.74 +/- 0.26 | 95.41 | 1.51 | 97.82 | 3 |
Original eps 3 / eps 1 rows and per-class F1: `results/final_v2_TEST_results.md`.

## 5. Honest status
- The boost stage raised BOTH methods (fixed +17.4 macro-F1, ADP +24.2), but ADP-Meta did NOT beat fixed noise:
  on boost the learned controller chose k = 1.0 (even spending) every round, so ADP == fixed exactly on 10-class.
  The pre-registered +2 rule is not met anywhere. Defensible claim: "ADP-Meta matches tuned fixed noise without hand
  tuning", not "improves".
- Binary 97.8% comes mostly from the non-private 5% pre-training: the best round was round 1 and DP training rounds
  did not improve it. Report binary only as a side result.
- Weak classes on boost 10-class test: backdoor F1 6.1, mitm F1 15.3. FAR 9.4% sits near the 10% cap.
- Boost results are 1 seed only (no +/-). `boost3` stage (3 seeds) not run yet (~6-8 h GPU).

## 6. What was tried to make ADP beat fixed noise (all at eps 8, macro-F1 vs fixed)
| Version | Idea | Result |
|---|---|---|
| v1 | online noise level per round | -4.4 (val) |
| v2 | RDP budget pacing + meta-training | -6.7 (test, 3 seeds); tie at eps 1 |
| v3 | pacing + adaptive clipping, twin baseline | -6.0 (val, 1 seed) |
| v4 | public-subspace gradient projection | -25.8 (val): rejected |
| Calibration | detection threshold for FAR 10% (eval only) | helps fixed, not ADP |
| v5 | emerging-attack drift waves + drift-aware controller | stopped; controller did not converge in 12 episodes |
| Boost | 5 clients + 5% public pre-training (+ binary) | 0.00 (10-class), +0.04 (binary) |
Why fixed wins: same total budget (ADP only redistributes noise), rounds see the same data mix so even spending is
near-optimal, noisy controller signal, few meta-training episodes, uneven spending collapses minority classes,
and a carefully tuned fixed baseline.

## 7. Open next steps (suggested)
1. Two missing baselines in the boost setting (seed 0, ~30 min): public pre-training only (no federation) and
   FedProx without DP - needed to show what DP federation adds and a fair privacy cost.
2. `boost3` (3 seeds) before any thesis claim on boost numbers.
3. ADP can only win where rounds genuinely differ (drift / non-stationary data) or where it controls something other
   than the noise split (e.g. per-class or per-client allocation); the static setting gives it nothing to adapt to.
4. Weak classes (backdoor, mitm) and FAR margin.
5. Future work, not implemented: secure aggregation / encryption.

## 8. Rules for anyone changing the code
- Keep DDQN, FedProx, record-level DP-SGD + RDP accounting, ADP-Meta active.
- Never use the test set for training, tuning or selection; validation only, then one Block 19 run.
- Report eps actually spent over all rounds; disclose the public slice.
- Edit `cells/*.py`, then `python tools/build_notebook.py`, then `python -m pytest tests -q`.
