# Objective 3 (Chapter 6) - handoff for code review

Federated IoT intrusion detection on ToN-IoT: **DDQN detector + FedProx + record-level differential privacy +
ADP-Meta (adaptive DP with a meta-learned controller)**.

- Repo: `azenelliot-beep/objective3drlidsmetalearningADP-15-CLIENT`, branch `claude/festive-sagan-terov2`.
- This summary describes the code at the commit that adds it; the code itself is unchanged since `d5d4fc0`.
- Status: no training is running. Nothing in this package was trained in this environment, which has no GPU and no
  dataset. All real numbers come from the author's Kaggle/Colab runs.

---------------------------------------------------------------------------------------------------------------------
## 0. Required final framework, and what deviates from it

**Required:** 20 rounds, 15 clients, the agreed preprocessing (section 2), DDQN, FL with FedProx, ADP and meta-learning.

| Experiment (stage tag) | 20 rounds | 15 clients | Agreed preprocessing | DDQN | FedProx | ADP + meta | Verdict |
|---|---|---|---|---|---|---|---|
| **Final, ADP-Meta v2 (`final`)** | yes | yes | yes | yes | yes | yes | **Meets the requirement. Use these as the thesis results.** |
| Calibration (`final`, Block 18d) | yes | yes | yes | yes | yes | yes | Meets it; post-hoc decision threshold only, evaluation-only |
| ADP-Meta v3 (`v3q`) | yes | yes | yes | yes | yes | v3 controller | Meets it (ADP variant); exploratory, 1 seed |
| ADP-Meta v4 (`v4q`) | yes | yes | yes | yes | yes | v4 + gradient projection | Meets it (variant); rejected |
| Drift v5 (`drq`) | yes | yes | no: data arrives in attack waves | yes | yes | v5 controller | **Deviates**; stopped, incomplete |
| **Boost (`bst10`)** | yes | **no: 5** | **no: 5% public slice pre-trained without DP** | yes | yes | yes | **Deviates; do not report as the final framework** |
| **Boost binary (`bstbin`)** | yes | **no: 5** | **no: public slice + binary labels** | yes | yes | yes | **Deviates** |

**WARNING:** the notebooks currently default to `STAGE = "boost"` in Block 18b, which is the non-compliant setting.
The required framework is `STAGE = "final"`. Changing the default is left to the review.

---------------------------------------------------------------------------------------------------------------------
## 1. Files

| Path | What it is |
|---|---|
| `original/before-viva_1.ipynb` | Author's notebook as received (baseline for diffs). Contained a leaked label, DP noise commented out, and a FedProx term divided by the parameter count; it reported 99.68% accuracy. |
| `cells/*.py` | **Source of truth**: every new or replaced notebook cell. |
| `tools/build_notebook.py` | Builds the notebooks from `original/` + `cells/` (it replaces Blocks 4, 16, 17 and 18, inserting 16b, 17b-17f and 18a-18d + 19). |
| `notebooks/chapter6_v4.ipynb` | Kaggle notebook (built). |
| `notebooks/chapter6_v4_colab.ipynb` | Colab notebook (built): reads data from and writes results to Google Drive. |
| `notebooks/chapter6_v4_eps16.ipynb` | Variant: STAGE `eps16`. |
| `notebooks/paste_cells_18c_18d_19.py` | Older paste-in cells for a live session; superseded. |
| `tests/` | Synthetic-data tests that execute the **built** notebook: 93 tests, last full run passed. |
| `CHANGELOG.md` | Every change, decision and result, in order. |
| `ABLATION_PLAN.md`, `RESULTS_TEMPLATE.md`, `results.md` | Plan, template, and the early results log (results.md section 2 predates the fixes). |
| `results/*.md`, `results/sweep1_val_output.txt` | Results as pasted by the author. |
| `results/measured_results.csv` | All measured numbers in one table, transcribed from the files and outputs named in its `source` column. |

**Per-cell content (all inside the notebook):**

| Block(s) | Content |
|---|---|
| 1-3 | Environment, imports, device and seeds |
| 4-8b | Load the data; drop leak/shortcut columns; de-duplicate; encode; 70/10/20 split; QuantileTransformer; ExtraTrees feature selection |
| 9-15 | Chapter 5 non-private pipeline (resampling, Dirichlet split, DDQN, FL client/server, plots); reference only |
| 16 / 16b | DP-SGD client (per-sample clipping, Gaussian noise, Poisson sampling, Opacus RDP accountant) and a GPU-batched version of the same maths (`FAST=True`) |
| 17 | ADP-Meta v1 |
| 17b | ADP-Meta v2 (budget pacing; **used in final**) |
| 17c | v3 |
| 17d | v4 |
| 17e | v5 drift |
| 17f | Boost scenarios |
| 18a | CFG, methods, run_dp_pipeline, model selection, saving |
| 18b | Stages |
| 18c | Validation summary |
| 18d | Weak-class check + threshold calibration |
| 19 | Guarded final test |

**Not included:** the dataset, Kaggle/Drive outputs, model checkpoints (`*_best.pt`, `*_ckpt.pt`) and the per-run
result JSONs. The JSONs for the boost runs are in the author's Drive (`MyDrive/Elham/results_ch6`); their numbers are
in `measured_results.csv`.

---------------------------------------------------------------------------------------------------------------------
## 2. What is implemented (and checked by tests)

**Data and protocol** (Blocks 4-8b; numbers from `results.md`):
- ToN-IoT `Train_Test_Network` CSV, 10 classes: normal, backdoor, ddos, dos, injection, mitm, password, ransomware,
  scanning, xss.
- Dropped `label` (a copy of the target) and `ts`, `src_ip`, `dst_ip`, `src_port`, `dst_port`.
- De-duplicated after dropping: 271,369 exact duplicates and 629 conflicting rows removed.
- Label-encoded categoricals.
- Stratified 70/10/20 split (seed 42): train 132,331 / val 18,904 / test 37,810.
- QuantileTransformer and ExtraTrees feature selection (29 features) fitted on train only.
- **No resampling in the DP pipeline**; resampling is used only by the Chapter 5 reference runs.
- Non-IID clients: Dirichlet α = 2, at least 100 records per client, seeded.

**Detector:**
- `SmallQNetwork` 29-64-32-10 (LayerNorm, ReLU), about 4.5k parameters, wrapped in the DDQN agent.
- The action is the predicted class, and the reward depends on whether the prediction is correct.
- Class-weighted reward, false-alarm penalty `REWARD_FA`.
- Trained with the TD loss (smooth-L1) only, never cross-entropy.
- γ = 0 under DP, so each step is effectively a contextual bandit (see section 5).

**Federation:** FedProx proximal term (μ/2)·||w − w_g||², μ = 0.01; all clients take part every round
(`CLIENT_FRAC = 1.0`); 20 rounds × 400 local steps.

**Differential privacy:**
- Record-level local DP-SGD in every client: per-sample gradient clipping to C, Gaussian noise σ·C, Poisson sampling
  (expected batch 256).
- C = `CLIP_MULT` × median per-sample gradient norm, measured on validation data only (C = 3.927 in final).
- Opacus RDP accountant, δ = 1e-5. The budget is checked before every round, so a round is never run if it would
  exceed ε.
- Reported ε = the total spent over all rounds, not ε at the selected round.

**ADP-Meta v2** (Block 17b, `ADP_ACTIONS="pacing"`):
- A REINFORCE policy (32-16 hidden units) picks a pacing multiplier k ∈ {0.5, 0.75, 1, 1.5, 2} each round.
- Each round spends min(1, k / rounds_left) of the client's *remaining* RDP budget (at a fixed RDP order); σ is found
  by bisection, rounding towards more noise. k = 1 every round reproduces fixed noise exactly.
- State (5 values): round progress t/T, budget left, validation macro-F1, change in macro-F1, and log(1 + median
  client update norm). Reward: the round's validation macro-F1 gain.
- REINFORCE with Monte-Carlo returns and a per-round baseline.

**Meta-learning (C6):**
- The controller is trained for `ADP_META_EPISODES = 10` episodes per ε in `META_EPS = (8, 3, 1)`.
- Training uses proxy federations built from validation half A, with the reward measured on validation half B.
- The controller is then frozen; the real clients never train it, and it never sees the test set.

**Model selection and test protocol:**
- The selected model is the round with the highest validation macro-F1 subject to validation FAR ≤ 10%; otherwise the
  round with the lowest validation FAR, recorded as `cap_met = False`.
- The test set is evaluated once (Block 19, `FINAL_TEST = True`, filtered by `TEST_TAG`).

**What the 93 tests check** (synthetic data that mimics ToN-IoT's class imbalance; they run the built notebook):
- Budget is never exceeded; the accountant matches Opacus.
- The fast GPU path equals the per-client loop.
- No record sits in two clients; the test set is never used in Blocks 17b-17f.
- Selection rule, stage configs and calibration maths.
- The drift masks never let inactive records influence training.
- Boost scenarios: exact restore of the original data, public slice disjoint from the clients, RNG not shifted, budget
  respected.

The tests do **not** check accuracy on real data.

---------------------------------------------------------------------------------------------------------------------
## 3. Exact configurations behind the reported results

**CFG defaults** (`cells/block18a_experiment_defs.py`):
```
ROUNDS=20, LOCAL_STEPS=400, BATCH=256, LR=1e-3, CLIENT_FRAC=1.0, MU=0.01, DELTA=1e-5, FAR_CAP=0.10,
PACE_ACTIONS=(0.5,0.75,1.0,1.5,2.0), ADP_LR=1e-2, ADP_ENTROPY=0.01, PROXY_MIN_PER_CLIENT=50,
WEIGHTS_FROM="train", REWARD_POWER=0.5, REWARD_FA=0.0, REWARD_MISS=0.0, CLIP_MULT=1.0, FAST=True,
META_EPS=(8.0,3.0,1.0), BINARY=False, NUM_CLIENTS_DP=None (=15), PUBLIC_FRAC=0.0, PRETRAIN_STEPS=2000
NUM_CLIENTS=15, ALPHA=2.0, MIN_PER_CLIENT=100 (Block 10); SEED / SPLIT_SEED = 42
```

**Overrides per stage** (`cells/block18b_run_experiments.py`; every stage starts from the defaults):

| Stage | Overrides | Seeds | Runs |
|---|---|---|---|
| `final` | `FINAL_CFG = CLIP_MULT=0.5, REWARD_FA=0.5, ADP_ACTIONS="pacing", ADP_META_EPISODES=10` | 0, 1, 2 | fedavg_np, fedprox_np, dp_fixed and dp_adpmeta at ε 8 / 3 / 1 |
| `v3_quick` | `REWARD_FA/CLIP_MULT` as final, `ADP_ACTIONS="pacing_clip"`, `ADP_META_EPISODES=12`, `META_EPS=(8,)`, `TWIN_POOL=2`, `META_TIME_LIMIT_MIN=180` | 0 | dp_fixed, dp_adpmeta at ε 8 |
| `v4_quick` | `ADP_ACTIONS="pacing_clip_rank"`, 12 episodes, `META_EPS=(8,)`, `TWIN_POOL=2`, `META_TIME_LIMIT_MIN=150` | 0 | dp_fixed, dp_fixed_proj, dp_adpmeta at ε 8 |
| `drift_quick` | v3 settings + `DRIFT=True`, `ADP_ACTIONS="drift"`, 12 episodes, `META_EPS=(8,)` | 0 | dp_fixed, dp_adpmeta at ε 8 |
| `boost` | `FINAL_CFG + NUM_CLIENTS_DP=5, PUBLIC_FRAC=0.05` (`bst10`), and the same + `BINARY=True` (`bstbin`) | 0 | dp_fixed, dp_adpmeta at ε 8 |

Method definitions are in `METHODS_DP_PIPELINE` (18a):
- `fedavg_np`: μ = 0, no clipping, no noise.
- `fedprox_np`: μ = 0.01, no clipping, no noise.
- `dp_fixed`: the same σ every round.
- `dp_adpmeta`: σ paced by the frozen controller.

All methods use the same reward, network, data and seeds, so every comparison is paired by seed.

---------------------------------------------------------------------------------------------------------------------
## 4. Measured results

Full table: `results/measured_results.csv`.

### 4a. Required framework (stage `final`) - TEST, seeds 0/1/2, mean ± std
| Method | ε | Acc | Macro-F1 | Weighted F1 | FAR | DR |
|---|---|---|---|---|---|---|
| FedAvg, no DP (baseline) | - | 95.20 ± 0.05 | 81.74 ± 0.26 | 95.41 ± 0.04 | 1.51 ± 0.29 | 97.82 ± 0.41 |
| FedProx, no DP (baseline) | - | 92.80 ± 0.44 | 75.99 ± 1.24 | 92.97 ± 0.36 | 2.01 ± 0.20 | 95.38 ± 0.93 |
| Fixed-noise DP + FedProx | 8 | 80.51 ± 3.19 | 46.27 ± 6.41 | 78.73 ± 4.81 | 8.55 ± 0.43 | 80.86 ± 7.21 |
| **ADP-Meta v2 + FedProx (thesis model)** | 8 | 76.47 ± 3.84 | 39.53 ± 6.52 | 72.65 ± 4.40 | 7.47 ± 1.21 | 70.16 ± 6.23 |
| Fixed-noise DP | 3 | 71.74 ± 3.53 | 31.28 ± 4.75 | 65.85 ± 5.12 | 5.13 ± 0.62 | 54.89 ± 9.41 |
| ADP-Meta v2 | 3 | 68.57 ± 2.08 | 24.22 ± 6.26 | 60.86 ± 3.14 | 4.84 ± 0.32 | 48.47 ± 5.51 |
| Fixed-noise DP | 1 | 66.38 ± 1.38 | 18.20 ± 2.60 | 56.92 ± 0.83 | 3.42 ± 1.76 | 33.18 ± 7.54 |
| ADP-Meta v2 | 1 | 66.08 ± 0.92 | 17.57 ± 2.05 | 56.51 ± 0.41 | 3.38 ± 1.69 | 31.78 ± 8.56 |

**ADP-Meta minus fixed noise, paired by seed (test, macro-F1):**

| ε | Difference | Seeds ADP better |
|---|---|---|
| 8 | −6.74 ± 9.18 | 1/3 |
| 3 | −7.05 ± 6.15 | 0/3 |
| 1 | −0.63 ± 1.16 | 2/3 |

The pre-registered rule (ADP ≥ fixed + 2 validation macro-F1) is **not met**. Per-class F1 values are in
`results/final_v2_TEST_results.md`. Validation and test agree within about 0.5 points.

**Calibration** (same models, seed 0, ε 8, post-hoc threshold for FAR ≈ 10%, test):

| Method | DR | Macro-F1 | FAR |
|---|---|---|---|
| Fixed noise, before → after | 79.21 → 84.21 | 49.62 → 50.62 | 8.08 → 10.37 |
| ADP-Meta, before → after | 73.25 → 74.51 | 46.21 → 44.75 | 6.25 → 10.44 |

### 4b. Experiments that change the required framework (report separately, never as the final framework)
| Stage | Deviation | Set | Method | Acc | Macro-F1 | FAR | DR | Seeds |
|---|---|---|---|---|---|---|---|---|
| `bst10` | 5 clients + 5% public pre-training without DP | test | fixed = ADP-Meta (identical) | 86.88 | 63.67 | 9.40 | 94.58 | 1 |
| `bstbin` | same + binary labels | test | fixed | 97.79 | 97.63 | 2.58 | 98.42 | 1 |
| `bstbin` | same + binary labels | test | ADP-Meta | 97.82 | 97.67 | 2.56 | 98.48 | 1 |
| `drq` | attack waves (non-stationary) | val | fixed | 72.29 | 35.79 | 4.83 | 61.78 | 1 (ADP run stopped) |

### 4c. ADP variants inside the required framework (validation, seed 0, ε 8)
| Variant | ADP Macro-F1 | Fixed Macro-F1 | Difference |
|---|---|---|---|
| v3, pacing + adaptive clipping | 43.05 | 49.01 | −5.96 |
| v4, + public-subspace projection | 23.23 | 49.01 | −25.77 (rejected) |
| v1, online noise levels (early, different settings) | | | about −4.4 |

---------------------------------------------------------------------------------------------------------------------
## 5. Known problems (for the reviewer)
1. **ADP-Meta has never beaten tuned fixed noise.** In the static setting every round sees the same data mix, so
   spending the budget evenly is near-optimal, and ADP can only move noise between rounds. On boost the controller
   chose k = 1.0 in every round, which is the fixed schedule, so the results are identical. Uneven spending has
   collapsed minority classes: password in v2, backdoor in v3.
2. **Weak classes:**
   - backdoor: about 50% of its rows are predicted as scanning;
   - mitm: about 85% of its rows are predicted as normal;
   - ransomware: 20 validation rows.
   All of these are near zero under DP, for every method.
3. **Unaccounted label statistics:** `WEIGHTS_FROM="train"` builds the reward class weights (and `class_weights_dp`)
   from the pooled label counts of the private training data. Those counts are not covered by ε. The validation-count
   option (`W_val`) was neutral in Sweep 1 (+0.4) but was not adopted. This should be switched or disclosed.
4. **γ = 0 under DP:** each sample is one step, so the DDQN is effectively a contextual bandit with a target network.
   This is defensible but must be stated; the non-private γ = 0.1 run showed no difference.
5. **Boost caveats:**
   - The public slice (6,616 records) has no DP protection.
   - Binary results come mostly from the non-private pre-training: the best round was round 1.
   - Only 1 seed; `boost3` has not been run.
   - There is no "pre-training only" baseline and no "FedProx without DP" baseline in the boost setting.
6. **FAR is near the cap:** the 10-class boost FAR of 9.4% sits just under the 10% selection cap.
7. **Variance:** ε 8 runs vary ±3-6 macro-F1 points across seeds (classes appear or disappear between seeds), so
   single-seed comparisons are unreliable.
8. **Non-private baseline differs from the DP runs:** FedProx without DP (92.8%) is below FedAvg without DP (95.2%)
   at μ = 0.01, 20 rounds.
9. **The default stage is `boost`, not `final`** (see section 0).

**Unsuccessful or abandoned experiments:**

| Experiment | Outcome |
|---|---|
| v1 | −4.4 |
| v2 | −6.7 on test |
| v3 | −6.0 |
| v4 | −25.8, rejected |
| v5 drift | controller did not converge in 12 episodes; run stopped |
| Calibration | helps fixed noise only |
| Boost | lifts both methods equally, no ADP gain |
| Sweep 1 / Sweep 2 | larger batches, MU 0 / 0.1, LR 3e-3, MISS 0.5, FA 1.0 and CLIP 0.25 did not beat `CLIP_MULT=0.5` with `REWARD_FA=0.5`; see CHANGELOG |

---------------------------------------------------------------------------------------------------------------------
## 6. Commands to reproduce

**Local checks** (CPU, synthetic data, no dataset needed):
```
pip install -r tests/requirements.txt        # torch, opacus==1.6.0, numpy, pandas, scikit-learn, imbalanced-learn, ...
python tools/build_notebook.py               # rebuild notebooks/ from original/ + cells/
python -m pytest tests -q                    # 93 tests, about 23 min on CPU
```

**Real runs** (Kaggle T4/P100 with the ToN-IoT network dataset attached, or Colab with the CSV at
`/content/drive/MyDrive/Elham/TON_IoT_Train_Test_Network.csv`):
1. In Block 18b, set `STAGE = "final"` (the required framework).
   - Others: `"boost"`, `"boost3"`, `"v3_quick"`, `"v4_quick"`, `"drift_quick"`, `"calib"`, `"eps16"`.
   - Finished runs are skipped, so a re-run resumes.
2. Run all cells. Block 18c prints the validation tables.
3. Block 18d: set `CALIB_TAG = "final"`. It prints the weak-class check and calibration on validation; set
   `CALIB_TEST = True` to apply it to the test set.
4. Block 19, once only: set `FINAL_TEST = True` and `TEST_TAG = "final"`. `TEST_TAG` takes a comma list, for example
   `"bst10,bstbin"`.

Results are written as `results/<method>_eps<e>__<tag>__seed<k>.json` (with `_best.pt` / `_TEST.json` / `_CALIB.json`)
plus CSV tables. Versions used: opacus 1.6.0; torch 2.x on Kaggle/Colab (the Colab cell writes `env_versions.json`).

---------------------------------------------------------------------------------------------------------------------
## 7. Rules for changes
- Keep DDQN, FedProx, record-level DP-SGD + RDP accounting, ADP-Meta and meta-learning active.
- Required final framework: 15 clients, 20 rounds.
- Validation only for any choice; the test set is used once, in Block 19.
- Report the ε actually spent over all rounds; disclose any public or non-private data.
- Workflow: edit `cells/*.py`, then `python tools/build_notebook.py`, then `python -m pytest tests -q`.
