# Ablation plan (validation only)

Baseline for every row = the current accepted configuration, seed 0, eps 8, record-level DP.
Accept if validation macro-F1 >= baseline + 2.0 and FAR <= baseline FAR + 1.0 (selection rule as in
CHANGELOG), then confirm on seed 1 before it becomes the new baseline. The fixed-noise method is tuned
first; ADP-Meta is compared against the *tuned* fixed-noise baseline, never an untuned one.

| Step | Change | Flag (CFG) | Runs (seed 0, eps 8) | Status |
|---|---|---|---|---|
| C0 | Same-pipeline baselines, selection rule, seeds, guarded test | - | fedavg_np, fedprox_np, dp_clip_nonoise, dp_fixed, dp_adpmeta | code done |
| C1 | Reward class weights from validation counts instead of pooled train counts (removes an unprotected statistic). Scaler/feature selection stay fixed and are disclosed. | `WEIGHTS_FROM` | dp_fixed | planned |
| C3 | DP-SGD signal-to-noise: batch {256, 1024, 2048} x steps {100, 200, 400}; then C multiplier {0.5, 1, 2}; then lr {1e-3, 3e-3} | `BATCH`, `LOCAL_STEPS`, `CLIP_MULT`, `LR` | dp_fixed | planned |
| C4 | Cost-sensitive reward: extra penalty lambda_FA for normal->attack, lambda_miss for attack->normal; weight exponent p in {0.5, 0.75, 1} | `REWARD_*` | dp_fixed | planned |
| C11 | FedProx mu in {0, 0.001, 0.01, 0.1} and adaptive mu (driven by validation macro-F1) | `MU`, `MU_SCHEDULE` | dp_fixed | planned |
| C9 | Episode-level DP-DDQN: L in {2, 8, 32} x gamma in {0, 0.1}, target net used, per-episode clipping | `EPISODE_LEN`, `GAMMA` | dp_fixed | planned |
| C5 | ADP-Meta budget pacing in RDP at fixed order, actions = share of remaining budget, reward = validation macro-F1 gain, 32-16 policy, update norm in state | `ADP_ACTIONS` | dp_adpmeta (stage `C5_pacing_online`) | code done |
| C10 | ADP-Meta also picks C multiplier {0.5, 1, 2} | `ADP_CLIP_ACTION` | dp_adpmeta | planned |
| C6 | Meta-train the controller on validation (half A proxies, half B reward), frozen greedy policy | `ADP_META_EPISODES` | dp_adpmeta (stage `C6_meta`) | code done |
| C2 | Class-aware clipping (demoted: results.md shows rare classes survive clipping without noise) | `CLIP_MODE` | dp_fixed | optional |
| C12 | Communication, runtime, per-client fairness, membership-inference attack | - | evaluation only | planned |

C6 decision (user): option (a), practice episodes on validation, disclosed. See CHANGELOG.

Final stage (after all ablations are frozen): Block 18b `STAGE = "final"`, seeds 0/1/2, then Block 19 once.
