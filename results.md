# Results log - corrected pipeline (ToN-IoT network, single seed = 42 unless stated)

All numbers below come from the corrected notebook. Test set was used only once per run, with the
best-validation-round model. Validation = 18,904 rows, test = 37,810 rows, train = 132,331 rows.

## 0. Pipeline corrections already applied
- Removed leaked column `label` and shortcut columns `ts, src_ip, dst_ip, src_port, dst_port`.
- Duplicates removed AFTER dropping those columns (271,369 exact duplicates + 629 conflicting rows removed).
- Stratified 70/10/20 split; QuantileTransformer (normal output) fitted on train only.
- ExtraTrees feature selection fitted on train only and actually applied (29 features kept).
- Capped random over/under-sampling on train only (non-private runs); sqrt-balanced reward weights.
- Non-IID Dirichlet split (alpha = 2.0, 15 clients), no IID fallback.
- DDQN trained only with TD/reward loss (no cross-entropy). gamma = 0.1 (non-private), gamma = 0 (record-level DP).
- FedProx proximal term in standard form (mu/2)*sum||w - w_g||^2 (old code divided by #params -> effectively FedAvg).
- DP noise actually applied; RDP accounting (Opacus); budget checked before each round.
- Class names now come from the LabelEncoder (old hard-coded attack_labels were in the wrong order).

## 1. Capacity diagnostics (validation)
| Diagnostic | Result |
|---|---|
| Random forest, original train | Acc 98.5%, FAR 0.4% |
| Random forest, resampled train | Acc 97.9%, FAR 1.6% |
| Centralised DDQN, StandardScaler | Acc 68.0%, FAR 29.2% |
| Centralised DDQN, QuantileTransformer | Acc 93.4%, FAR 5.7% |
| Centralised supervised CE (reference only), QuantileTransformer | Acc 92.9%, FAR 7.1% |
| DDQN gamma 0.1 vs gamma 0.0 (single client) | 63.9% vs 63.6% -> gamma has no effect (contextual-bandit-like MDP) |

## 2. Test results - all scenarios (record-level DP, old DP data = undersampled, 42.6k-param DDQN)
| run | DP | Acc % | Macro P % | Macro R % | Macro F1 % | Weighted F1 % | FAR % | DR % | epsilon | best round |
|---|---|---|---|---|---|---|---|---|---|---|
| fl_fedavg_nonDP | no | 94.99 | 79.69 | 94.57 | 83.38 | 95.55 | 4.76 | 99.78 | - | 46 |
| fl_fedprox_nonDP | no | 92.79 | 73.64 | 94.23 | 78.79 | 93.58 | 7.27 | 99.69 | - | 40 |
| dp_control_nonoise | no (same DP pipeline, sigma=0) | 86.94 | 65.33 | 84.34 | 70.86 | 88.03 | 13.62 | 99.38 | - | 18 |
| dp_adpmeta_eps8 | record | 60.19 | 29.42 | 35.52 | 26.52 | 60.18 | 30.87 | 90.12 | 7.209 | 13 |
| dp_fixed_eps8 | record | 60.28 | 30.93 | 32.32 | 25.50 | 60.49 | 29.34 | 89.18 | 7.758 | 13 |
| dp_adpmeta_eps3 | record | 52.26 | 25.70 | 37.63 | 25.69 | 54.46 | 42.37 | 94.10 | 2.717 | 20 |
| dp_fixed_eps3 | record | 58.95 | 35.07 | 33.35 | 23.52 | 59.77 | 28.42 | 93.14 | 2.912 | 18 |
| dp_adpmeta_eps1 | record | 13.48 | 9.92 | 19.55 | 8.14 | 9.46 | 94.89 | 89.61 | 0.875 | 8 |
| dp_fixed_eps1 | record | 42.85 | 24.25 | 23.80 | 13.05 | 44.88 | 47.61 | 88.12 | 0.963 | 18 |

Per-class test F1 (%):
| class | adpmeta_eps1 | adpmeta_eps3 | adpmeta_eps8 | control_nonoise | fixed_eps1 | fixed_eps3 | fixed_eps8 | fedavg_nonDP | fedprox_nonDP |
|---|---|---|---|---|---|---|---|---|---|
| backdoor | 0.00 | 0.00 | 0.00 | 47.93 | 0.00 | 0.00 | 0.00 | 98.48 | 76.60 |
| ddos | 0.00 | 38.92 | 52.84 | 89.41 | 0.00 | 41.63 | 51.05 | 94.16 | 92.79 |
| dos | 1.42 | 41.40 | 47.97 | 76.67 | 0.00 | 0.00 | 33.90 | 90.99 | 88.59 |
| injection | 1.16 | 0.00 | 0.20 | 90.52 | 0.05 | 0.05 | 3.76 | 94.82 | 93.28 |
| mitm | 0.00 | 0.00 | 0.00 | 49.09 | 0.00 | 0.00 | 0.00 | 38.10 | 35.10 |
| normal | 9.20 | 71.56 | 79.06 | 92.51 | 65.77 | 81.54 | 79.86 | 97.50 | 96.14 |
| password | 27.48 | 38.85 | 38.44 | 68.84 | 26.60 | 36.32 | 37.54 | 93.44 | 87.03 |
| ransomware | 0.00 | 0.00 | 0.00 | 38.05 | 0.00 | 0.00 | 0.00 | 46.71 | 37.96 |
| scanning | 29.96 | 35.21 | 0.00 | 71.74 | 26.70 | 35.36 | 0.00 | 90.04 | 92.31 |
| xss | 12.18 | 30.95 | 46.69 | 83.90 | 11.37 | 40.26 | 48.86 | 89.54 | 88.12 |

## 3. DP tuning diagnostics (validation only, eps = 8 unless stated)
| ID | Setting | Round 10 val result |
|---|---|---|
| D4-A | no noise, batch 64, 50 steps, C=1 | ~60% (setup bottleneck) |
| D4-B | no noise, batch 256, 100 steps, C=1 | 67.0% (round 20), rising |
| D4-C | eps=8, batch 256, 100 steps | 42.4% (round 20) |
| D5-D | no noise, batch 256, 400 steps, C=1 | 62.8% |
| D5-E | no noise, batch 256, 400 steps, no clipping | 79.1% -> clipping at C=1 was the problem |
| D6 | validation per-sample grad norm | median 14.66, 90th pct 17.77 |
| D6-F | no noise, C = 14.66 | 79.5% |
| D6-G | eps=8, single sigma 9.785, C = 14.66 | 51.2% |
| D7-H | eps=8, per-client sigma, batch 256, 50 steps | 37.4% |
| D7-I | eps=8, per-client sigma, batch 512, 25 steps | 32.8% |
| D7-J | eps=8, per-client sigma, batch 256, 100 steps | 38.7% |
| D8-K | eps=8, per-client sigma, batch 256, 400 steps, lr 1e-3 | 57.7%, FAR 30.4% (selected) |
| D8-L | same, lr 1e-4 | 10.0% (no learning) |
| D9 | suggestion #1: ALL train data per client (sigma median 1.99) | Acc 64.2%, macro-F1 19.6, FAR 15.8% |
| D10 | #1 + small DDQN 29-64-32-10 (4,522 params, C=7.83) | Acc 66.2%, macro-F1 28.3, FAR 28.0% |
| D10 | #1 + small DDQN + server EMA (beta 0.7) | Acc 61.8%, macro-F1 25.4, FAR 28.1% -> EMA not used |

## 4. Latest private run (Block 18 FINAL) - TEST
FedProx + DDQN (small) + ADP-Meta, record-level DP, all train data per client, 15 clients/round, eps budget 8.
Stopped at round 16 (budget reached), best validation round 9 (val acc 73.22%).

| Metric | Value |
|---|---|
| Accuracy | 73.17% |
| Macro precision / recall / F1 | 26.37 / 31.80 / 28.09 |
| Weighted F1 | 69.27 |
| FAR | 10.19% |
| DR | 78.00% |
| epsilon (delta = 1e-5) | 7.956 |

| class | P % | R % | F1 % | support |
|---|---|---|---|---|
| backdoor | 0.00 | 0.00 | 0.00 | 393 |
| ddos | 49.56 | 76.34 | 60.10 | 2375 |
| dos | 27.55 | 33.59 | 30.27 | 1286 |
| injection | 50.98 | 86.03 | 64.02 | 3944 |
| mitm | 0.00 | 0.00 | 0.00 | 207 |
| normal | 87.53 | 89.81 | 88.65 | 23906 |
| password | 2.33 | 0.07 | 0.13 | 3033 |
| ransomware | 0.00 | 0.00 | 0.00 | 41 |
| scanning | 0.00 | 0.00 | 0.00 | 895 |
| xss | 45.76 | 32.14 | 37.76 | 1730 |

## 5. Known open issues
- Single seed everywhere; ADP-Meta vs fixed noise not yet compared under the latest settings.
- Client-level DP (Chapter 6 mechanism) implemented but not yet run; expected noise >> signal with 15 clients.
- No gradient-leakage attack code; no runtime/efficiency measurement for DP runs; only ToN-IoT available.
