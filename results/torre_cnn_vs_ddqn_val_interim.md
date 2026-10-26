# Interim: Torre et al. 1D-CNN vs Chapter 6 DDQN inside the SAME Chapter 6 pipeline - VALIDATION only

Status 2026-10-26: 11 of 15 CNN runs finished (the background job hit its 2-hour limit). Missing: seed 2 of
fedprox_np and dp_fixed eps 8/3/1. **The test set has not been used for these runs** (Block 19 runs once, after
all 15 are done). Re-launch `notebooks/chapter6_v4_torre_cnn.ipynb`: finished runs are skipped.

Same for both detectors: leakage-free data (271,369 duplicates + 629 conflicts removed, 29 ExtraTrees features,
70/10/20 stratified split, QuantileTransformer on train), 15 Dirichlet(alpha 2) clients, 20 rounds x 400 local
DP-SGD steps, batch 256, Adam 1e-3, FedProx mu 0.01 (fedavg_np: mu 0), per-sample clipping with
C = 0.5 x median per-sample grad norm on validation (CNN: C = 11.05; DDQN: C = 3.927), RDP accountant,
delta 1e-5, selection = max val macro-F1 with val FAR <= 10 %.
Different: detector (CNN 54,794 params, cross-entropy; DDQN 4,522 params, TD loss on a class-weighted reward with
false-alarm penalty 0.5), hardware (CNN: CPU here; DDQN: Kaggle GPU), and tuning (the DDQN settings were tuned on
validation in Sweeps 1-2; the CNN only inherits CLIP_MULT 0.5, nothing was tuned for it).

| Setting | CNN macro-F1 | DDQN macro-F1 | CNN Acc | DDQN Acc | CNN FAR | DDQN FAR | CNN DR | DDQN DR |
|---|---|---|---|---|---|---|---|---|
| FedAvg, non-private (CNN seeds 0-2) | **88.00 ± 0.36** | 82.12 ± 0.82 | 97.49 ± 0.09 | 95.32 ± 0.03 | 0.64 ± 0.08 | 1.22 ± 0.24 | 98.38 ± 0.07 | 97.82 ± 0.52 |
| FedProx, non-private (CNN seeds 0-1) | **83.51 ± 0.10** | 76.96 ± 1.63 | 96.60 ± 0.20 | 92.97 ± 0.68 | 0.66 ± 0.02 | 1.69 ± 0.10 | 98.25 ± 0.04 | 95.50 ± 0.94 |
| Fixed DP eps 8 (CNN seeds 0-1) | **51.98 ± 0.18** | 46.00 ± 6.17 | 86.16 ± 0.42 | 80.44 ± 3.07 | 4.03 ± 0.90 | 8.44 ± 0.75 | 84.90 ± 0.18 | 81.06 ± 7.40 |
| Fixed DP eps 3 (CNN seeds 0-1) | **44.58 ± 1.43** | 31.09 ± 4.97 | 80.95 ± 0.88 | 71.86 ± 3.52 | 3.44 ± 0.09 | 4.89 ± 0.69 | 71.71 ± 1.40 | 55.20 ± 9.17 |
| Fixed DP eps 1 (CNN seeds 0-1) | **22.01 ± 6.41** | 18.23 ± 2.48 | 69.59 ± 2.39 | 66.55 ± 1.39 | 1.72 ± 1.44 | 3.22 ± 1.60 | 43.10 ± 1.49 | 33.26 ± 7.33 |
| ADP-Meta eps 8 / 3 / 1 (DDQN only) | - | 39.65 / 24.48 / 17.57 | - | 76.64 / 68.82 / 66.26 | - | 7.20 / 4.60 / 3.17 | - | 70.33 / 48.61 / 31.96 |

DDQN = `results/final_v2_val_summary.md` (seeds 0-2). Spent eps of every CNN DP run is within 0.002 of its budget.

CNN per-class validation F1 (mean of finished seeds):

| class | FedAvg | FedProx | DP eps 8 | DP eps 3 | DP eps 1 |
|---|---|---|---|---|---|
| backdoor | 92.5 | 67.6 | 0.0 | 0.0 | 0.0 |
| ddos | 95.7 | 95.3 | 81.2 | 74.2 | 64.9 |
| dos | 94.5 | 92.8 | 66.9 | 54.2 | 31.1 |
| injection | 96.0 | 96.1 | 86.1 | 70.0 | 10.3 |
| mitm | 60.1 | 54.9 | 1.9 | 0.0 | 0.0 |
| normal | 99.2 | 99.2 | 93.7 | 90.7 | 85.0 |
| password | 98.0 | 95.1 | 57.6 | 52.9 | 0.0 |
| ransomware | 61.3 | 60.6 | 0.0 | 0.0 | 0.0 |
| scanning | 92.2 | 84.1 | 53.6 | 45.2 | 28.3 |
| xss | 90.4 | 89.4 | 78.8 | 58.5 | 0.6 |
