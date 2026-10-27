# Torre et al. 1D-CNN vs Chapter 6 DDQN inside the SAME Chapter 6 pipeline - FINAL TEST (used once)

Run 2026-10-26/27 from `notebooks/chapter6_v4_torre_cnn.ipynb` on ToN-IoT `Train_Test_Network.csv`, seeds 0/1/2,
mean ± sample std. Block 19 evaluated only the `torre_cnn` runs, once; the DDQN test set was not reused.
DDQN rows = `results/final_v2_TEST_results.md` (same seeds, same protocol, Kaggle GPU). Raw files: `results/torre_cnn/`.
Supersedes `results/torre_cnn_vs_ddqn_val_interim.md`.

Identical for both: leakage-free data (271,369 duplicates + 629 conflicts removed, 29 ExtraTrees features,
stratified 70/10/20, QuantileTransformer fitted on train), 15 Dirichlet(alpha 2) clients, 20 rounds x 400 local
DP-SGD steps, batch 256, Adam 1e-3, FedProx mu 0.01 (fedavg_np: mu 0), per-sample clipping with
C = 0.5 x median per-sample grad norm on validation (CNN C = 11.05, DDQN C = 3.927), RDP accountant, delta 1e-5,
selection = max val macro-F1 with val FAR <= 10 % (met by every run).
Different: detector and loss (CNN 54,794 params, cross-entropy; DDQN 4,522 params, TD loss on a class-weighted reward
with false-alarm penalty 0.5), hardware (CNN CPU, DDQN GPU), and tuning (DDQN settings tuned on validation in
Sweeps 1-2; nothing tuned for the CNN).

| Setting | CNN Acc | DDQN Acc | CNN macro-F1 | DDQN macro-F1 | CNN FAR | DDQN FAR | CNN DR | DDQN DR |
|---|---|---|---|---|---|---|---|---|
| FedAvg, non-private | **97.34 ± 0.06** | 95.20 ± 0.05 | **87.78 ± 1.02** | 81.74 ± 0.26 | **0.84 ± 0.04** | 1.51 ± 0.29 | **98.43 ± 0.05** | 97.82 ± 0.41 |
| FedProx, non-private | **96.49 ± 0.15** | 92.80 ± 0.44 | **83.52 ± 0.84** | 75.99 ± 1.24 | **0.88 ± 0.01** | 2.01 ± 0.20 | **98.19 ± 0.04** | 95.38 ± 0.93 |
| Fixed DP, eps 8 | **86.19 ± 0.27** | 80.51 ± 3.19 | **51.94 ± 0.04** | 46.27 ± 6.41 | **4.39 ± 0.65** | 8.55 ± 0.43 | **85.46 ± 0.74** | 80.86 ± 7.21 |
| Fixed DP, eps 3 | **80.60 ± 0.78** | 71.74 ± 3.53 | **43.73 ± 1.42** | 31.28 ± 4.75 | **3.66 ± 0.11** | 5.13 ± 0.62 | **70.90 ± 1.39** | 54.89 ± 9.41 |
| Fixed DP, eps 1 | **69.22 ± 1.83** | 66.38 ± 1.38 | **21.17 ± 4.87** | 18.20 ± 2.60 | **2.07 ± 1.19** | 3.42 ± 1.76 | **41.23 ± 2.84** | 33.18 ± 7.54 |
| ADP-Meta, eps 8 / 3 / 1 (DDQN only) | - | 76.47 / 68.57 / 66.08 | - | 39.53 / 24.22 / 17.57 | - | 7.47 / 4.84 / 3.38 | - | 70.16 / 48.47 / 31.78 |

Difference CNN minus DDQN (means): macro-F1 +6.0 / +7.5 / +5.7 / +12.5 / +3.0 and accuracy
+2.1 / +3.7 / +5.7 / +8.9 / +2.8 (FedAvg / FedProx / eps 8 / eps 3 / eps 1). Spent eps of every CNN DP run is
within 0.002 of its budget. Weighted F1 (CNN): 97.35 / 96.51 / 85.27 / 78.72 / 60.03.

Per-class test F1, CNN (mean ± std):

| class | FedAvg | FedProx | DP eps 8 | DP eps 3 | DP eps 1 |
|---|---|---|---|---|---|
| backdoor | 93.03 ± 2.70 | 72.69 ± 6.33 | 0.00 | 0.00 | 0.00 |
| ddos | 95.44 ± 0.07 | 94.62 ± 0.46 | 80.49 ± 0.55 | 73.62 ± 2.15 | 63.75 ± 8.34 |
| dos | 93.55 ± 0.17 | 91.83 ± 0.33 | 64.98 ± 0.64 | 52.65 ± 3.02 | 19.60 ± 17.39 |
| injection | 95.57 ± 0.08 | 95.33 ± 0.37 | 86.80 ± 0.48 | 69.98 ± 4.27 | 9.79 ± 5.20 |
| mitm | 60.33 ± 1.13 | 54.80 ± 4.52 | 0.32 ± 0.56 | 0.00 | 0.00 |
| normal | 99.12 ± 0.01 | 99.03 ± 0.01 | 93.71 ± 0.25 | 90.35 ± 0.35 | 84.38 ± 1.08 |
| password | 98.11 ± 0.19 | 96.27 ± 2.13 | 60.06 ± 1.76 | 55.35 ± 2.44 | 2.96 ± 5.07 |
| ransomware | 60.66 ± 7.78 | 58.15 ± 11.21 | 0.00 | 0.00 | 0.00 |
| scanning | 91.83 ± 1.26 | 83.52 ± 1.42 | 54.37 ± 3.52 | 41.93 ± 6.63 | 30.95 ± 29.62 |
| xss | 90.13 ± 0.14 | 88.91 ± 0.65 | 78.69 ± 0.86 | 53.45 ± 5.97 | 0.23 ± 0.40 |

Findings
- The CNN is better than the DDQN on every metric in every setting (accuracy, macro-F1, FAR, DR), with the same
  formal (eps, delta) and the same data, clients, DP-SGD and selection rule. Its DP results are also far more
  stable across seeds (eps 8 macro-F1 std 0.04 vs 6.41).
- The fixed-noise CNN beats the ADP-Meta DDQN at every eps (macro-F1 +12.4 / +19.5 / +3.6 at eps 8 / 3 / 1).
- Under DP both detectors lose the rare classes: backdoor, MITM and ransomware have F1 ~ 0 for the CNN at every eps.
- Validation and test agree within about 0.5 macro-F1 (validation 88.00 / 83.89 / 51.91 / 44.09 / 21.34), so model
  selection did not overfit validation.

Not shown by this experiment: ADP-Meta with the CNN detector (not run), CNN hyperparameter tuning (none was done),
paired per-seed significance against the DDQN (DDQN per-seed test values are not stored in this repo).
