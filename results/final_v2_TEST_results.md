# Chapter 6 - FINAL TEST results (used once, Block 19, TEST_TAG = "final"), seeds 0/1/2, mean ± std (ddof=1)
ToN-IoT network, leakage-free pipeline, 15 clients (Dirichlet alpha = 2), 20 rounds x 400 local DP-SGD steps,
record-level local DP (RDP accountant, delta = 1e-5), clip x0.5 (C = 3.927), false-alarm penalty 0.5 for all methods.
Model selection: max validation macro-F1 with validation FAR <= 10% (met in every run).

| Method | eps | Accuracy % | Macro-F1 % | Weighted F1 % | FAR % | DR % |
|---|---|---|---|---|---|---|
| FedAvg (non-private) | - | 95.20 ± 0.05 | 81.74 ± 0.26 | 95.41 ± 0.04 | 1.51 ± 0.29 | 97.82 ± 0.41 |
| FedProx (non-private) | - | 92.80 ± 0.44 | 75.99 ± 1.24 | 92.97 ± 0.36 | 2.01 ± 0.20 | 95.38 ± 0.93 |
| Fixed-noise DP | 8 | 80.51 ± 3.19 | 46.27 ± 6.41 | 78.73 ± 4.81 | 8.55 ± 0.43 | 80.86 ± 7.21 |
| ADP-Meta DP (v2, meta-learned) | 8 | 76.47 ± 3.84 | 39.53 ± 6.52 | 72.65 ± 4.40 | 7.47 ± 1.21 | 70.16 ± 6.23 |
| Fixed-noise DP | 3 | 71.74 ± 3.53 | 31.28 ± 4.75 | 65.85 ± 5.12 | 5.13 ± 0.62 | 54.89 ± 9.41 |
| ADP-Meta DP (v2, meta-learned) | 3 | 68.57 ± 2.08 | 24.22 ± 6.26 | 60.86 ± 3.14 | 4.84 ± 0.32 | 48.47 ± 5.51 |
| Fixed-noise DP | 1 | 66.38 ± 1.38 | 18.20 ± 2.60 | 56.92 ± 0.83 | 3.42 ± 1.76 | 33.18 ± 7.54 |
| ADP-Meta DP (v2, meta-learned) | 1 | 66.08 ± 0.92 | 17.57 ± 2.05 | 56.51 ± 0.41 | 3.38 ± 1.69 | 31.78 ± 8.56 |

ADP-Meta minus fixed noise, paired by seed:
| eps | d macro-F1 | d Acc | d FAR | seeds ADP better |
|---|---|---|---|---|
| 8 | -6.74 ± 9.18 | -4.03 ± 6.38 | -1.08 ± 0.79 | 1/3 |
| 3 | -7.05 ± 6.15 | -3.17 ± 2.63 | -0.29 ± 0.38 | 0/3 |
| 1 | -0.63 ± 1.16 | -0.29 ± 0.46 | -0.04 ± 0.07 | 2/3 |

Per-class test F1 (mean ± std):
| class | FedAvg | FedProx | Fixed e8 | ADP e8 | Fixed e3 | ADP e3 | Fixed e1 | ADP e1 |
|---|---|---|---|---|---|---|---|---|
| backdoor | 60.49 ± 0.20 | 49.66 ± 7.43 | 13.35 ± 23.12 | 13.38 ± 23.18 | 12.29 ± 21.29 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ddos | 92.96 ± 0.32 | 88.38 ± 0.76 | 67.80 ± 4.55 | 63.95 ± 3.45 | 54.28 ± 12.19 | 52.92 ± 11.22 | 53.75 ± 15.30 | 54.00 ± 15.28 |
| dos | 91.31 ± 0.23 | 70.73 ± 5.29 | 49.23 ± 8.00 | 43.80 ± 9.10 | 26.26 ± 23.36 | 23.13 ± 21.85 | 20.65 ± 21.32 | 14.99 ± 13.27 |
| injection | 94.24 ± 0.25 | 92.01 ± 1.10 | 78.58 ± 3.03 | 54.97 ± 44.00 | 31.76 ± 28.81 | 6.80 ± 1.62 | 2.97 ± 2.49 | 2.26 ± 2.16 |
| mitm | 57.34 ± 2.00 | 50.36 ± 3.43 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.58 ± 1.01 | 0.88 ± 0.87 | 1.49 ± 1.34 | 1.50 ± 1.37 |
| normal | 98.61 ± 0.03 | 97.66 ± 0.16 | 90.31 ± 1.92 | 88.19 ± 1.60 | 85.85 ± 1.98 | 84.55 ± 1.34 | 82.05 ± 0.68 | 81.79 ± 0.89 |
| password | 93.88 ± 0.42 | 91.15 ± 0.86 | 34.21 ± 32.07 | 16.94 ± 29.34 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ransomware | 67.43 ± 1.36 | 67.07 ± 1.24 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| scanning | 74.09 ± 0.45 | 68.09 ± 3.13 | 58.36 ± 8.18 | 48.43 ± 16.40 | 37.42 ± 9.72 | 36.33 ± 11.91 | 14.44 ± 15.48 | 14.59 ± 15.76 |
| xss | 87.06 ± 0.26 | 84.79 ± 0.81 | 70.85 ± 2.18 | 65.64 ± 4.39 | 64.33 ± 1.74 | 37.63 ± 32.76 | 6.67 ± 11.29 | 6.58 ± 11.29 |

Findings: test agrees with validation within ~0.5 points (selection did not overfit validation). Pre-registered
rule for ADP-Meta beating fixed noise is not met at any eps; ADP-Meta has a lower FAR at every eps and is
equivalent to fixed noise at eps 1. Privacy cost at eps 8: 80.5% vs 92.8% (FedProx) accuracy = 87% retained.
