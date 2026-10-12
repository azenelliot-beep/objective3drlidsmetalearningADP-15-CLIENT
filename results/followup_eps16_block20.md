# Follow-up: Block 20 post-processing on the eps = 16 models (DISCLOSED second use of the test set)
Pasted by the author 2026-10-12. No retraining; offsets learned on validation only; same (eps, delta).

## (a) Binary IDS view (test, attack vs normal), raw decision
| run | BinAcc | BinPrec | DR | FAR | BinF1 | 10-class Acc | 10-class MacroF1 |
|---|---|---|---|---|---|---|---|
| ADP-Meta eps16 | 87.55 ± 3.45 | 85.82 ± 2.29 | 79.50 ± 12.78 | 7.77 ± 2.36 | 82.07 ± 6.60 | 80.60 ± 4.57 | 46.54 ± 11.40 |
| Fixed eps16 | 87.78 ± 2.68 | 85.05 ± 1.45 | 81.01 ± 8.08 | 8.28 ± 1.09 | 82.85 ± 4.47 | 81.12 ± 3.55 | 45.72 ± 6.42 |

## (b) Per-class ROC-AUC from Q-values (test, mean of 3 seeds)
| class | ADP-Meta | Fixed |
|---|---|---|
| backdoor | 0.761 | 0.834 |
| ddos | 0.937 | 0.938 |
| dos | 0.827 | 0.840 |
| injection | 0.904 | 0.909 |
| mitm | 0.635 | 0.582 |
| normal | 0.939 | 0.928 |
| password | 0.838 | 0.837 |
| ransomware | 0.679 | 0.759 |
| scanning | 0.880 | 0.879 |
| xss | 0.839 | 0.849 |

## (c) Validation-only calibration (per-class Q offsets)
| run | split | mode | Acc | MacroF1 | FAR | DR | BinAcc |
|---|---|---|---|---|---|---|---|
| ADP-Meta eps16 | val | raw | 80.44 ± 4.44 | 46.65 ± 11.29 | 7.84 ± 2.54 | 79.32 ± 12.80 | 87.44 ± 3.34 |
| ADP-Meta eps16 | val | calibrated | 82.15 ± 4.16 | 55.65 ± 10.82 | 8.37 ± 1.44 | 86.22 ± 9.96 | 89.64 ± 3.69 |
| Fixed eps16 | val | raw | 81.18 ± 3.43 | 45.86 ± 6.24 | 8.18 ± 1.28 | 81.21 ± 8.42 | 87.92 ± 2.62 |
| Fixed eps16 | val | calibrated | 82.54 ± 3.65 | 54.06 ± 7.52 | 7.42 ± 2.21 | 84.52 ± 11.59 | 89.62 ± 4.03 |
| ADP-Meta eps16 | test | raw | 80.60 ± 4.57 | 46.54 ± 11.40 | 7.77 ± 2.36 | 79.50 ± 12.78 | 87.55 ± 3.45 |
| ADP-Meta eps16 | test | calibrated | 82.24 ± 4.49 | 55.87 ± 10.79 | 8.44 ± 1.45 | 86.25 ± 10.17 | 89.61 ± 4.01 |
| Fixed eps16 | test | raw | 81.12 ± 3.55 | 45.72 ± 6.42 | 8.28 ± 1.09 | 81.01 ± 8.08 | 87.78 ± 2.68 |
| Fixed eps16 | test | calibrated | 82.63 ± 3.82 | 54.50 ± 7.71 | 7.49 ± 2.17 | 84.54 ± 11.79 | 89.58 ± 4.22 |

ADP-Meta minus fixed after calibration (test): macro-F1 +1.37 ± 5.01, Acc -0.39 ± 0.93, FAR +0.95 ± 1.01,
ADP better on 1/3 seeds -> equivalent.
Findings: calibration adds ~+9 macro-F1 on both validation and test (the gain transfers), +1.5 accuracy and
+5 to +7 DR, with FAR still under 10%. AUCs of 0.6-0.9 for classes with near-zero F1 show the DP models rank
these classes better than chance; the raw argmax decision under-uses that.
