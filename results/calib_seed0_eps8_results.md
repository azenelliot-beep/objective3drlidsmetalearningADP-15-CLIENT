# Block 18d - weak-class check + threshold calibration, ADP-Meta v2 FINAL settings, seed 0, eps 8 (pasted by the author)
Evaluation only (no training; epsilon unchanged). tau chosen on validation for FAR <= 10%.
The models reproduce the earlier seed-0 final runs exactly (val macro-F1 45.98 ADP / 49.01 fixed).

## (B) Calibration
| run | set | rule | DR % | FAR % | Macro-F1 % | Acc % |
|---|---|---|---|---|---|---|
| fixed noise | val | argmax | 79.14 | 7.58 | 49.01 | 80.74 |
| fixed noise | val | calibrated (2-fold cross-fit) | 84.26 | 9.99 | 50.25 | 80.77 |
| fixed noise | test | argmax | 79.21 | 8.08 | 49.62 | 80.65 |
| fixed noise | test | calibrated (tau = 0.04) | **84.21** | 10.37 | **50.62** | 80.62 |
| ADP-Meta v2 | val | argmax | 73.40 | 5.91 | 45.98 | 79.97 |
| ADP-Meta v2 | val | calibrated (2-fold cross-fit) | 74.57 | 9.99 | 44.73 | 77.53 |
| ADP-Meta v2 | test | argmax | 73.25 | 6.25 | 46.21 | 79.74 |
| ADP-Meta v2 | test | calibrated (tau = 0.09) | 74.51 | 10.44 | 44.75 | 77.26 |

ADP-Meta minus fixed noise on test: dDR -5.97 (argmax) -> -9.70 (calibrated, equal FAR); dMacroF1 (calibrated) -5.87.
Test password F1: fixed 39.0 -> 53.3 with calibration; ADP-Meta 0.0 -> 0.0.

## (A) Where weak-class validation rows go (argmax)
| class (val rows) | fixed noise | ADP-Meta v2 | diagnosis |
|---|---|---|---|
| backdoor (196) | 37% correct, 53% -> scanning | 37% correct, 54% -> scanning | confused with another attack: representation, not threshold |
| mitm (103) | 1% correct, 85% -> normal | 1% correct, 85% -> normal | called normal, but calibration does not move it: no separating signal |
| ransomware (20) | 0% correct, 75% -> normal | 0% correct, 45% normal / 55% dos | too few rows to learn under DP |
| password (1517) | F1 36 (val) | 0% correct, 85% -> normal | ADP-Meta-specific class collapse (training-time, not threshold) |

## Findings
- Calibration is a real, validated gain for fixed-noise DP: +5.0 DR and +1.0 macro-F1 on test at FAR ~10%; the
  cross-fit validation estimate (84.26 DR) predicted the test value (84.21) almost exactly.
- ADP-Meta does not benefit (+1.3 DR, -1.5 macro-F1, -2.5 accuracy): its deficit is the password collapse, which the
  threshold cannot recover. At equal FAR the detection gap to fixed noise widens (-6.0 -> -9.7).
- backdoor and mitm fail for BOTH methods for reasons a threshold cannot fix (confusion with scanning; no signal vs
  normal). Single seed - seeds 1/2 needed before any thesis claim.
