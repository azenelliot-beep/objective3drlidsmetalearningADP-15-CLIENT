# ADP-Meta v3 - stage "v3_quick" - VALIDATION, seed 0, eps 8 (pasted by the author)
Frozen v3 design (joint pacing + clip, init at the fixed schedule, twin-baseline meta-training on eps 8, 180-min cap).
Same detector / FedProx / DP settings as the final stage. Test set NOT used.

| run | best round | Acc % | Macro-F1 % | FAR % | DR % | eps |
|---|---|---|---|---|---|---|
| dp_fixed (fixed noise) | 20 | 80.74 | **49.01** | 7.58 | 79.14 | 8.0 |
| ADP-Meta v3 | 14 | 78.60 | 43.05 | 9.00 | 76.78 | 8.0 |

Per-class F1 (%): v3 vs fixed - backdoor 0.0 vs 33.8 | ddos 66.4 vs 65.9 | dos 58.2 vs 55.0 | injection 73.9 vs 80.4 |
mitm 1.5 vs 1.5 | normal 89.0 vs 90.4 | password 18.3 vs 36.2 | ransomware 0 vs 0 | scanning 53.4 vs 55.9 | xss 69.7 vs 71.1

Paired (seed 0): v3 - fixed = -5.96 macro-F1. Pre-registered rule (>= +2) NOT met; final_v3 not run.
Note: v3 partly avoids v2's password collapse (18.3 vs v2 0.0) but loses backdoor entirely (0 vs 33.8).
Fixed noise reproduces 49.01 exactly again (pipeline deterministic).
