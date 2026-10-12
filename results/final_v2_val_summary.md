# Final stage (ADP-Meta v2 + meta-learning vs fixed noise) - VALIDATION, seeds 0/1/2 (pasted by the author 2026-10-12)
Settings frozen after Sweep 2: clip x0.5 (C = 3.927), false-alarm penalty 0.5, same reward for every method,
ADP-Meta v2 with 10 meta-episodes per eps. Selection: max val macro-F1 with val FAR <= 10% (all runs met the cap).

| run | Macro-F1 | Acc | FAR | DR |
|---|---|---|---|---|
| fedavg_np | 82.12 ± 0.82 | 95.32 ± 0.03 | 1.22 ± 0.24 | 97.82 ± 0.52 |
| fedprox_np | 76.96 ± 1.63 | 92.97 ± 0.68 | 1.69 ± 0.10 | 95.50 ± 0.94 |
| dp_fixed eps8 | 46.00 ± 6.17 | 80.44 ± 3.07 | 8.44 ± 0.75 | 81.06 ± 7.40 |
| dp_adpmeta eps8 | 39.65 ± 6.08 | 76.64 ± 3.76 | 7.20 ± 1.30 | 70.33 ± 6.23 |
| dp_fixed eps3 | 31.09 ± 4.97 | 71.86 ± 3.52 | 4.89 ± 0.69 | 55.20 ± 9.17 |
| dp_adpmeta eps3 | 24.48 ± 6.40 | 68.82 ± 2.14 | 4.60 ± 0.27 | 48.61 ± 5.81 |
| dp_fixed eps1 | 18.23 ± 2.48 | 66.55 ± 1.39 | 3.22 ± 1.60 | 33.26 ± 7.33 |
| dp_adpmeta eps1 | 17.57 ± 1.85 | 66.26 ± 0.95 | 3.17 ± 1.59 | 31.96 ± 8.45 |

Paired ADP-Meta minus fixed (macro-F1): eps8 -6.35 ± 8.70 (ADP better 1/3 seeds), eps3 -6.61 ± 5.87 (0/3),
eps1 -0.66 ± 1.17 (1/3). Per-seed: eps8 -3.02/+0.20/-16.22, eps3 -0.75/-12.50/-6.57, eps1 -0.01/+0.04/-2.01.

Pre-registered rule (mean better AND better on >= 2 of 3 seeds): NOT met at any eps.
Conclusion: ADP-Meta v2 provides the same formal record-level DP with slightly lower FAR, but lower macro-F1/DR
than tuned fixed noise at eps 3 and 8, and is equivalent at eps 1 (the controller effectively paces uniformly).
