# ADP-Meta v4 / Option 2 - stage "v4_quick" - VALIDATION, seed 0, eps 8 (pasted by the author)
Exploratory one-session stage (one seed, one eps, meta-training capped at 150 min). Test set NOT used.

| run | best round | cap met | Acc % | Macro-F1 % | FAR % | DR % | eps |
|---|---|---|---|---|---|---|---|
| dp_fixed (fixed noise) | 20 | yes | 80.74 | **49.01** | 7.58 | 79.14 | 8.0 |
| dp_fixed_proj (fixed noise + rank-64 public projection) | 20 | yes | 75.30 | 38.38 | 7.93 | 77.00 | 8.0 |
| dp_adpmeta v4 (meta-learned pacing + clip + rank) | 17 | yes | 66.89 | 23.23 | 3.90 | 34.74 | 8.0 |

Per-class F1 (%):
| class | dp_fixed | dp_fixed_proj | ADP-Meta v4 |
|---|---|---|---|
| backdoor | 33.8 | 0.0 | 0.0 |
| ddos | 65.9 | 60.8 | 35.9 |
| dos | 55.0 | 51.4 | 51.6 |
| injection | 80.4 | 53.6 | 0.0 |
| mitm | 1.5 | 0.0 | 0.0 |
| normal | 90.4 | 89.6 | 82.1 |
| password | 36.2 | 43.2 | 0.0 |
| ransomware | 0.0 | 0.0 | 0.0 |
| scanning | 55.9 | 43.7 | 40.1 |
| xss | 71.1 | 41.5 | 22.6 |

Paired (seed 0): ADP-Meta v4 - dp_fixed = -25.77 macro-F1; ADP-Meta v4 - dp_fixed_proj = -15.14 macro-F1.

Findings:
- dp_fixed reproduces the Sweep-2 reference exactly (49.01), so the v4 code did not change the baseline path.
- Projection alone costs -10.6 macro-F1: discarding the gradient outside a 64-dim public subspace (PDP-SGD style, no
  residual) removes signal for backdoor, injection and xss that the ~8x smaller noise does not make up for.
- ADP-Meta v4 (few practice episodes) drifted to rank 32 / clip x0.5 / faster spending: lower FAR, much lower DR.
- Pre-registered rule (v4 >= dp_fixed_proj + 2) NOT met. Decision: Option 2 rejected; "final_v4" and a v4 test
  evaluation are not run. Reported as a disclosed negative follow-up; Chapter 6 v2 test results stand.
