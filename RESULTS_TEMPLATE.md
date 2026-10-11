# Chapter 6 results template (TEST set, mean ± std over seeds 0/1/2, filled from Block 19 only)

Selection: highest validation macro-F1 with validation FAR <= 10% (else lowest FAR; "cap met" counts seeds
where the cap was met). Epsilon = total spent over all rounds, delta = 1e-5, RDP accountant (Opacus 1.6.0).
Privacy unit: one flow record (record-level DP-SGD applied locally on each client).

## Main table
| Method | Target eps | eps spent | Accuracy % | Macro-F1 % | Weighted F1 % | FAR % | DR % | Cap met |
|---|---|---|---|---|---|---|---|---|
| FedAvg, non-private (same pipeline) | - | - | | | | | | /3 |
| FedProx, non-private (same pipeline) | - | - | | | | | | /3 |
| Fixed-noise DP (FedProx + DDQN) | 1 | | | | | | | /3 |
| ADP-Meta DP (FedProx + DDQN) | 1 | | | | | | | /3 |
| Fixed-noise DP (FedProx + DDQN) | 3 | | | | | | | /3 |
| ADP-Meta DP (FedProx + DDQN) | 3 | | | | | | | /3 |
| Fixed-noise DP (FedProx + DDQN) | 8 | | | | | | | /3 |
| ADP-Meta DP (FedProx + DDQN) | 8 | | | | | | | /3 |
| *Reference: FedAvg, Chapter 5 pipeline* | - | - | | | | | | /3 |
| *Reference: FedProx, Chapter 5 pipeline* | - | - | | | | | | /3 |

## Per-class F1 % (mean ± std)
| Class | FedAvg np | FedProx np | Fixed eps1 | ADP eps1 | Fixed eps3 | ADP eps3 | Fixed eps8 | ADP eps8 |
|---|---|---|---|---|---|---|---|---|
| backdoor | | | | | | | | |
| ddos | | | | | | | | |
| dos | | | | | | | | |
| injection | | | | | | | | |
| mitm | | | | | | | | |
| normal | | | | | | | | |
| password | | | | | | | | |
| ransomware | | | | | | | | |
| scanning | | | | | | | | |
| xss | | | | | | | | |

## ADP-Meta vs fixed noise (paired by seed)
| eps | Δ macro-F1 (ADP - fixed), mean ± std | seeds where ADP > fixed |
|---|---|---|
| 1 | | /3 |
| 3 | | /3 |
| 8 | | /3 |

With 3 seeds, call a difference meaningful only if the paired mean difference is larger than its std and
has the same sign on all 3 seeds; otherwise report "no significant difference".
