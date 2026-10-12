# Follow-up (after the main test evaluation): eps = 16, frozen final settings, VALIDATION, seeds 0/1/2
Pasted by the author 2026-10-12. Weaker privacy point on the privacy-utility curve; disclosed as a follow-up.

| run | Macro-F1 | Acc | FAR | DR |
|---|---|---|---|---|
| dp_fixed eps16 | 45.86 ± 6.24 | 81.18 ± 3.43 | 8.18 ± 1.28 | 81.21 ± 8.42 |
| dp_adpmeta eps16 | 46.65 ± 11.29 | 80.44 ± 4.44 | 7.84 ± 2.54 | 79.32 ± 12.80 |

Paired ADP - fixed (macro-F1): +1.38 / -4.54 / +5.53 -> +0.79 ± 5.06, ADP better on 2/3 seeds.
Formally meets "mean better AND >= 2/3 seeds", but the difference is far inside the seed noise (not significant).
Fixed noise at eps 16 (45.9) is no better than at eps 8 (46.0 val): beyond eps ~8 the noise is no longer the main
limit on this pipeline (clip-only no-noise control reached 71.1 macro-F1).
ADP seed 2 detected ransomware (F1 66.7), the only DP run so far that did.
