# Results: synthetic

## Main configuration, seeds [1, 2, 42] (mean ± sample std)

| Class | without PP: A / P / R / F1 | with PP: A / P / R / F1 |
|---|---|---|
| Normal | 99.44 ± 0.09 / 99.02 ± 0.24 / 99.88 ± 0.06 / 99.45 ± 0.09 | 97.11 ± 1.16 / 98.02 ± 1.13 / 96.21 ± 3.39 / 97.07 ± 1.24 |
| DDoS | 98.76 ± 0.15 / 97.25 ± 0.44 / 96.54 ± 0.67 / 96.89 ± 0.37 | 90.71 ± 4.02 / 82.74 ± 14.36 / 74.37 ± 30.27 / 74.04 ± 16.57 |
| DoS | 98.70 ± 0.07 / 96.12 ± 0.50 / 95.15 ± 0.89 / 95.63 ± 0.24 | 89.38 ± 4.28 / 74.19 ± 22.13 / 59.43 ± 15.41 / 62.96 ± 8.77 |
| Recon | 99.02 ± 0.09 / 95.76 ± 0.38 / 96.67 ± 0.40 / 96.21 ± 0.36 | 94.32 ± 2.47 / 77.55 ± 14.76 / 85.79 ± 10.49 / 80.19 ± 5.24 |
| Rare | 99.67 ± 0.07 / 98.00 ± 1.32 / 85.09 ± 3.54 / 91.05 ± 1.78 | 97.87 ± 2.40 / 67.06 ± 34.14 / 81.24 ± 13.13 / 67.61 ± 21.60 |
| Overall (Table 10) | 99.12 ± 0.07 / 97.23 ± 0.30 / 94.66 ± 0.70 / 95.85 ± 0.45 | 93.88 ± 1.82 / 79.91 ± 3.59 / 79.41 ± 4.19 / 76.38 ± 2.51 |

## FL performance (Tables 11/12 format, averaged over clients, then seeds)

| Setting | I# | clients converged | CCT (s) | TFD (s) | L-A (%) | F-A (%) | rounds with NaN F-A |
|---|---|---|---|---|---|---|---|
| without_pp | 10.00 ± 0.00 | 0/12 | 2.51 ± 0.10 | 18.12 ± 0.10 | 97.69 ± 0.40 | 98.08 ± 0.44 | 0 |
| with_pp | 10.00 ± 0.00 | 0/12 | 2.67 ± 0.22 | 18.47 ± 0.17 | 97.05 ± 0.52 | 85.01 ± 4.49 | 0 |

PP overhead on mean CCT: +0.17 s (+6.6 %)  [paper: about +12 % CCT, +8 % TFD averaged over datasets]

PP overhead on mean TFD: +0.34 s (+1.9 %)  [paper: about +12 % CCT, +8 % TFD averaged over datasets]

## Ablations of the ambiguous choices (seed 42; one change each vs the main configuration)

| Run | overall A | overall F1 | final F-A (%) | min over rounds of max\|dW\| | clients converged | rounds with NaN F-A |
|---|---|---|---|---|---|---|
| with_pp_abl_alg1 | 97.02 | 87.42 | 92.62 | 2.36e-01 | 0/4 | 0 |
| with_pp_abl_allarrays | 80.09 | 13.37 | nan | 2.33e-01 | 0/4 | 32 |
| with_pp_abl_conv | 92.76 | 78.78 | 81.94 | 8.03e-02 | 0/4 | 0 |
| with_pp_abl_lambda0.001 | 99.08 | 95.65 | 97.60 | 3.50e-02 | 0/4 | 0 |
| with_pp_abl_lambda0.01 | 98.98 | 95.18 | 97.46 | 3.71e-02 | 0/4 | 0 |
| with_pp_abl_ownfull | 82.23 | 42.85 | 55.40 | 5.46e-01 | 0/4 | 0 |
| with_pp_abl_sensitivity | 97.38 | 89.70 | 93.36 | 1.40e-01 | 0/4 | 0 |
| with_pp_seed42 | 94.61 | 78.92 | 86.72 | 2.33e-01 | 0/4 | 0 |
| without_pp_abl_alg1 | 97.91 | 89.60 | 94.62 | 4.45e-02 | 0/4 | 0 |
| without_pp_abl_conv | 98.91 | 94.75 | 97.40 | 7.01e-03 | 0/4 | 0 |
| without_pp_abl_indepinit | 98.92 | 94.51 | 97.42 | 3.24e-02 | 0/4 | 0 |
| without_pp_seed42 | 99.08 | 95.36 | 97.64 | 3.59e-02 | 0/4 | 0 |
