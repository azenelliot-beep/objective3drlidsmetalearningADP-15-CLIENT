# Reimplementation: Torre et al., "Toward Enhancing Privacy Preservation of a Federated Learning CNN IDS in IoT"

ACM TOSEM 34(2), Article 53, 2025, <https://doi.org/10.1145/3695998>. This folder is self-contained and does not
touch the Chapter 6 code elsewhere in this repository.

## Read this first

1. **None of the seven datasets was available in this environment.** UCI, Hugging Face and Kaggle are blocked by
   the network policy, and the paper ships no data. Every number produced here comes from a **synthetic** table that
   exercises the same code path. Nothing in this folder reproduces Tables 3-12; those runs need the CSVs (see
   [Datasets](#datasets)).
2. **No official code exists** (no link in the paper, none found online). The DP and Diffie-Hellman layers follow
   **PrivacyFL** (Mugunthan et al., ref [59]), which the paper names as the source of its DP method. Its structure
   (client agents, a directory of active clients, offsets that are added before and subtracted after the current
   client, dropout once weights are within tolerance, simulated latencies, "intercepts") is clearly the base of the
   paper's system. I read PrivacyFL's code and use it where the paper is silent. Every such use is labelled
   `[PRIVACYFL]`. It is not the authors' code.
3. **The "homomorphic encryption" of Algorithms 4 and 5 is not homomorphic encryption.** It applies a reversible,
   per-list random add/subtract/multiply/divide to the integer part of each value, and the key material (operation,
   `B_I`, previous offset) goes into directories `D_X`/`D_Y` that the server reads to decrypt **before** averaging
   (Step 4). No computation ever runs on ciphertexts, and nothing relates the scheme to LWE/RLWE, despite §3.10.3.
   I implement it exactly as described. It is invertible (verified bit-for-bit), but it gives no protection against
   anyone who holds the directory. The server needs that directory, so the server can undo it. The same holds for
   the server-side re-encryption (Step 6).
4. **At the paper's example noise scale (λ = 0.1, §3.10.1), the DP layer hurts accuracy. The paper says PP changes
   accuracy "negligibly".** After each client removes its own noise share, the federated model still carries
   (1/n)·Σ_{j≠k} noise_j, with a standard deviation of about 0.03 per weight. The paper does not report Δ or ε for
   its experiments, so it is impossible to tell what noise level the authors actually used.
5. **The convergence rule (max |federated − local| ≤ 0.0005 over all weights) is never met here, even without PP**,
   so every client runs all 10 iterations. The paper reports convergence after 2-10 iterations (Tables 11/12).
   Either their data saturated training, or "weights and biases" meant something narrower (see ambiguity A3).

## The method in one paragraph

Four clients and one server run up to 10 FL iterations. The dataset is pre-processed and cut at random into 40
equal pieces, one per (client, iteration), and each piece is split 75/25. In every iteration each client:

- trains a 1D CNN (Conv1D 32×3 → MaxPool 2 → Flatten → Dense 128 ReLU → Dropout 0.5 → BatchNorm → softmax; Adam,
  cross-entropy, batch 512, 10 epochs) on its piece, under a lock and a barrier, starting from the last federated
  model. This gives L1;
- copies L1 to L2 and protects it in three layers:
  - **DP:** per element, Gamma(1/n, λ) − Gamma(1/n, λ). The n client shares sum to Laplace(μ, λ);
  - **DK:** pairwise ECDH secrets → integer deltas. O_tot = Σ deltas of clients before me − Σ deltas of clients
    after me; it cancels in the sum;
  - **"HE":** Algorithm 4, with directories D_X/D_Y.

The server then:

- reverses the HE (Algorithm 5);
- averages (FedAvg), so the DK offsets cancel;
- adds a fixed offset and re-applies the HE for each client.

Finally, each client:

- reverses the HE, removes the fixed offset, and subtracts its DP noise;
- measures local (L1) and federated (L2) accuracy on its 25% test part;
- drops out when L1 and L2 agree within 0.0005.

## How to run

```bash
pip install -r torre2025_fl_cnn_pp/requirements.txt          # see the file for the paper's exact versions
python -m pytest torre2025_fl_cnn_pp/tests -q                 # 24 tests, about 1 minute on CPU

# 1-minute end-to-end smoke test on synthetic data (smaller epochs and iterations; never use for results)
python -m torre2025_fl_cnn_pp.run --dataset synthetic --smoke

# Full protocol (10 epochs, batch 512, 4 clients, up to 10 iterations), with and without PP
python -m torre2025_fl_cnn_pp.run --dataset ton_iot --data-path /data/ton_iot/Train_Test_Network.csv
python -m torre2025_fl_cnn_pp.summarize results_torre2025 ton_iot      # seed-averaged tables + ablations
```

- **Output layout:** results go to `results_torre2025/<dataset>/<with_pp|without_pp><suffix>/`:
  - `summary.json`: per-class Tables 3-9, the overall Table 10 row, the per-client Tables 11/12 row, mechanism
    counters, offset-cancellation checks and the full config;
  - `history.jsonl`: per client and iteration L-A, F-A, CCT, TFD, max|ΔW|, max|ΔB|;
  - `run.log`, plus `checkpoint_C<k>_it<t>.npz` (each client's federated model after every iteration);
  - `report.md`, written next to the run folders, puts the obtained numbers beside the paper's.
- **Resume:** finished runs are skipped on relaunch; `--force` recomputes.
- **Development options:** `--sample-frac 0.05` reproduces the paper's "5 % dataset" development setting (§3.1),
  and `--max-rows N` caps the row count. Both are recorded in `summary.json` and must not be used for final numbers.
- **Configuration:** every option is in [`config.py`](config.py), each tagged `[PAPER §x]`, `[PRIVACYFL]` or
  `[ASSUMPTION]`. The command-line flags expose the ambiguous choices (`--architecture`, `--federated-arrays`,
  `--dp-noise-removal`, `--dp-scale-mode`, `--dp-fixed-scale`, `--independent-init`, `--tolerance`, `--seed`).

### Datasets

| `--dataset` | `--data-path` | Classes (paper table) |
|---|---|---|
| `ton_iot` | `Train_Test_Network.csv` (461,043 rows) | Table 3 (10 classes; `scanning` → Reconnaissance) |
| `iot_23` | directory with the `conn.log.labeled` files (or CSV exports) | Table 4 |
| `bot_iot` | directory with the 74 full-release `UNSW_2018_IoT_Botnet_Dataset_*.csv` (header-less files are handled) | Table 5 |
| `cic_iot_2023` | directory with the merged `part-*.csv` files | Table 6 |
| `cic_iomt_2024` | directory with the WiFi/MQTT `*.pcap.csv` files (train and test merged; label from the file name) | Table 7 |
| `rt_iot_2022` | `RT_IOT2022.csv` (UCI) | Table 8 |
| `edge_iiot` | `DNN-EdgeIIoT-dataset.csv` (2,219,201 rows) | Table 9 |

## Implementation checklist (paper → code → evidence)

Provenance column: **P** = stated in the paper, **F** = PrivacyFL code (ref [59], not the authors' code),
**A** = assumption (see below), **D** = deliberate deviation.

| # | Requirement | Paper | Code | Verification | Prov. |
|---|---|---|---|---|---|
| 1 | 4 clients, 1 server, ≤ 10 iterations | §3, §3.9 | `config.py`, `fl.run_federated` | `test_all_mechanisms_execute` | P |
| 2 | Random split into 40 equal pieces, one per client-iteration | §3.1 | `data.prepare` | `test_loader_and_split` | P |
| 3 | Each piece split 75 % train / 25 % test | §3.1 | `data.prepare` | `test_loader_and_split` (ratio) | P |
| 4 | Per-dataset cleaning: hex → int, strings → 0, NaN → 0, listed/empty columns dropped, LabelEncoder, MinMaxScaler (IoMT, RT-IoT, EdgeIIoT) | §4.2 | `data.SPECS`, `to_number`, `Preprocessor` | `test_data.py` (a fake file for each of the 7 formats) | P |
| 5 | Classes and dropped classes of Tables 1, 3-9 | §4.2, Tables 1, 3-9 | `data.*_MAP` | `test_dropped_classes`; groupings reproduce the Table 1 counts | P/A |
| 6 | No under- or over-sampling | §4.2 | none applied | code | P |
| 7 | Pre-processing fitted on training rows only | (user requirement) | `Preprocessor.fit(train rows)` | `test_preprocessor_is_fitted_on_training_rows_only` | D |
| 8 | CNN L1-L8: Conv1D(32, 3, relu), MaxPool1D(2), Flatten, Dense(128, relu), Dropout(0.5), BatchNorm, Dense(softmax) | Alg. 1, Fig. 4 | `model.build_cnn` | `test_architectures` | P (A1) |
| 9 | categorical cross-entropy, Adam, accuracy; fit batch 512, 10 epochs | Alg. 1 l.10-11 | `model.build_cnn`, `train_cnn` | config dump in `summary.json` | P |
| 10 | Lock (one client in the CNN at a time) + barrier | §3.2 | `ClientAgent.train` | runs to completion with 4 threads | P |
| 11 | L2 = copy of L1 | §3.2 | `ClientAgent.encrypt` | - | P |
| 12 | DP: DN = GD1 − GD2 (Gamma), L2 = L2 + A; L(μ, λ) = μ/n + Σ(γ − γ') | Alg. 2, §3.3, §3.10.1 | `privacy.laplace_gamma_noise`, `add_dp_noise` | `test_dp_shares_sum_to_laplace` (KS test: sum ~ Laplace, a single share is not) | P + F (shape 1/n) |
| 13 | λ = Δ/x, Δ = max_i \|x_i − y_i\| | §3.3 | `dp_scale_from_sensitivity`, `max_abs_difference`; `dp_scale_mode` | `test_dp_sensitivity_formula`; ablation | P (A4) |
| 14 | DK: initialisation, key pairs, public-key exchange, key agreement | §3.3, §3.10.2 | `privacy.DiffieHellmanParty` | `test_dh_offsets_cancel...` (symmetric secrets) | P + F (ECDH P-384, HKDF-SHA256) |
| 15 | Algorithm 3: add the deltas of clients before CC, subtract those after; sum into O_tot first, then L2 += O_tot | Alg. 3 | `offset_total`, `add_offset` | offsets sum to 0 for every subset of active clients; runtime check `offset_cancellation_max_err` | P |
| 16 | Deltas re-derived from part of the previous seed after each use, before the next client | Alg. 3 l.16, §3.3 | `update_deltas` (under the lock) | `test_dh_offsets_cancel...` (deltas stay in sync for 5 rounds) | P + F |
| 17 | HE Eqs. (1)-(4) on E_int; division only if no element is 0 without O_tot; B_I list-wise, or a factor of E_int for division; E_flt sign rule; D_X / D_Y | Alg. 4, §3.3 | `privacy.he_encrypt` | round-trip error < 1e-8 for offsets 0, 7919, −612345, 812345; all 4 operations and sign flips exercised; no division on zero lists | P (A6) |
| 18 | HE decryption INVERSE_Eopt, integrity check, entries removed from D_X / D_Y | Alg. 5, §3.4 | `privacy.he_decrypt` | `test_he_integrity_check_detects_tampering`, empty directory after decryption | P |
| 19 | Server: HE decryption of each client, then FedAvg | §3.4, §3.5 | `ServerAgent.aggregate` | mechanism counters; average = mean(L1 + DP) to < 1e-6 | P |
| 20 | Server: fixed offset + HE, right before each client re-obtains L2 | §3.6 | `ServerAgent.send_back` | counters (one encryption per client per iteration) | P (A7 for the value) |
| 21 | Client: HE decryption, remove fixed offset, remove DP noise | §3.7 | `ClientAgent.receive` | `test_without_noise_pp_is_lossless` (λ = 0 → identical to plain FedAvg) | P + F (A2) |
| 22 | L-A and F-A on the 25 % test part, via an evaluation model pre-initialised with the weights | §3.8 | `ClientAgent.receive`, `model.accuracy_with_weights` | `history.jsonl` | P |
| 23 | Convergence: max\|ΔW\| and max\|ΔB\| ≤ 0.0005 → client drops out | §3.8 | `ClientAgent.receive`, `run_federated` | `test_dropout_after_convergence` | P |
| 24 | Next iteration starts from the current federated L2 | §3.9 | `federated[i]` in `run_federated` | - | P |
| 25 | Latencies C1 0.6 s, C2 3 s, C3 5 s, C4 6 s; CCT and TFD | §4.4 | `config.latencies_s`, `run_federated` | `history.jsonl` | P + F (TFD formula) |
| 26 | Per-class A, P, R, F1 (one-vs-rest); Table 10 = mean over classes | §4.3, Tables 3-10 | `metrics.py` | `test_table10_is_macro_average_of_tables_3_to_9` (all 7 datasets ±0.01) | P (verified arithmetic) |
| 27 | Experiments 1 and 2: with vs without PP | §4.4 | `run.py --pp both` | results below | P |
| 28 | RQ4 literature comparison | §4.5, Tables 13-16 | not re-run (numbers from other papers) | - | - |

## Assumptions, ambiguities and deviations

Material ambiguities (A1-A5 change results). Each is a config switch, and an ablation was run for each.

- **A1 Architecture.** Fig. 4 and the text chain Dense → Dropout → BatchNorm → Output. Algorithm 1 instantiates
  Dropout and BatchNorm but calls the output layer on `dense_layer_1`, so as printed they are not in the model.
  Default: `fig4`. Alternative: `algorithm1_literal`.
- **A2 DP-noise removal (Step 7).** The paper says only "array-wise subtraction of the noise". PrivacyFL subtracts
  `own_noise / n_active` (the client's own contribution to the mean), which is the default (`own_share`).
  `own_full` subtracts the whole own noise; that injects more noise than it removes.
- **A3 What "weights and biases" (L2) are.** §1.1 calls them the learnable parameters → default `trainable`
  (BatchNorm moving statistics are not exchanged and stay local). `all` = `model.get_weights()`; see the results
  for why this fails. `conv` = only the Conv1D kernel (3, 1, 32) and bias (32): Fig. 1 shows exactly these shapes,
  and §3.1 describes one 3-D weight array plus one 1-D bias array.
- **A4 DP scale.** Neither Δ nor x (read as ε) is reported. Default λ = 0.1, the paper's worked example (§3.10.1).
  Alternative `sensitivity`: Δ = max |L1 − starting weights|, ε = 1 (PrivacyFL default).
- **A5 Initialisation.** The paper does not say whether clients share initial weights. Default: shared (FedAvg
  convention, refs [44, 51]). With independent initialisations, the first average mixes unrelated networks.

Other assumptions (no or small effect on results):

- **A6 HE details.**
  - A "list" L is one row (axis 0) of a weight array, or a whole bias vector.
  - B_I is drawn uniformly from 1..100.
  - E_int = trunc(E) and E_flt = E − E_int (the integer/fraction reading of Alg. 4).
  - The division factor B_I divides (E_int − O_p).
  - The directory stores, per list: the operation, B_I, O_p, a sign-flip mask and the encrypted values (for
    Alg. 5's integrity check). Without the sign-flip mask the scheme is not invertible when E_int = 0, so this
    field is an addition.
- **A7 Fixed server offset:** value not reported; 7919.
- **A8 Shape of the Gamma variables:** α = 1/n, scale β = λ, μ/n per share (PrivacyFL). This is what makes the
  paper's equation a Laplace mechanism.
- **A9 Piece split:** not stratified ("random sampling").
- **A10 Label groupings.** RT-IoT2022: the Table 1 counts are matched exactly only if the five NMAP_* scans are
  counted as Normal and the 37 brute-force rows are dropped. IoT-23: labels other than the six classes of Table 4
  (HeartBeat, FileDownload, Mirai, Torii, combined labels) are dropped. CIC IoT 2023 and Edge-IIoT use the dataset
  authors' official groupings; their counts match Table 1.
- **A11 Row-ID columns dropped** (RT-IoT unnamed index, BoT-IoT `pkSeqID`, IoT-23 `uid`). They are record
  identifiers that encode row order and therefore the label. Binary label columns (`label`, `attack`,
  `Attack_label`, `subcategory`) are always dropped. IPs, ports and timestamps are kept, as in the paper. They can
  inflate accuracy on these datasets; this is not addressed by the paper.
- **A12 Timing.**
  - CCT = the client's own Step 2 + Step 3 compute time (lock waits excluded).
  - TFD = max_k(2·latency_k + CCT_k) + server time + latency_c (PrivacyFL's simulated-time model).
- **A13 Final evaluation (Tables 3-10)** is not specified. Here: each client's final federated model is evaluated
  on the union of its ten 25 % test parts, and predictions are pooled over clients. No test data is used for any
  decision; convergence uses weights only.
- **A14 Libraries.** TensorFlow 2.20 / Keras 3 instead of 2.11 (Python 3.13 has no TF 2.11 wheels).
  `use_multiprocessing` no longer exists in Keras 3 `fit` and had no effect for in-memory arrays.
  The PP arithmetic runs in float64; adding ~6e5 offsets in float32 would destroy the weights, which may be the
  "loss of noise data" the authors mention in §3.3.

Deviation **D1**: pre-processing (column typing, empty-column removal, LabelEncoder, MinMaxScaler) is fitted on the
training rows only, not before splitting as in Fig. 3. Categories unseen in training get a reserved code.

Inconsistencies in the paper (noted, not resolved):

- Abstract/§5 report average recall 92.43, but the Table 10 rows give 92.28.
- §4.3 text gives a different overall average (96.83 / 95.59 / 92.45).
- Some per-class F1 values cannot follow from the printed P and R: EdgeIIoT MITM P 99.96, R 100, F1 85.29; IoMT
  DoS P 93.84, R 93.31, F1 96.48.
- Table 1's EdgeIIoT Normal count is printed "1615,64,3".

## Results

### Paper vs. this implementation

| | Paper (reported) | This implementation (obtained) |
|---|---|---|
| TON-IoT (Tables 3, 10, 11, 12) | A 99.63, P 96.48, R 93.18, F1 94.45 | **Not run.** The data cannot be downloaded in this environment |
| Other six datasets | Tables 4-12 | Not run (out of scope at the user's request; loaders are implemented and tested on fake files) |

### Actual runs: synthetic data only (verification, not a reproduction)

The data is a synthetic table of 200,000 rows, 5 overlapping classes with one rare (2 %) class, 26 features, plus the
data defects of §4.2 (hex ports, strings, NaN, an empty column, a row-ID column). The full protocol was used:
4 clients, 10 iterations, 10 epochs, batch 512, 40 pieces with 75/25 splits. Seeds 42, 1 and 2. Full tables are in
[`results/synthetic_summary.md`](results/synthetic_summary.md).

| Setting (3 seeds, mean ± std) | Overall A | Overall P | Overall R | Overall F1 | final F-A | clients converged |
|---|---|---|---|---|---|---|
| Without PP | 99.12 ± 0.07 | 97.23 ± 0.30 | 94.66 ± 0.70 | 95.85 ± 0.45 | 98.08 ± 0.44 | 0 / 12 |
| With PP (λ = 0.1) | 93.88 ± 1.82 | 79.91 ± 3.59 | 79.41 ± 4.19 | 76.38 ± 2.51 | 85.01 ± 4.49 | 0 / 12 |

**PP overhead:** mean CCT +6.6 %, mean TFD +1.9 %. The paper reports about +12 % and +8 %. Here CCT is a few
seconds and TFD is dominated by the simulated latencies, so the percentages are not comparable with the paper's
minute-long runs.

Ablations (seed 42, one change each; reference: F1 95.36 without PP, 78.92 with PP):

| Change | F1 without PP | F1 with PP | Reading |
|---|---|---|---|
| `--federated-arrays all` (BN moving statistics also noised) | - | 13.37 (NaN models in 32/40 rounds) | noise makes BN variances negative → the model collapses |
| `--dp-noise-removal own_full` | - | 42.85 | subtracting the full own noise adds noise |
| `--architecture algorithm1_literal` (no Dropout/BN in the graph) | 89.60 | 87.42 | PP costs little without BN; costs 6 F1 points without PP |
| `--federated-arrays conv` (Fig. 1 reading) | 94.75 | 78.78 | same PP cost as the default |
| `--independent-init` | 94.51 | - | small effect at the full protocol |
| `--dp-fixed-scale 0.01` | - | 95.18 | PP effect negligible |
| `--dp-fixed-scale 0.001` | - | 95.65 | PP effect negligible |
| `--dp-scale-mode sensitivity` (Δ/ε, ε = 1) | - | 89.70 | intermediate |

What this shows:

- All mechanisms execute; the counters in every `summary.json` show one DP/DK/HE pass per client per iteration.
- The DH offsets cancel in the server average; the measured error is < 1e-6 in every round.
- The encryption is lossless: λ = 0 reproduces plain FedAvg (test).
- The paper's "negligible accuracy change with PP" is reproduced only if the DP scale is about 10× or more smaller
  than its worked example, or if BatchNorm is not in the model.
- The 0.0005 convergence tolerance was never reached in any run (smallest max|ΔW| ≈ 7e-3, in the conv-only run
  without PP), so I# = 10 everywhere, unlike the paper's 2-10.

Not verified: anything on real data, all paper numbers, and runtime at the paper's data scale.
