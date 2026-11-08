# SPEC: reimplementation of Kaur (2024), GRU-assisted FL-DRL IDS, on TON_IoT

Paper: A. Kaur, IEEE Trans. Artificial Intelligence, 2024, DOI 10.1109/TAI.2024.3443787.
PDF: `kaur2024/paper/objective2A.pdf` (13 pages, author's accepted version). Page numbers below are PDF pages.

Everything for this baseline lives under `kaur2024/`. The repo's existing Chapter 6 code and
results (`cells/`, `notebooks/`, `original/`, `tests/`, `results/`) are not touched.

Status: **Phase 2 core runs in progress.** Local optimiser = plain SGD with η = 0.2 (user decision
2026-11-08, option A in §7).
User decisions (2026-11-07): IID and non-IID both run on TON_IoT with 15 clients, labelled as
the user's protocol, not the paper's (C1). Gap-fills G1-G17 are approved as proposed, to be
changed only if the code fails or behaves suspiciously. Energy constants (G11) are not studied
further. Training runs on a Kaggle GPU.

Phase 1 status:
- Paper verification (§1) and X4: done here.
- Data inspection, split, partitions and X3: `kaur2024/src/phase1_data.py`, run by you on Kaggle on
  2026-11-07. The CSV is `TON_IoT_Train_Test_Network.csv`, sha256
  `65d5465df1809b984fd10e4703bc9c012d1aefd11803773856364216f0a3520d`, 461,043 rows x 45 columns.
  The printed reports are copied into `results/phase1_data_report.md` and
  `results/extra/x3_leakage.md`. The index files (`partitions/*.npy`, `*.npz`) are still only on
  Kaggle; the script recreates them identically from the same CSV and seed.

Tags: [PAPER] stated in the paper, [GAP-FILL] needs your approval, [EXTRA] for thesis rigour only.

---

## 0. Where the PDF contradicts the task prompt (the PDF wins; your decision is needed)

| # | Prompt says | PDF says | Consequence |
|---|---|---|---|
| C1 | "Clients [PAPER, Table VIII protocol]: N = 15" on TON_IoT; IID + non-IID runs on TON_IoT | Sec. V-D, p.10: "For experimental purpose, **Edge-IIoT dataset** (FL-specific dataset) have been divided into several local datasets … client size is considered as 3, 9, and 15". Table VIII (p.11) is therefore on Edge-IIoT. The paper gives **no client count and no non-IID result for TON_IoT**. Fig. 3 (p.9) says only that the TON_IoT tuning runs were IID. | On TON_IoT, N = 15 and the whole non-IID experiment (E2) are **[GAP-FILL]**, not [PAPER]. The Table VIII format can be used for reporting, but there are no paper TON_IoT numbers to compare against. Only Table V (TON_IoT, setting unstated) and Fig. 6/7 can be compared. |
| C2 | "DRL [PAPER, … Table I(B)]: … gamma 0.99" | Table I(B) (p.7) lists only "Replay buffer size 10,000" under DRL. γ = 0.99 comes from **Fig. 5 (p.9)**, which compares γ ∈ {0.09, 0.59, 0.99}; 0.99 performs best. | Still [PAPER], but the citation is Fig. 5, not Table I(B). |
| C3 | "Local training [PAPER, Fig. 4]: batch size 20, learning rate 0.2" with Adam | Fig. 4 (p.9): η = 0.2, Λ = 20 is the best of the settings tried, but the text gives it **92.66%** accuracy (and 84.05% for η = 0.07 in Fig. 3). The headline 99.95% (Fig. 6, Table V) has **no stated η or Λ**. η is defined as the step size of plain gradient descent (Eqs. 17, 24), while Table I(B) says Adam. | lr 0.2 and batch 20 are [PAPER, Fig. 4], but the paper never says they produced Table V. Adam vs SGD is an internal contradiction in the paper. I keep Adam (Table I(B) is explicit) and lr 0.2, as you instructed; if it diverges I stop and report. |
| C4 | G7: C_dq = ‖w_i − w_G‖₂ / (total weight of participating clients) | Eq. 5 (p.4): C_dq^t = (1/τ) Σ_τ (w_i^τ − w_G^τ)/Δ_τ, i.e. a **running mean over rounds τ**, where Δ_τ = "the sum of weights of all contributing devices after the τth round". | G7 must add the 1/τ Σ_τ running mean (amended G7 below). |
| C5 | G4: Eq. 20 does not fit 10-class classification | Correct, and Eq. 21 (p.6) adds a regulariser λh(w) with λ ∈ [0,1] whose value and h are never given. Algorithm 1 line 9 says "compute loss according to (20) and (21)". | G4 also has to fix λ (amended G4 below). |
| C6 | "1 FC output layer" cited to Table I(B) | Table I(B) lists only layers = 2, hidden = 128, dropout = 0.3, Adam. The single FC layer is in the text on p.6 ("input layer, two GRU layers and one fully connected layer"). | Citation fix only. |

C1 is the one that matters: your thesis comparison on TON_IoT IID/non-IID with 15 clients becomes
"the paper's method under a protocol the paper applied only to Edge-IIoT". That is a legitimate
baseline, but it must not be described as reproducing Table VIII.

---

## 1. [PAPER] items, verified against the PDF

| Item | Value | Source |
|---|---|---|
| Classifier | GRU, 2 layers, 128 hidden units, dropout 0.3, Adam | Table I(B), p.7 |
| Output layer | one fully connected layer after the 2 GRU layers | Sec. IV-B, p.6 |
| Rounds / local epochs | 100 rounds, 5 local epochs | Table I(B), p.7 |
| Batch size, lr | Λ = 20, η = 0.2 (best tuning setting; see C3) | Fig. 4 legend and text, p.9 |
| Aggregation | w_G = Σ D_i w_i / Δ, Δ = Σ D_i over selected devices | Eq. 18 (p.5), Eq. 25 (p.6), Δ defined under Eq. 16 |
| DRL | primary + target Q-networks, experience replay, mini-batch drawn uniformly | Sec. III-B, p.5, Eqs. 12-13 |
| Replay buffer | 10,000 | Table I(B), p.7 |
| γ | 0.99 | Fig. 5, p.9 (see C2) |
| State | s_t = {ς_t, υ_t, κ_{t−1}, ζ_t, C_dq^t}: computing resources ς, υ; previous selection κ_{t−1}; SNR ζ; data quality C_dq | Eq. 6, p.4 |
| Action | a_t = (κ_i^t, P_i,u): select device i (κ = 1/0) with transmit power P | Eq. 7, p.4 |
| Reward | r = −Σ_{i∈selected} (C_training^t + C_dq^t)·a_i^t / Σ a_i^t | Eq. 8, p.4 |
| Energy | E^data = D_i ς_i υ_i² (1); E^comp = D_i ς_i K_L,i m_i υ_i² (2); E^comm = ρq / Σ_u t'_i,u B_i log₂(1+ζ_i,u) (3); ζ = P g / N₀; C_training = Σ_i (E^data+E^comp+E^comm)/N (4) | p.4 |
| Data quality | Eq. 5 (see C4) | p.4 |
| Channel/compute constants | N₀ = −174 dBm; max transmit power 30 dBm; B_max = 100 MHz; Rayleigh fading, average path loss 10⁻³; 16-bit quantisation; CPU 1100 MHz; ς = 20 cycles/bit | Sec. V-A, p.9 |
| Preprocessing | remove unwanted features; handle missing values; one-hot encode categorical features; min-max (Eq. 19) | Sec. IV-A, p.5-6 |
| Task | 10 classes: Normal, Scanning, DoS, DDoS, Ransomware, Backdoor, Injection, XSS, Password, MITM | Sec. V-A and Table II, p.8 |
| Split sizes | Table II: train 395,467 / test 89,518 (per class below) | Table II, p.8 |
| FedAvg baseline | "All the device participating in the FL training process without accessing any criteria" | Baseline 2, ref. [10], p.9 |
| Metrics | accuracy, precision, recall, F1 (Eqs. 31-34); ROC/AUC (Fig. 7); average energy (Eq. 4; Figs. 9-11) | p.8-10, p.12 |
| Reporting format | best client, worst client and global accuracy at rounds 1 and 50, IID and non-IID | Table VIII, p.11 (Edge-IIoT, see C1) |
| Paper's TON_IoT result | proposed: acc 99.95, macro P 98.63, R 99.98, F1 99.99 | Table V, p.11 |

Table II (TON_IoT), transcribed:

| Class | Train | Test | Test share |
|---|---|---|---|
| Normal | 245,000 | 55,000 | 18.3% |
| Scanning | 20,000 | 3,973 | 16.6% |
| DoS | 20,000 | 4,000 | 16.7% |
| DDoS | 20,000 | 4,000 | 16.7% |
| Ransomware | 16,030 | 3,970 | 19.9% |
| Backdoor | 20,000 | 4,000 | 16.7% |
| Injection | 20,000 | 4,000 | 16.7% |
| XSS | 13,844 | 6,116 | 30.6% |
| Password | 20,000 | 4,000 | 16.7% |
| MITM | 593 | 459 | 43.6% |
| **Total** | **395,467** | **89,518** | |

The test share varies from 17% to 44% by class, so Table II is not a single stratified split ratio.

Not reproduced (out of scope by your instruction): the FL-DRL [30] and non-FL DRL [18] baselines.

---

## 2. Gap-fills (each needs your approval)

The first block is your proposed gap-fills, amended where the PDF requires it. The second block is
new gaps that the code cannot run without.

### G1: data file and split  [GAP-FILL]
- File: `/kaggle/input/datasets/fadiabuzwayed/ton-iot-train-test-network/TON_IoT_Train_Test_Network.csv`
  (sha256 above). No NaN anywhere; `-` is used for "not applicable".
- **[Certain, from the run]** The file has 461,043 rows: 300,000 normal, 20,000 of each of the
  nine attack classes, and 1,043 MITM. It has 45 columns; the paper says 46 features (p.7).
  **Table II cannot be drawn from it.**
  - Table II needs 24,000 DoS, DDoS, Backdoor, Injection and Password rows (20,000 train + 4,000
    test), 23,973 Scanning and 1,052 MITM. Its total, 484,985, is larger than the whole file.
  - The paper therefore probably sampled from the full TON_IoT network data, not this subset.
- Proposed rule (implemented in `phase1_data.py`, seed 0): for each class, take exactly Table II's
  test count at random, then up to Table II's train count from what remains. Every shortfall is
  reported.
  - Result: test = 89,518 (exactly Table II); train = 371,485 (Table II is 395,467).
  - Six classes get 16,000 or 16,027 train rows instead of 20,000, MITM gets 584 instead of 593,
    and 40 XSS rows go unused.
- Alternative: one stratified 80/20 split. It is simpler, but it would not match Table II's test
  counts. I recommend the rule above, because it keeps the test set identical in size and class
  mix to the paper's, and the test set is what the comparison is scored on.

### G2: dropped features  [GAP-FILL] (as proposed)
- Drop `ts, src_ip, dst_ip, src_port, dst_port, label`; the target is `type`.
- Any string column with more than 50 unique values is dropped instead of one-hot encoded. The
  rule is applied to **training rows only** in Phase 2. Phase 1 reports cardinality on the whole
  file, for information only.
- **[Certain, from the run, whole-file counts]** Only two columns cross the >50 threshold and
  are dropped: `dns_query` (14,149 unique) and `http_uri` (74). My earlier guess that the SSL and
  user-agent columns would be dropped was wrong. `http_user_agent` has 36 values and is one-hot
  encoded.
  - Phase 2 recomputes this on training rows only. [Likely] The result is the same, but
    `http_uri` (74 in the whole file) is the one to watch.
- The remaining 25 string columns are one-hot encoded; on whole-file categories that gives 149
  columns.
- Kept numeric columns (14): duration, src_bytes, dst_bytes, missed_bytes, src_pkts,
  src_ip_bytes, dst_pkts, dst_ip_bytes, dns_qclass, dns_qtype, dns_rcode, http_request_body_len,
  http_response_body_len, http_status_code.
- [Likely] The input width is d ≈ 163, a little lower if some categories never occur in train.
- G2b, missing values: the file has no NaN. `-` ("not applicable") stays as its own category in
  every string column. `http_trans_depth` and `weird_addl` look numeric, but they are text with
  `-` and have 11 and 4 values, so they are one-hot encoded like the other string columns.
  Sec. IV-A only says "handling missing values, if any".
- G2c: test categories not seen in training are encoded as all zeros.
- G2d: Eq. 19 min-max is fitted on the training rows only, and test values are clipped to [0, 1].
  The clipping is my addition. It uses no test information, but say if you would rather not clip.

### G3: GRU input  [GAP-FILL] (as proposed)
- Each record is a length-1 sequence of shape (1, d). The paper claims "temporal dependencies" but
  never says how records form sequences.
- [Likely] With length 1, the GRU acts like a gated MLP. This is the honest reading of a
  per-record classifier.

### G4: loss  [GAP-FILL] (amended per C5)
- Softmax cross-entropy over 10 classes, with **λ = 0** (no Eq. 21 regulariser), because neither λ
  nor h(·) is given.
- Dropout 0.3 is applied between the GRU layers and before the FC layer. Where exactly the paper
  applies it is not stated.

### G5: selection and action  [GAP-FILL] (refined)
- The DQN outputs **one Q-value per (client, power level)**: 15 × 3 = 45 outputs, power in
  {10, 20, 30} dBm. This implements Eq. 7's pair (κ, P).
  - Client score = max over power. Select the top 8 clients; each uses its argmax power.
- ε-greedy, decaying linearly from 1.0 to 0.05 over rounds 1-50, then fixed at 0.05. An
  exploratory pick is 8 random clients, each with a random power.
- Joint-action Q (needed because Eq. 8 gives one reward per round):
  Q(s, a) = mean of the 8 selected Q-values. The target is r + γ × (mean of the top 8 of the
  target network's per-client max Q at s′).
- **Consequence to accept:** one transition per round means the DQN sees at most 100 transitions
  in a whole run, so the 10,000 replay buffer never fills.
  - [Likely] The policy will stay close to its exploration schedule.
  - X2 (random top-8) measures exactly this. Nothing in the paper says how many transitions per
    round it used.

### G6: DQN network  [GAP-FILL] (as proposed, plus details)
- MLP with 2 × 64 ReLU units; Adam, lr 1e-3; target network synced every 10 updates; replay batch
  32.
- One gradient update per round once the buffer holds 32 or more transitions. So the first update
  happens at round 32 and the target syncs about 7 times in 100 rounds.
- Huber loss? **No:** MSE, as in Eq. 12.
- State input (Eq. 6), 15 clients × 5 features = a 75-dimensional vector:
  [ς_i, υ_i, κ_i^{t−1}, ζ_i^t (in dB), C_dq,i^t].
  - ς and υ are constants for every device (20 cycles/bit, 1100 MHz; Sec. V-A), so they carry no
    information. They are kept only to stay literal to Eq. 6.
- G6b, state scaling: each feature is divided by a fixed constant (ς/20, υ/1100 MHz, ζ_dB/100);
  C_dq is used as is. These constants are not in the paper.

### G7: data quality C_dq  [GAP-FILL] (amended per C4)
- C_dq,i^t = (1/n_i) Σ over rounds τ ≤ t in which i participated of ‖w_i^τ − w_G^τ‖₂ / Δ_τ.
  - w are all GRU and FC parameters flattened. w_G^τ is the global model **after** aggregation in
    round τ.
  - Δ_τ = Σ D_j over the clients that participated in round τ (the same Δ as Eq. 18).
  - n_i = the number of rounds i has participated in. That is the paper's 1/τ, applied per device,
    since unselected devices have no w_i^τ.
- A client that has never been selected has C_dq = 0. The L2 norm is the gap-fill: the paper's
  numerator is a vector.

### G8: non-IID partition  [GAP-FILL] (as proposed, plus G8b)
- Dirichlet label skew with α = 0.5 over 15 clients, drawn per class.
- G8b: redraw until every client has at least 20 samples (one batch), up to 1,000 tries; the number
  of tries is logged.
- The plot is saved to `results/partitions/class_distribution_noniid_dir0.5.png`.

### G9: fixed partition seed  [GAP-FILL] (as proposed)
- One `numpy.default_rng(0)` stream generates, in order: the split, then the IID partition, then
  the Dirichlet partition. All are saved once, with the CSV's sha256, in `manifest.json`.
- Training seeds touch only model init, batch order, dropout, the channel draw and DQN exploration.

### New gaps the code needs

**G10 K_L** (your request to interpret it). Three readings exist:
- K_L appears in Eq. 2 as "number of computational rounds for local training".
- Algorithm 1 lists it as an input called "local communication rounds K_L", but never uses it in
  the loop.
- Fig. 4 tries K_L = 20 and 40, alongside an x-axis labelled "Epoch" running to 180, which matches
  neither 100 rounds nor 5 local epochs.

Interpretation: the local work that is actually performed is what Table I(B) fixes, 5 local epochs
per round. K_L enters only the energy model, and I set **K_L = 5 = local epochs**, so that Eq. 2
measures the computation that really happens. Fig. 4's K_L = 20/40 is read as a separate tuning
sweep, not the Table V setting. The alternative, K_L = 40 (the best curve in Fig. 4(b)), would
contradict Table I(B)'s 5 local epochs.

**G11 energy units and missing constants.** As written, Eqs. 1-3 are not dimensionally closed.
Proposed:
- D_i in bits = n_i × d × 16. This uses the paper's 16-bit quantisation, because ς is in
  cycles/bit.
- An effective switched-capacitance coefficient **κ_cpu = 10⁻²⁸** multiplies Eqs. 1 and 2.
  - It is absent from the paper; without it, E^data ≈ D_i × 2.4 × 10¹⁹ J.
  - 10⁻²⁸ is the usual value in the FL energy literature the paper cites ([11], [35], [36]).
- m_i = model size in Mbit at 16 bits per parameter.
- q (Eq. 3) = number of model parameters × 16 bits.
- ρ = P_i in watts, so E^comm = P × time = P q / rate. The paper calls ρ "normalisation for
  communication energy efficiency" with no value.
- One sub-channel per device (U = 1, t′ = 1), with **B_i = B_max / (number of selected devices)**,
  which satisfies Σ B_i ≤ B_max.
- N₀ = −174 dBm is read as a noise **density** (dBm/Hz), so the noise power is N₀ × B_i. Read
  literally as −174 dBm in total, the SNR would be about 10¹⁸.
- g_i = 10⁻³ × Exp(1), redrawn for each device every round from the training seed.
- **Consequence:** absolute joules depend on all of these and cannot be compared with the paper's
  Figs. 9-11 (they have no stated units). Comparisons between methods within this reimplementation
  are valid, because every method uses the same model.

**G12 energy for FedAvg** (no power selection): all 15 clients transmit at 30 dBm (the stated
maximum), with B_i = B_max / 15. Random top-8 (X2): random clients, each with a random power from
{10, 20, 30} dBm.

**G13 multiclass metrics:**
- Eq. 31 is the binary formula, so accuracy = standard multiclass accuracy.
- AUC = macro one-vs-rest ROC-AUC on softmax scores. Per-class AUC and a micro-average are also
  reported.
- Fig. 7's single ROC curve is plotted as micro-average OvR.
- Macro P/R/F1 = unweighted mean over the 10 classes. Weighted versions are reported alongside.

**G14 best/worst client accuracy** (Table VIII columns): the accuracy of each client's **local
model after its local training in that round**, on the global test set; best and worst are taken
over the clients that participated that round. The paper does not say whether this is a local or
global test set.

**G15 round indexing:** "round 1" = after the first aggregation; "round 50/100" = after the 50th
and 100th aggregations. The global model is saved at both.

**G16 Eq. 8 scale:** C_training (joules) and C_dq (dimensionless, divided by Δ of about 10⁵) are
added unscaled, as written. [Likely] The energy term will dominate the reward. No rescaling is
proposed, to stay faithful; say if you want one.

---

## 3. X4 [EXTRA]: Table V is internally inconsistent (done)

Full output: `kaur2024/results/extra/x4_table5_consistency.md` (from
`python kaur2024/tools/x4_table5_consistency.py`).

- [Certain] For the proposed model, the per-class values cannot produce the reported summary
  numbers:
  - Macro recall: reported 99.98, but the per-class mean is **93.38**. DoS 80.98, Injection 84.89
    and XSS 76.45 make 99.98 impossible.
  - Accuracy: reported 99.95, but Σ recall_c × n_c / N with Table II test counts gives **96.47**.
    That is the exact accuracy implied by the per-class recalls.
  - Macro F1: reported 99.99, but the per-class mean is 95.02.
- [Certain] Every one of the proposed model's per-class F1 values is below the harmonic mean of its
  own P and R (Normal: 98.09 reported vs 99.36 implied). So the F1 row was not computed from the
  P and R rows.
- [Certain] FL-DRL's reported accuracy, 84.25, is lower than the 92.01 implied by its recalls.
  FedAvg is roughly consistent: 68.07 reported vs 69.87 implied.
- Implication for Phase 3: the paper's headline 99.95% is not a reliable target. I will compare
  against the per-class rows **and** the implied 96.47% / macro F1 95.02, and say so in REPORT.md.

## 4. X3 [EXTRA]: leakage (done; full output in `results/extra/x3_leakage.md`)

| Key | Test rows also in train | ...of which train has a different `type` | Dups within train / test |
|---|---|---|---|
| All columns except `type`/`label` (includes ts, IPs, ports) | 2,708 / 89,518 (3.03%), all normal | 0 | 8,160 / 948 |
| Model features (after G2 drops) | **54,064 / 89,518 (60.39%)** | **8,909** | 220,061 / 49,743 |

Shared test rows by class (model-feature key): normal 33,948, ransomware 3,963, backdoor 3,710,
xss 3,484, scanning 3,117, dos 2,883, ddos 1,789, password 1,109, injection 57, mitm 4.

What this means [Likely]:
- Once identifiers and timestamps are dropped, most of the test set consists of exact copies of
  training feature vectors. High accuracy here is largely lookup.
  - The effect is uneven: it is almost all of ransomware (3,963 of 3,970) and backdoor test rows,
    but almost none of injection or MITM.
- 8,909 test rows have a feature vector that train also labels with a different class. No model
  can be right on all of these, which caps the achievable accuracy below 100%.
  - The exact cap depends on the label mix per vector and will be computed in Phase 2.
- This applies equally to your own model if it is scored on the same split, so comparisons
  between models stay fair. Absolute numbers, though, are inflated relative to unseen traffic.

**G17 [GAP-FILL / EXTRA]: proposal for handling this.**
- Do not deduplicate the core runs. The paper doesn't, and the test set must stay as in Table II.
- Every evaluation also reports the full metric set on the **"unseen" test subset**: the 35,454
  test rows whose model-feature vector does not occur in train. This is [EXTRA] and reported
  separately.
- The subset is defined from train-vs-test feature equality only, without labels or predictions,
  so no test information enters training or any tuning.
  - The mask is saved to `results/partitions/` in Phase 2 so your model is scored on exactly the
    same rows.
  - It has to be computed on the encoded features, so its size may shift slightly from 35,454.

## 5. Compute estimate (needed for Phase 2 planning)
[Likely] On this container (4 CPUs, no GPU):
- One IID DQN run is about 8 clients × ~23k samples / 20 per batch × 5 epochs ≈ 46k Adam steps per
  round, so ≈ 4.6M steps per run. That is roughly 1-2 h per run on CPU; FedAvg (15 clients) takes
  about twice as long.
- X1 + X2 total 30 runs, so on the order of **50-80 CPU-hours**. A GPU on Kaggle would be much
  faster.

---

## 6. Phase 2 implementation notes (new [GAP-FILL] items, same approval rule)

**G18 SNR in the DQN state.** The state (Eq. 6) is built before the action, so its SNR uses
maximum power (30 dBm) and bandwidth B_max / 8. The SNR used in Eq. 3 then uses the chosen
power and B_max / (number selected).

**G19 local optimiser details.**
- Adam with β = (0.9, 0.999) and ε = 1e-7 (the Keras defaults; the paper used Keras, Table I(A)).
- A fresh optimiser for each client in each round.
- The last batch of an epoch may be smaller than 20 (Keras default).

**G20 Eq. 4 for unselected devices.** C_training = Σ over selected devices of
(E_data + E_comp + E_comm) / N, with N = 15. Unselected devices use no energy in that round.

**Speed-up (implementation only, no modelling change).** All clients selected in a round train
in parallel as one stacked tensor on the GPU, using the exact equations of `torch.nn.GRU`. Tests
in `kaur2024/tests/test_kaur_core.py` check that:
- the stacked forward pass equals `nn.GRU` + `nn.Linear` for sequence lengths 1 and 3;
- the stacked Adam equals `torch.optim.Adam`;
- a client trained together with others ends with exactly the same weights as when trained alone,
  including clients of different sizes;
- a small client gets exactly its own number of steps;
- the training path learns a learnable synthetic task (at lr 0.01; this test only checks the code
  works and says nothing about the paper's setting).

**Expected behaviour to watch for (not changed, reported).**
- [Likely] With lr = 0.2, Adam may collapse to predicting "normal" (61.4% of test rows) or
  diverge. `fl.py` stops on non-finite weights and logs the largest predicted-class share every
  round. Per the task rules, a divergence or collapse is reported to you, not fixed.
- [Likely] Under Eq. 8 the reward is dominated by energy, and energy grows with D_i. In the
  non-IID setting the DQN will therefore learn to prefer **small** clients. This is the paper's
  reward as written; X2 (random selection) shows what it costs or gains.

---

## 7. Smoke test on real data: Adam + lr 0.2 collapses (2026-11-08, run on CPU in this container)

The CSV was uploaded as a zip; its sha256 matches the Kaggle file. Re-running `phase1_data.py`
here gave identical split and partitions (all counts equal). The only differences in the report
are cosmetic: the pandas version labels the text dtype `str` instead of `object`, and one tie
changes the order of the example `http_uri` values.

**E1 (IID, DQN), seed 0, 2 rounds, configuration as approved (Adam, lr 0.2, batch 20, 5 epochs)**
(`results/extra/smoke/e1_iid_dqn_adam_lr0.2_rounds2/`):

| round | global acc | macro F1 | macro AUC | predicted-class share | mean train loss | local models |
|---|---|---|---|---|---|---|
| 1 | 61.44% | 7.61% | 0.512 | 100% one class | 2.78 | 6 predict only Normal (61.44%), 2 only one other class (4.47%) |
| 2 | 61.44% | 7.61% | 0.442 | 100% one class | 1.96 | 7 only Normal, 1 only one other class (4.44%) |

- 61.44% = 55,000 / 89,518: every test row is predicted "Normal".
- The round-1 training loss (2.78) is above that of uniform guessing over 10 classes (ln 10 = 2.30).
- About 70 s per IID round on this CPU.

**Diagnostic** (`tools/diag_lr_collapse.py`, output in `results/extra/diag_lr_collapse.txt`; not a
result, nothing tuned). One client, 1 epoch, the same code:

| optimiser | lr | train loss | test acc | predicted-class share |
|---|---|---|---|---|
| Adam | 0.2 (configured) | 2.20 | 61.44% | 100% (collapsed) |
| SGD | 0.2 (Eqs. 17/24 reading) | 0.89 | 69.43% | 58.9% |
| Adam | 0.001 (framework default) | 0.75 | 73.98% | 59.7% |

Conclusion: the collapse comes from Adam at step size 0.2, not from a bug (the same code learns at
lr 0.001, and with SGD at 0.2). The paper is internally inconsistent here (C3): Table I(B) says
Adam, while Eqs. 17 and 24 define η as a plain gradient step. Fig. 4's η = 0.2 fits the SGD reading.

Facts confirmed on real data:
- d = 158 input features.
- Dropped for having more than 50 unique values in train: `dns_query` and `http_uri`.
- Unseen test subset (G17), on encoded features: 35,438 rows (35,454 when counted on raw strings
  in X3).

**Decision needed (the code is NOT changed until you choose):**
- A. SGD with η = 0.2: keeps the paper's numeric value and Eqs. 17/24, and departs from Table I(B)'s
  "Adam". *Recommended.*
- B. Adam with lr = 1e-3 (Keras default): keeps Table I(B), and departs from Fig. 4's η.
- C. Keep Adam with 0.2: every core result would be the 61.44% collapse.

**Decision (2026-11-08): option A.** Local training uses plain gradient descent with η = 0.2
([PAPER] Eqs. 17/24, Fig. 4). This departs from Table I(B)'s "Adam" and is reported as such.
Smoke test with SGD (E1, 2 rounds, `results/extra/smoke/e1_iid_dqn_sgd_lr0.2_rounds2/`):
- accuracy 75.82% after round 1 and 76.44% after round 2;
- largest predicted-class share 0.72;
- mean train loss 0.60;
- about 48 s per round on 4 CPU threads.

---

## 8. Early finding during the core runs (2026-11-08): accuracy plateaus at ~76.7% (IID)

- E1 global accuracy: 75.82% at round 1, 76.67% at round 10. The training loss is falling only
  slowly (0.60 to 0.56).
- At round 1, recall is 0 for Ransomware, Injection and MITM, and about 5% for DoS and Password.
  Ransomware and Injection are mostly predicted as Normal.
- **Not a code bug, and not a limit of the data** (`tools/diag_duplicate_labels.py`, output in
  `results/extra/diag_duplicate_labels.txt`). 3,661 of the 3,970 Ransomware test rows have an exact
  copy in train labelled Ransomware, so even a lookup would reach 92% recall on that class.
- **Cause [Certain]: Eq. 19 min-max on heavy-tailed columns.** After scaling, the 99th percentile
  of src_bytes is 1e-6, of dst_ip_bytes 1.9e-4 and of duration 7e-4; the medians are about 0. The
  numeric features are effectively invisible to the network, which therefore works almost only from
  the one-hot flags (proto, conn_state, service, dns_*, ...). On those flags, Ransomware
  (tcp / OTH / service '-') overlaps a large group of Normal traffic.
- Dropping ports (G2) also removes what is probably a strong signal for Injection, Ransomware and
  MITM.
- **Not changed.** Eq. 19 is [PAPER] and G2 is approved. Changing either to raise accuracy would be
  tuning towards the paper's numbers. Both are recorded as candidate explanations for the gap to
  Table V. A separately labelled [EXTRA] ablation (e.g. keeping ports, or log-scaling before
  min-max) can be run later if the user asks for it.
