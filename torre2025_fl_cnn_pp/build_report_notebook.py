"""Build notebooks/torre2025_reimplementation_TON_IoT.ipynb: a self-contained report notebook (method, checks,
experiments on TON-IoT, results vs the paper, discussion) meant to be executed once and shared with its outputs.

    python -m torre2025_fl_cnn_pp.build_report_notebook
    # execute (writes the outputs into the file):
    TON_DIR=/path/to/folder/with/the/csv jupyter nbconvert --to notebook --execute --inplace \
        notebooks/torre2025_reimplementation_TON_IoT.ipynb
"""
import os

import nbformat as nbf

from .build_kaggle_notebook import MODULES

PKG = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(PKG), "notebooks", "torre2025_reimplementation_TON_IoT.ipynb")

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell

TITLE = """# Reimplementation of Torre et al. (ACM TOSEM 2025) on TON-IoT

**Paper.** D. Torre, A. Chennamaneni, J. Jo, G. Vyas, B. Sabrsula. *Toward Enhancing Privacy Preservation of a
Federated Learning CNN Intrusion Detection System in IoT: Method and Empirical Study.* ACM Transactions on Software
Engineering and Methodology 34(2), Article 53, 2025. https://doi.org/10.1145/3695998

**What this notebook does.** It reimplements the paper's complete method: a 4-client federated 1D-CNN intrusion
detector with three privacy-preservation (PP) layers, which are Differential Privacy (DP), Diffie-Hellman (DH) key
exchange and the paper's "Homomorphic Encryption" (HE). It runs the paper's two experiments (FL without PP, FL with PP)
on the **TON-IoT** network dataset only, and compares the results with the paper's Tables 3, 10, 11 and 12.

**How to read it.** Every cell has been executed and its output is saved.
- Sections 1–2: the method and the assumptions.
- Section 3: the source code (it can be skipped).
- Sections 4–8: checks, experiments and results.
- Section 9: the discussion.

**Sources of information.** The paper has no public code. Every implementation choice is labelled as one of:
- **[PAPER]**: stated by the authors.
- **[PRIVACYFL]**: taken from the PrivacyFL simulator (Mugunthan et al., ref. [59]). The paper builds its DP and DH
  layers on PrivacyFL, but it is not the authors' own code.
- **[ASSUMPTION]**: not reported in the paper, so chosen here."""

METHOD = """## 1. The method (as reimplemented)

**Federated learning protocol (Fig. 3, Sections 3.1–3.9) [PAPER]**
- 4 clients and 1 server; up to 10 FL iterations.
- The dataset is shuffled and cut into **40 equal pieces**, one per (client, iteration). Each piece is split
  **75 % train / 25 % test**.
- In each iteration, every client does the following:
  1. **Step 2:** trains the CNN on its piece, starting from the last federated model, and obtains L1. A lock and a
     barrier serialise CNN training across clients.
  2. **Step 3:** copies L1 to L2 and encrypts L2 with DP, then DH, then HE.
- The server then does the following:
  1. **Step 4:** decrypts the HE of each client.
  2. **Step 5:** averages the client models (FedAvg).
  3. **Step 6:** adds a fixed offset and re-applies HE for each client.
- Each client then does the following:
  1. **Step 7:** reverses the HE, removes the fixed offset, then removes its DP noise.
  2. **Step 8:** measures the local (L-A) and federated (F-A) accuracy on its 25 % test part.
  3. Leaves training when `max|federated − local| ≤ 0.0005` holds for both weights and biases.
- Simulated client latencies are 0.6, 3, 5 and 6 s, used to compute the client computation time (CCT) and the
  time to receive the federated data (TFD).

**CNN (Algorithm 1, Fig. 4) [PAPER]**
- Layers: Input → Conv1D(32, kernel 3, ReLU) → MaxPool1D(2) → Flatten → Dense(128, ReLU) → Dropout(0.5) →
  BatchNormalization → Dense(softmax).
- Training: categorical cross-entropy, Adam, batch 512, 10 epochs per iteration.

**DP layer (Algorithm 2, Sections 3.3 and 3.10.1)**
- Each element gets `μ/n + Gamma(1/n, λ) − Gamma(1/n, λ)`. Summed over the n clients, these shares are exactly
  Laplace(μ, λ), which is the paper's equation `L(μ,λ) = μ/n + Σ(γ_p − γ'_p)`.
- The Gamma shape 1/n comes from **[PRIVACYFL]**.
- λ is not reported for the experiments. The default λ = 0.1 is the paper's worked example (§3.10.1).

**DH layer (Algorithm 3) [PAPER + PRIVACYFL]**
- Pairwise ECDH secrets are turned into integer deltas.
- Each client adds the deltas of the clients listed before it and subtracts those of the clients after it, as one
  total offset `O_tot`. The offsets therefore cancel exactly in the server's average.
- The deltas are re-derived after every use.

**HE layer (Algorithms 4 and 5) [PAPER]**
- For each list of values, one operation is chosen at random (add, subtract, multiply or divide), together with an
  integer `B_I`. The operation is applied to the integer part of each element, as in Eqs. (1)–(4).
- Division is never chosen for a list whose operand is zero; for division, `B_I` is a factor of the operand.
- A sign rule keeps the integer and fractional parts consistent.
- The key material is stored in directories `D_X` / `D_Y`, which the server uses to decrypt **before** averaging.

**Metrics (Section 4.3) [PAPER]**
- Per-class one-vs-rest accuracy, precision, recall and F1.
- The overall row (Table 10) is the mean over the classes. This was checked against the paper: averaging Table 3
  reproduces Table 10 exactly."""

ASSUMPTIONS = """## 2. Assumptions, deviations and known issues in the paper

**Choices that affect results** (each one is a switch, and an ablation was run for it in Section 7):

| ID | Item | Choice here | Why |
|---|---|---|---|
| A1 | BatchNorm / Dropout | in the model (Fig. 4) | Algorithm 1 creates them but never connects them; Fig. 4 and the text include them |
| A2 | DP noise removal (Step 7) | subtract own noise ÷ active clients | [PRIVACYFL]; the paper only says "subtraction of the noise" |
| A3 | What "weights and biases" (L2) are | the trainable parameters | §1.1: "the learnable parameters"; adding noise to BatchNorm's running variance makes it negative (NaN models) |
| A4 | DP scale λ | 0.1 | the paper's worked example; neither Δ nor ε is reported |
| A5 | Initial weights | the same for all clients | FedAvg convention; not stated in the paper |
| A15 | Feature scaling for TON-IoT | MinMaxScaler | §4.2 lists none for TON-IoT, but raw values reach 3.9 × 10⁹ and the CNN then predicts almost only Normal |

**Other assumptions**
- HE: `B_I` is drawn from 1..100. The server's fixed offset is 7919. A sign-flip record is added to the directory,
  without which decryption is not exact.
- The 40 pieces are split at random, without stratification.
- The binary `label` column is dropped, because it is a copy of the target.
- The final per-class evaluation is each client's final federated model on its own 10 test parts, pooled over the
  4 clients. The paper does not say how its tables were computed.
- Libraries: TensorFlow 2.20 / Keras 3, where the paper used 2.11. The PP arithmetic runs in float64.

**Deliberate deviation**
- Preprocessing is fitted on the training rows only, to avoid test leakage. The paper fits it before splitting.

**Issues found in the paper**
1. **The "HE" layer is not homomorphic encryption.** The server reverses it before averaging, so no computation is
   ever done on ciphertexts, and the server can read every client's (noised, offset) model.
2. **No ε is reported for the DP layer.** The weights are never clipped, so the sensitivity is unbounded, and the
   noise cannot be calibrated to any finite ε as specified.
3. **Internal inconsistencies.**
   - The average recall in the abstract (92.43) differs from the one Table 10 gives (92.28).
   - Some per-class F1 values cannot follow from the reported precision and recall. EdgeIIoT MITM is reported with
     P 99.96, R 100 and F1 85.29.
4. **The dataset has 11,071 exact duplicate rows (2.4 %).** They are kept here, as in the paper, so a row can appear
   in both a training and a test part."""

SRC_MD = """## 3. Source code

The package is written to disk below, so the notebook runs on its own. It is the same code as `torre2025_fl_cnn_pp/`
in the repository, which also has 24 unit tests.
- `privacy.py`: DP, DH and HE.
- `fl.py`: the client and server agents.
- `data.py`: loading and preprocessing.
- `model.py`: the CNN.

**This section can be skipped.**"""

SETUP = """import os, glob, json, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# ---- settings ------------------------------------------------------------------------------------------
DATA_DIR = os.environ.get("TON_DIR", "/kaggle/input/datasets/fadiabuzwayed/ton-iot-train-test-network")
RESULTS = os.environ.get("RESULTS_DIR", "results_torre2025_report")
SEEDS = [42, 1, 2]
# ---------------------------------------------------------------------------------------------------------
csvs = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))
DATA_FILE = ([c for c in csvs if "train_test_network" in os.path.basename(c).lower()] or csvs)[0]
print("dataset file:", os.path.basename(DATA_FILE))

import tensorflow as tf, keras, sklearn
tf.get_logger().setLevel("ERROR")
print("python packages: tensorflow", tf.__version__, "| keras", keras.__version__, "| scikit-learn",
      sklearn.__version__, "| numpy", np.__version__, "| pandas", pd.__version__)"""

DATA_MD = """## 4. The dataset

This is the TON-IoT network `Train_Test_Network` file, the one the paper uses (§4.2). The checks below compare it with
the paper's Table 1."""

DATA = """raw = pd.read_csv(DATA_FILE, low_memory=False)
print("rows x columns:", raw.shape, "(paper: 461,043 records)")
paper_t1 = {"normal": 300000, "backdoor": 20000, "ddos": 20000, "dos": 20000, "injection": 20000, "password": 20000,
            "ransomware": 20000, "scanning": 20000, "xss": 20000, "mitm": 1043}
counts = raw["type"].value_counts()
t1 = pd.DataFrame({"rows in file": counts, "paper Table 1": pd.Series(paper_t1)})
t1["match"] = t1["rows in file"] == t1["paper Table 1"]
display(t1)
print("missing values:", int(raw.isna().sum().sum()),
      "| binary 'label' agrees with 'type' on every row:", bool(((raw["label"] == 1) == (raw["type"] != "normal")).all()),
      "| exact duplicate rows:", int(raw.duplicated().sum()))

from torre2025_fl_cnn_pp.config import Config
from torre2025_fl_cnn_pp.data import prepare
fd = prepare("ton_iot", DATA_FILE, Config())
info = fd.info
print(f"after preprocessing: {fd.n_features} features | {info['n_pieces']} pieces of {info['rows_per_piece']} rows "
      f"(75/25 split each) | removed columns: {info['removed_columns']}")
tr = np.concatenate([p["y_train"] for p in fd.pieces.values()]); te = np.concatenate([p["y_test"] for p in fd.pieces.values()])
display(pd.DataFrame({"train rows": np.bincount(tr, minlength=10), "test rows": np.bincount(te, minlength=10)},
                     index=fd.classes))
del raw"""

CHECKS_MD = """## 5. Checks of the three PP layers

These checks run live, before any experiment:
- **HE.** Encryption followed by decryption must return the model exactly, including the single-client case (offset
  0), where the sign rule matters.
- **DH.** The four clients' offsets must sum to exactly 0, so they vanish in the server's average.
- **DP.** The four clients' noise shares must add up to a Laplace(0, λ) distribution. A single share must not be
  Laplace on its own.
- **Losslessness.** With λ = 0, the full PP pipeline (DP, DH, HE, fixed offset, decryption) must give the same
  federated model as plain FedAvg."""

CHECKS = """from scipy import stats
from torre2025_fl_cnn_pp import privacy as pp
rng = np.random.default_rng(0)
model = [rng.normal(0, .2, (3, 1, 32)), rng.normal(0, .1, 32), rng.normal(0, .1, (416, 128)), rng.normal(0, .1, 128)]
names = ["C1", "C2", "C3", "C4"]
parties = pp.diffie_hellman_setup(names)
offsets = {n: parties[n].offset_total(set(names)) for n in names}
print("DH offsets per client:", offsets, "-> sum =", sum(offsets.values()))
err = 0.0
for o in list(offsets.values()) + [0]:
    x = pp.add_offset(model, o); enc, d = pp.he_encrypt(x, o, 100, rng); dec = pp.he_decrypt(enc, d)
    err = max(err, max(float(np.max(np.abs(a - b))) for a, b in zip(dec, x)))
print(f"HE encrypt -> decrypt, max error over 5 offsets: {err:.1e}")
total = sum(pp.laplace_gamma_noise([(200000,)], 4, 0.0, 0.1, rng)[0] for _ in range(4))
one = pp.laplace_gamma_noise([(200000,)], 4, 0.0, 0.1, rng)[0]
print(f"DP: sum of 4 shares vs Laplace(0, 0.1), KS p-value = {stats.kstest(total, 'laplace', args=(0, .1)).pvalue:.3f} "
      f"(should be large) | one share alone, p = {stats.kstest(one, 'laplace', args=(0, .1)).pvalue:.1e} (should be ~0)")

from torre2025_fl_cnn_pp.fl import run_federated
small = Config().with_overrides(epochs=1, batch_size=64, max_iterations=1, synthetic_rows=4000, dp_fixed_scale=0.0)
sd = prepare("synthetic", None, small)
a, b = run_federated(sd, small, use_pp=False), run_federated(sd, small, use_pp=True)
diff = max(abs(x["fed_acc"] - y["fed_acc"]) for x, y in zip(a["history"], b["history"]))
print(f"lambda = 0: largest federated-accuracy difference, PP pipeline vs plain FedAvg = {diff:.2e} | "
      f"offsets cancel in the server average to {max(b['offset_cancellation_max_err']):.1e}")"""

EXP_MD = """## 6. Experiments on TON-IoT

**Main experiment.** The paper's protocol is used in full: 4 clients, up to 10 iterations, 10 epochs, batch 512, and
convergence tolerance 0.0005.
- Experiment 1 runs FL without PP; experiment 2 runs FL with DP, DH and HE.
- Both are run for 3 seeds (42, 1, 2). The paper reports one run without a seed.

**Ablations.** Each changes one ambiguous choice, using seed 42:
- no feature scaling, as §4.2 literally says (A15);
- a 10× smaller DP scale, λ = 0.01 (A4);
- λ = Δ/ε with ε = 1 (A4);
- subtracting the full own noise (A2);
- Algorithm 1 as printed, with no BatchNorm or Dropout (A1);
- exchanging only the Conv1D layer, the reading suggested by Fig. 1 (A3).

Each run's log, per-iteration history and summary are written to `RESULTS/ton_iot/`. The whole section takes about
40 minutes on a 4-core CPU."""

EXP = """import contextlib, io
from torre2025_fl_cnn_pp.run import main as _main
QUICK = os.environ.get("NB_QUICK") == "1"          # only for testing this notebook; never for results
EXTRA = ["--epochs", "1", "--max-iterations", "2", "--max-rows", "8000"] if QUICK else []
def run(args):
    \"\"\"One experiment; its full log goes to RESULTS/ton_iot/<run>/run.log, one summary line is printed here.\"\"\"
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        _main(args + EXTRA)
    tag = args[args.index("--tag-suffix") + 1]
    for f in sorted(glob.glob(os.path.join(RESULTS, "ton_iot", f"*{tag}", "summary.json"))):
        o = json.load(open(f))["overall"]
        print(f"{os.path.basename(os.path.dirname(f)):32s} A={o['A']:.2f} P={o['P']:.2f} R={o['R']:.2f} F1={o['F1']:.2f}")
    print(f"   ({(time.time() - t) / 60:.1f} min)")
t0 = time.time()
for s in SEEDS:
    run(["--dataset", "ton_iot", "--data-path", DATA_FILE, "--results-dir", RESULTS, "--pp", "both",
         "--seed", str(s), "--tag-suffix", f"_seed{s}"])
ABLATIONS = [
    (["--pp", "both", "--minmax", "off"], "_abl_noscale"),
    (["--pp", "on", "--dp-fixed-scale", "0.01"], "_abl_lambda0.01"),
    (["--pp", "on", "--dp-scale-mode", "sensitivity"], "_abl_sensitivity"),
    (["--pp", "on", "--dp-noise-removal", "own_full"], "_abl_ownfull"),
    (["--pp", "both", "--architecture", "algorithm1_literal"], "_abl_alg1"),
    (["--pp", "both", "--federated-arrays", "conv"], "_abl_conv"),
]
for args, tag in ABLATIONS:
    run(["--dataset", "ton_iot", "--data-path", DATA_FILE, "--results-dir", RESULTS, "--seed", "42",
         "--tag-suffix", tag] + args)
print(f"all runs finished in {(time.time() - t0) / 60:.0f} min")"""

LOAD = """from torre2025_fl_cnn_pp.paper_results import PAPER_OVERALL, PAPER_PER_CLASS, PAPER_FL
R = {os.path.basename(os.path.dirname(f)): json.load(open(f))
     for f in sorted(glob.glob(os.path.join(RESULTS, "ton_iot", "*", "summary.json")))}
main = {m: [R[f"{m}_seed{s}"] for s in SEEDS] for m in ("without_pp", "with_pp")}
def ms(v): return f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"
print("runs loaded:", len(R))"""

RES_MD = """## 7. Results

### 7.1 Overall detection performance (paper Table 10)

The overall row is the mean over the 10 classes of the per-class one-vs-rest metrics, defined the same way as in the
paper."""

RES_OVERALL = """rows = [["Paper (Table 10)", *[f"{v:.2f}" for v in PAPER_OVERALL["ton_iot"]]]]
for m, label in (("without_pp", "Ours, without PP"), ("with_pp", "Ours, with PP (λ = 0.1)")):
    rows.append([label + ", seed 42", *[f"{main[m][0]['overall'][k]:.2f}" for k in "APR"], f"{main[m][0]['overall']['F1']:.2f}"])
    rows.append([label + ", 3 seeds", *[ms([r["overall"][k] for r in main[m]]) for k in ("A", "P", "R", "F1")]])
display(pd.DataFrame(rows, columns=["", "Accuracy", "Precision", "Recall", "F1"]).set_index(""))"""

RES_CLASS_MD = """### 7.2 Per-class results (paper Table 3)

The paper does not say whether Tables 3 and 10 come from the run with or without PP. They match the run without PP."""

RES_CLASS = """classes = [r["class"] for r in main["without_pp"][0]["per_class"]]
tab = {}
for k, c in enumerate(classes):
    p = PAPER_PER_CLASS["ton_iot"][c]
    tab[c] = {"paper A": p[0], "paper P": p[1], "paper R": p[2], "paper F1": p[3],
              "ours A": np.mean([r["per_class"][k]["A"] for r in main["without_pp"]]),
              "ours P": np.mean([r["per_class"][k]["P"] for r in main["without_pp"]]),
              "ours R": np.mean([r["per_class"][k]["R"] for r in main["without_pp"]]),
              "ours F1": np.mean([r["per_class"][k]["F1"] for r in main["without_pp"]]),
              "ours F1, with PP": np.mean([r["per_class"][k]["F1"] for r in main["with_pp"]]),
              "test rows": main["without_pp"][0]["per_class"][k]["support"]}
per_class = pd.DataFrame(tab).T
display(per_class.round(2))
print("'ours' = mean over 3 seeds")

# Figure 1: per-class F1 - paper vs reimplementation without PP vs with PP (three validated palette slots)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
x = np.arange(len(classes)); w = 0.26
fig, ax = plt.subplots(figsize=(11, 4.2))
for i, (col, color, label) in enumerate([("paper F1", ORANGE, "Paper (Table 3)"),
                                         ("ours F1", BLUE, "Ours, without PP"),
                                         ("ours F1, with PP", AQUA, "Ours, with PP (λ = 0.1)")]):
    ax.bar(x + (i - 1) * w, per_class[col].values, w - 0.02, color=color, label=label, zorder=2)
ax.set_xticks(x, classes, color=INK); ax.set_ylabel("F1 (%)", color=MUTED); ax.set_ylim(0, 105)
ax.grid(axis="y", color="#e6e5e1", zorder=0); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
ax.set_title("Figure 1. Per-class F1 on TON-IoT", loc="left", color=INK, pad=28)
plt.tight_layout(); plt.show()"""

RES_FL_MD = """### 7.3 Federated-learning performance (paper Tables 11 and 12)

| Column | Meaning |
|---|---|
| I# | The iteration at which the client converged. "10, not converged" means it never met the 0.0005 tolerance. |
| CCT | Client computation time. |
| TFD | Simulated time to receive the federated data. |
| L-A / F-A | Local / federated accuracy at the client's final iteration. |

Both are shown for seed 42, next to the paper."""

RES_FL = """for m, t in (("without_pp", "Table 11 (without PP)"), ("with_pp", "Table 12 (with PP)")):
    paper = PAPER_FL["ton_iot"]["pp" if m == "with_pp" else "nopp"]
    rows = []
    for c in main[m][0]["clients"]:
        p = paper[c["client"]]
        rows.append([c["client"], f"{c['I#']}" + ("" if c["converged"] else ", not converged"), p[0],
                     f"{c['CCT_mean_s']:.1f} s", p[1], f"{c['TFD_mean_s']:.1f} s", p[2],
                     f"{c['L-A']:.2f}", p[3], f"{c['F-A']:.2f}", p[4]])
    display(Markdown(f"**{t}, seed 42**"))
    display(pd.DataFrame(rows, columns=["client", "I#", "paper I#", "CCT", "paper CCT", "TFD", "paper TFD",
                                        "L-A", "paper L-A", "F-A", "paper F-A"]).set_index("client"))
agg = pd.DataFrame({m: {"final L-A (%)": ms([np.mean([c["L-A"] for c in r["clients"]]) for r in main[m]]),
                        "final F-A (%)": ms([np.mean([c["F-A"] for c in r["clients"]]) for r in main[m]]),
                        "mean CCT (s)": ms([np.mean([c["CCT_mean_s"] for c in r["clients"]]) for r in main[m]]),
                        "mean TFD (s)": ms([np.mean([c["TFD_mean_s"] for c in r["clients"]]) for r in main[m]]),
                        "clients converged": f"{sum(c['converged'] for r in main[m] for c in r['clients'])} / 12"}
                    for m in ("without_pp", "with_pp")})
display(Markdown("**Averaged over clients and the 3 seeds**")); display(agg)

# Figure 2: federated accuracy per iteration (mean over clients, band = min-max over seeds)
fig, ax = plt.subplots(figsize=(8, 3.8))
for m, color, label in (("without_pp", BLUE, "Without PP"), ("with_pp", AQUA, "With PP (λ = 0.1)")):
    curves = []
    for r in main[m]:
        h = pd.DataFrame(r["history"]); curves.append(h.groupby("iteration")["fed_acc"].mean().values * 100)
    curves = np.array(curves); it = np.arange(1, curves.shape[1] + 1)
    ax.fill_between(it, curves.min(0), curves.max(0), color=color, alpha=0.15, linewidth=0)
    ax.plot(it, curves.mean(0), color=color, linewidth=2, marker="o", markersize=5, label=label)
    ax.annotate(f"{label}: {curves.mean(0)[-1]:.1f} %", (it[-1], curves.mean(0)[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)
ax.axhline(98.31, color=ORANGE, linewidth=1.5, linestyle="--")
ax.annotate("dashed line: paper final F-A without PP, 98.31 %", (5, 98.31), xytext=(0, -32), textcoords="offset points",
            color=INK, fontsize=9)
ax.set_xlabel("FL iteration", color=MUTED); ax.set_ylabel("Federated accuracy (%)", color=MUTED)
ax.set_xticks(range(1, 11)); ax.grid(axis="y", color="#e6e5e1"); ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0.7, 12.8); ax.legend(frameon=False, loc="lower right")
ax.set_title("Figure 2. Federated accuracy per FL iteration (mean of 4 clients; band = range over 3 seeds)",
             loc="left", color=INK, fontsize=10)
plt.tight_layout(); plt.show()"""

ABL_MD = """### 7.4 Ablations of the ambiguous choices (seed 42, one change each)"""

ABL = """ablation_rows = [("Default settings (reference)", "without_pp_seed42", "with_pp_seed42"),
                 ("No feature scaling (literal §4.2) [A15]", "without_pp_abl_noscale", "with_pp_abl_noscale"),
                 ("DP scale λ = 0.01 [A4]", None, "with_pp_abl_lambda0.01"),
                 ("DP scale λ = Δ/ε, ε = 1 [A4]", None, "with_pp_abl_sensitivity"),
                 ("Subtract the full own noise [A2]", None, "with_pp_abl_ownfull"),
                 ("Algorithm 1 as printed: no BatchNorm/Dropout [A1]", "without_pp_abl_alg1", "with_pp_abl_alg1"),
                 ("Only the Conv1D layer is exchanged (Fig. 1) [A3]", "without_pp_abl_conv", "with_pp_abl_conv")]
def f1(k): return f"{R[k]['overall']['F1']:.2f}" if k else "-"
def fa(k): return f"{np.mean([c['F-A'] for c in R[k]['clients']]):.2f}" if k else "-"
display(pd.DataFrame([[n, f1(a), f1(b), fa(a), fa(b)] for n, a, b in ablation_rows],
                     columns=["change", "F1 without PP", "F1 with PP", "final F-A without PP", "final F-A with PP"]).set_index("change"))

# Figure 3: macro-F1 with PP for each DP-related choice (one series, sorted)
pp_rows = [(n, R[b]["overall"]["F1"]) for n, a, b in ablation_rows if b]
pp_rows.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.barh([n for n, _ in pp_rows], [v for _, v in pp_rows], color=AQUA, height=0.6, zorder=2)
for i, (_, v) in enumerate(pp_rows):
    inside = v > 80                                    # keep the label clear of the dashed paper line
    ax.annotate(f"{v:.1f}", (v, i), xytext=(-4 if inside else 4, 0), textcoords="offset points", va="center",
                ha="right" if inside else "left", color="white" if inside else INK, fontsize=9)
ax.axvline(PAPER_OVERALL["ton_iot"][3], color=ORANGE, linestyle="--", linewidth=1.5)
ax.set_ylim(-0.6, len(pp_rows) + 0.2)
ax.annotate(f"paper F1 {PAPER_OVERALL['ton_iot'][3]:.2f}", (PAPER_OVERALL["ton_iot"][3], len(pp_rows) - 0.15),
            xytext=(-4, 0), textcoords="offset points", ha="right", va="center", color=INK, fontsize=9)
ax.set_xlim(0, 105); ax.set_xlabel("Overall F1 (%) with PP", color=MUTED)
ax.grid(axis="x", color="#e6e5e1", zorder=0); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Figure 3. F1 with PP for each ablation (seed 42)", loc="left", color=INK, fontsize=10)
plt.tight_layout(); plt.show()"""

DISCUSSION_PLACEHOLDER = """## 8. Discussion

All numbers below are from the outputs above (TON-IoT, full protocol, seeds 42, 1 and 2).

### 8.1 What reproduces
- **Detection without PP.** Seed 42 gives A 99.58 / P 96.42 / R 92.97 / F1 94.37. The paper reports
  99.63 / 96.48 / 93.18 / 94.45, so every metric is within 0.25 points.
  - Over 3 seeds the result is 99.55 ± 0.04 / 96.27 ± 0.44 / 89.70 ± 3.08 / 91.17 ± 3.14.
  - The spread comes almost entirely from **MITM**, which has only 1,043 rows (287 test rows). Its F1 is 79.3 on seed
    42 but 48.6 on average. The paper also names MITM as its weak class (F1 74.09, recall 61.0).
- **Per class (Table 3).** Most classes are within ±5 F1 of the paper.
  - Ours is higher for Reconnaissance (+2.3) and Backdoor (+3.9).
  - Ours is lower for DDoS (−3.9), DoS (−3.4) and XSS (−5.1).
- **FL accuracy without PP (Table 11).** The final local / federated accuracy is 97.50 / 97.72 %. The paper reports
  98.16 / 98.31 %.
- **The dataset is the paper's.** The 461,043 rows and the class counts match Table 1 exactly.

### 8.2 What does not reproduce
- **Accuracy with PP (Table 12).** At λ = 0.1 (the paper's worked example), the federated accuracy falls to
  73.79 ± 2.22 % and F1 to 32.22 ± 5.79. The paper reports F-A 98.20 % and an "almost unnoticeable" change.
  - The cause is the noise that stays in the federated model after each client removes its own share. It is about
    0.03 per weight, which is enough to wipe out several attack classes (Password F1 8.2, XSS 8.7, MITM 1.2).
  - With **λ = 0.01**, F1 recovers to 91.95 and F-A to 96.89 %, close to the paper. The paper's numbers are therefore
    consistent with a much smaller noise scale than its example. The paper reports neither Δ nor ε, so this cannot be
    confirmed.
- **Convergence.** No client ever met the 0.0005 tolerance (0 of 12 runs, with or without PP), so all clients ran 10
  iterations. The paper reports convergence at iteration 9.25 on average. Adam keeps moving the weights by more than
  0.0005 per iteration even at about 98 % accuracy, so the rule as specified does not trigger on this data.
- **Computation-time overhead.** The overhead of PP here is not measurable: CCT is 3.62 vs 3.56 s and TFD is
  19.33 vs 19.41 s. The paper reports +27 % CCT and +12 % TFD on TON-IoT. Absolute times also differ (hardware), so
  timing should not be compared.

### 8.3 What the ablations show
- **Feature scaling is required.** Without it (literal §4.2), the CNN predicts almost only Normal (F1 7.97 without
  PP). The paper's results imply that some scaling was used.
- **BatchNorm makes the model sensitive to the DP noise.**
  - Algorithm 1 as printed (no BatchNorm or Dropout) is weaker without PP (F1 83.27) but better with PP
    (46.83 vs 35.00).
  - Exchanging only the Conv1D layer behaves much like the default (91.71 without PP, 41.05 with PP).
- **Two alternative readings of the DP layer are worse.**
  - Subtracting the full own noise gives F1 31.17.
  - λ = Δ/ε with ε = 1 gives F1 19.57.

### 8.4 Assessment of the privacy claims
- **The "HE" layer is not homomorphic.** The server decrypts every client's model before averaging, so it protects
  nothing against the server.
- **The DP layer gives no formal guarantee.** No ε is reported, and there is no clipping (unbounded sensitivity).
  It also trades accuracy for noise in a way the paper does not report.
- **The DH layer works.** It is a pairwise-masking (secure-aggregation) scheme: the server only learns the sum, and
  the masks cancel exactly (measured residual 3.6e-11 above, float rounding).

### 8.5 Limitations of this reimplementation
- **One dataset.** Only TON-IoT was run, not the paper's other six datasets.
- **Unreported details** were filled in as assumptions; Section 2 lists them, with ablations for the important ones.
- **Not bit-reproducible with PP.** The ECDH keys come from cryptographic randomness, which cannot be seeded, and the
  resulting float64 rounding differences are amplified by training. An earlier execution of the same configuration
  gave F1 35.63 ± 6.59 with PP, against 32.22 ± 5.79 here. The conclusions are the same. Results without PP are
  bit-identical across executions.
- **Duplicates kept.** The 11,071 duplicate rows are kept, as in the paper, which may inflate test scores slightly."""


def build():
    nb = nbf.v4.new_notebook()
    cells = [MD(TITLE), MD(METHOD), MD(ASSUMPTIONS), MD(SRC_MD),
             CODE("import os\nos.makedirs('torre2025_fl_cnn_pp', exist_ok=True)")]
    for m in MODULES:
        with open(os.path.join(PKG, m)) as fh:
            src = fh.read() or "# package marker\n"
        cells.append(CODE(f"%%writefile torre2025_fl_cnn_pp/{m}\n{src}"))
    cells += [MD("## Setup"), CODE(SETUP), MD(DATA_MD), CODE(DATA), MD(CHECKS_MD), CODE(CHECKS),
              MD(EXP_MD), CODE(EXP), CODE(LOAD), MD(RES_MD), CODE(RES_OVERALL), MD(RES_CLASS_MD), CODE(RES_CLASS),
              MD(RES_FL_MD), CODE(RES_FL), MD(ABL_MD), CODE(ABL), MD(DISCUSSION_PLACEHOLDER)]
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    nbf.write(nb, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
