"""Build kaur2024/notebooks/kaur2024_reimplementation.ipynb: the full source code (as %%writefile
cells, identical to kaur2024/src and kaur2024/configs) plus the saved results, with outputs
pre-rendered by executing the results cells here (no Jupyter needed to build it).

  python kaur2024/tools/build_notebook.py
"""
import base64
import contextlib
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
K = REPO / "kaur2024"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(src, outputs=None):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": outputs or [], "source": src.strip("\n")})


# ---------------------------------------------------------------- executor for results cells
NS = {}
_outputs = []


def _display(obj):
    if hasattr(obj, "to_html"):
        _outputs.append({"output_type": "display_data", "metadata": {},
                         "data": {"text/html": obj.to_html(), "text/plain": obj.to_string()}})
    elif isinstance(obj, _Img):
        _outputs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": obj.b64, "text/plain": "<image>"}})
    else:
        _outputs.append({"output_type": "display_data", "metadata": {}, "data": {"text/plain": str(obj)}})


class _Img:
    def __init__(self, filename):
        self.b64 = base64.b64encode(Path(filename).read_bytes()).decode()


def _show(*a, **k):
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close("all")
    _outputs.append({"output_type": "display_data", "metadata": {},
                     "data": {"image/png": base64.b64encode(buf.getvalue()).decode(), "text/plain": "<Figure>"}})


def run_code(src):
    """Execute a results cell here and store its outputs in the notebook."""
    _outputs.clear()
    out = io.StringIO()
    plt.show = _show
    NS.update(display=_display, Image=_Img)
    with contextlib.redirect_stdout(out):
        exec(compile(src.replace("from IPython.display import display, Image\n", ""), "<cell>", "exec"), NS)
    res = []
    if out.getvalue():
        res.append({"output_type": "stream", "name": "stdout", "text": out.getvalue()})
    res += list(_outputs)
    code(src, res)


# ======================================================================= notebook content
md("""
# Reimplementation of Kaur (2024): GRU-assisted FL-DRL intrusion detection on TON_IoT

**Paper:** A. Kaur, "Intrusion Detection Approach for Industrial Internet of Things Traffic using Deep
Recurrent Reinforcement Learning Assisted Federated Learning", *IEEE Trans. Artificial Intelligence*,
2024, DOI 10.1109/TAI.2024.3443787.

**What is reimplemented (TON_IoT only):** a GRU classifier trained by 15 federated clients and
aggregated weighted by data size (Eqs. 18, 25). A DQN selects which clients take part each round and
their transmit power (Eqs. 5-8, 12-13). The energy model is Eqs. 1-4. The FedAvg baseline uses all
15 clients every round. Both IID and non-IID data are run.

**Tags** used in code comments and configs:
- **[PAPER]**: stated in the paper (with a citation).
- **[GAP-FILL]**: the paper omits it; the value was approved by the thesis author (see `kaur2024/SPEC.md`).
- **[EXTRA]**: added for thesis rigour, not part of the paper.

**Departures from the paper:**
1. **Local optimiser: plain SGD with η = 0.2** (Eqs. 17/24) instead of Adam (Table I(B)). Adam at
   0.2 collapsed to a single class (SPEC §7).
2. **Each traffic record is a length-1 sequence**, so the GRU uses no temporal context (G3).
3. **Split:** the test set matches Table II exactly (89,518 rows). Training has 371,485 rows
   instead of 395,467, because the file does not contain enough rows (G1).
4. **Protocol:** IID/non-IID with 15 clients is the paper's Edge-IIoT protocol (Table VIII),
   applied here to TON_IoT.
5. **Unstated details** were filled in: DQN top-8 of 15 selection, ε-greedy exploration, network
   sizes, energy constants and the Dirichlet(0.5) non-IID split (G1-G20).

**Notebook layout:**
- Part 1: the full source code (`%%writefile` cells; identical to `kaur2024/src`).
- Part 2: how to run it (switched off by default; a full run takes hours on CPU).
- Part 3: the saved results, with outputs already rendered.
""")

md("## Setup\nRun from anywhere inside the repository; the cell moves to the repository root.")
code("""
import os, json, sys
from pathlib import Path
p = Path.cwd()
while not (p / "kaur2024").is_dir() and p != p.parent:
    p = p.parent
os.chdir(p)
sys.path.insert(0, str(p / "kaur2024" / "src"))
print("repository root:", p)
""")

md("# Part 1: Source code\nEach cell writes one file of the implementation (the same files as in the repository).")
for f, title in [("configs/base.yaml", "Configuration: every setting with its tag"),
                 ("configs/e1_iid_dqn.yaml", "Experiment E1: IID, DQN (proposed model)"),
                 ("configs/e2_noniid_dqn.yaml", "Experiment E2: non-IID, DQN (proposed model)"),
                 ("configs/fedavg_iid.yaml", "Baseline: FedAvg, IID"),
                 ("configs/fedavg_noniid.yaml", "Baseline: FedAvg, non-IID"),
                 ("src/phase1_data.py", "Phase 1: data inspection, fixed split and client partitions (seed 0)"),
                 ("src/data.py", "Preprocessing (fit on train only): drop columns, one-hot, min-max (Eq. 19)"),
                 ("src/gru_model.py", "GRU classifier (2 x 128, dropout 0.3, 1 FC) and parallel client training"),
                 ("src/channel.py", "Channel and energy model (Eqs. 1-4)"),
                 ("src/dqn_selector.py", "DQN client selection (Eqs. 6-8, 12-13), random and FedAvg selectors"),
                 ("src/evaluate.py", "Model-independent evaluation (accuracy, P/R/F1, AUC)"),
                 ("src/fl.py", "Federated training loop (Algorithm 1)"),
                 ("src/runner_utils.py", "Run helpers"),
                 ("src/run_core.py", "Core experiments runner and tables")]:
    md(f"### `kaur2024/{f}`: {title}")
    code(f"%%writefile kaur2024/{f}\n" + (K / f).read_text())

md("""
# Part 2: How to run
Switched off by default. On a 4-core CPU each run takes about 4-8 hours. Set `RUN = True` to
execute. The CSV must be at `kaur2024/data/raw/TON_IoT_Train_Test_Network.csv` (sha256
`65d5465d...`).
""")
code("""
RUN = False
CSV = "kaur2024/data/raw/TON_IoT_Train_Test_Network.csv"
if RUN:
    # 1) fixed split + IID / non-IID partitions (seed 0), identical every time
    !python kaur2024/src/phase1_data.py --csv {CSV}
    # 2) the four core runs (E1, E2, FedAvg IID, FedAvg non-IID), seed 0, 100 rounds
    !python kaur2024/src/run_core.py --csv {CSV} --parallel 4
    # single run example:
    # !python kaur2024/src/fl.py --config kaur2024/configs/e2_noniid_dqn.yaml --seed 0 --csv {CSV}
""")

# ------------------------------------------------------------------------------ results
md("# Part 3: Results (saved runs, training seed 0)\nAll numbers are read from `kaur2024/results/core/*/seed0/`.")
os.chdir(REPO)
run_code("""
import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
R = "kaur2024/results/core"
RUNS = {"E1 IID, DQN (proposed)": "e1_iid_dqn", "E2 non-IID, DQN (proposed)": "e2_noniid_dqn",
        "FedAvg IID (baseline)": "fedavg_iid", "FedAvg non-IID (baseline)": "fedavg_noniid"}
CLASSES = ["Normal","Scanning","DoS","DDoS","Ransomware","Backdoor","Injection","XSS","Password","MITM"]

def M(run, rnd):
    f = f"{R}/{run}/seed0/metrics_round{rnd}.json"
    try:
        return json.load(open(f))
    except FileNotFoundError:
        return None

def summary_row(m):
    g, u = m["global"], m["global_unseen_subset"]
    return {"Accuracy": 100*g["accuracy"], "Precision (macro)": 100*g["macro"]["precision"],
            "Recall (macro)": 100*g["macro"]["recall"], "F1 (macro)": 100*g["macro"]["f1"],
            "AUC (macro OvR)": 100*g["auc"]["macro_ovr"], "Energy (J, cumulative)": m["energy_cum"],
            "Unseen-rows accuracy": 100*u["accuracy"], "Unseen-rows F1 (macro)": 100*u["macro"]["f1"]}
print("loaded")
""")

md("## Main result: reimplemented Kaur (2024) model (GRU + FL + DQN) at round 50 and round 100")
run_code("""
rows = {}
for name in ["E1 IID, DQN (proposed)", "E2 non-IID, DQN (proposed)"]:
    for rnd in (50, 100):
        rows[(name, rnd)] = summary_row(M(RUNS[name], rnd))
t = pd.DataFrame(rows).T.round(2)
t.index.names = ["run", "round"]
display(t)
""")

md("## Table VIII format: accuracy (%) of best client, worst client and global model\n"
   "Paper Table VIII is on Edge-IIoT, so there is no paper TON_IoT reference for this table. "
   "FedAvg stopped at rounds 55 and 51 (container restart), so it has no round 100.")
run_code("""
rows = {}
for name, run in RUNS.items():
    r = {}
    for rnd in (1, 50, 100):
        m = M(run, rnd)
        for k, key in (("best", "best_client_acc"), ("worst", "worst_client_acc")):
            r[f"R{rnd} {k}"] = None if m is None else round(100*m[key], 2)
        r[f"R{rnd} global"] = None if m is None else round(100*m["global"]["accuracy"], 2)
    rows[name] = r
display(pd.DataFrame(rows).T)
""")

md("## All four runs at round 50 (equal-budget comparison point)")
run_code("""
display(pd.DataFrame({name: summary_row(M(run, 50)) for name, run in RUNS.items()}).T.round(2))
""")

md("## Table V format: per-class precision / recall / F1 (%) at round 50")
run_code("""
for name, run in RUNS.items():
    g = M(run, 50)["global"]
    t = pd.DataFrame({c: {k: 100*g["per_class"][c][k] for k in ("precision","recall","f1")} for c in CLASSES})
    t["macro"] = [100*g["macro"][k] for k in ("precision","recall","f1")]
    print(name, "| accuracy", round(100*g["accuracy"], 2), "| AUC", round(100*g["auc"]["macro_ovr"], 2))
    display(t.round(2))
""")

md("## Paper (Table V, TON_IoT) vs reimplementation (round 50)\n"
   "The paper's own per-class rows imply 96.47% accuracy and a 95.02 macro F1, not its reported 99.95 / 99.99 "
   "(X4: `kaur2024/results/extra/x4_table5_consistency.md`).")
run_code("""
paper = {"Accuracy": 99.95, "Precision (macro)": 98.63, "Recall (macro)": 99.98, "F1 (macro)": 99.99}
implied = {"Accuracy": 96.47, "Precision (macro)": 99.19, "Recall (macro)": 93.38, "F1 (macro)": 95.02}
ours_iid = {k: v for k, v in summary_row(M("e1_iid_dqn", 50)).items() if k in paper}
ours_non = {k: v for k, v in summary_row(M("e2_noniid_dqn", 50)).items() if k in paper}
display(pd.DataFrame({"Paper (reported)": paper, "Paper (implied by its per-class rows)": implied,
                      "Ours IID (E1)": ours_iid, "Ours non-IID (E2)": ours_non}).T.round(2))
""")

md("## Accuracy and energy per round")
run_code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for name, run in RUNS.items():
    L = [json.loads(l) for l in open(f"{R}/{run}/seed0/log.jsonl") if '"global_test_acc"' in l]
    ax[0].plot([x["round"] for x in L], [100*x["global_test_acc"] for x in L], label=name)
    ax[1].plot([x["round"] for x in L], [x["energy_cum"] for x in L], label=name)
ax[0].set(xlabel="round", ylabel="global test accuracy (%)", title="Accuracy per round")
ax[1].set(xlabel="round", ylabel="cumulative energy C_training (J)", title="Energy (Eq. 4), cumulative")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
plt.show()
""")

md("## Round-to-round variation\n"
   "Non-IID accuracy swings by several points between rounds (it depends on which clients are selected). "
   "A single-round value can therefore land on a high or low point, so means over 10 rounds are reported too.")
run_code("""
rows = {}
for name, run in RUNS.items():
    a = np.array([json.loads(l)["global_test_acc"] for l in open(f"{R}/{run}/seed0/log.jsonl") if '"global_test_acc"' in l]) * 100
    r = {"round 50": a[49]}
    r.update({"rounds 41-50 mean": a[40:50].mean(), "rounds 41-50 min": a[40:50].min(), "rounds 41-50 max": a[40:50].max()})
    if len(a) >= 100:
        r.update({"round 100": a[99], "rounds 91-100 mean": a[90:100].mean(),
                  "rounds 91-100 min": a[90:100].min(), "rounds 91-100 max": a[90:100].max()})
    rows[name] = r
display(pd.DataFrame(rows).T.round(2))
""")

md("## Confusion matrix: E2 non-IID, DQN, round 50")
run_code("""
cm = np.array(M("e2_noniid_dqn", 50)["global"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(10), CLASSES, rotation=45, ha="right"); ax.set_yticks(range(10), CLASSES)
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] / cm[i].sum() > .5 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.colorbar(im, label="row-normalised")
plt.show()
""")

md("## ROC curve (micro-average, one-vs-rest): proposed model, round 50 (paper Fig. 7 style)")
run_code("""
display(Image(f"{R}/e1_iid_dqn/seed0/roc_round50.png"))
display(Image(f"{R}/e2_noniid_dqn/seed0/roc_round50.png"))
""")

md("## Data: class counts vs Table II and client partitions (fixed seed 0)")
run_code("""
print(open("kaur2024/results/phase1_data_report.md").read().split("## Client partition")[0].split("## Class counts")[1])
display(Image("kaur2024/results/partitions/class_distribution_iid.png"))
display(Image("kaur2024/results/partitions/class_distribution_noniid_dir0.5.png"))
""")

md("""
## Why the accuracy is far below the paper's 99.95% (diagnostics)
- **Min-max scaling (Eq. 19) on heavy-tailed columns.** Byte, packet and duration values end up near
  0 for 99% of rows, so the model effectively sees only the one-hot flags
  (`kaur2024/results/extra/diag_duplicate_labels.txt`, SPEC §8).
- **The model underfits.** It is not limited by the data: for 3,661 of the 3,970 Ransomware test rows,
  the identical feature vector occurs in train with the same label.
- **The paper's Adam with η = 0.2 collapses to one class.** SGD with η = 0.2 is used instead
  (`kaur2024/results/extra/diag_lr_collapse.txt`, SPEC §7).
- **60% of test rows duplicate a training row's features** (X3). The "unseen rows" columns score only
  the 35,438 test rows that never occur in train.
""")
run_code("""
print(open("kaur2024/results/extra/diag_lr_collapse.txt").read())
print(open("kaur2024/results/extra/diag_duplicate_labels.txt").read())
""")

md("""
## How to score another model on exactly the same test set
```python
from evaluate import evaluate
m = evaluate(y_true, y_pred, y_score)                       # full test set (89,518 rows)
mask = np.load("kaur2024/results/partitions/unseen_test_mask.npy")
m_unseen = evaluate(y_true, y_pred, y_score, mask=mask)     # 35,438 unseen rows
```
Use the split in `kaur2024/results/partitions/` (`split_train_idx.npy`, `split_test_idx.npy`, client
partitions) so that both models see identical data.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:03d}"
    if c["cell_type"] == "code":
        c["source"] = c["source"]
out = K / "notebooks" / "kaur2024_reimplementation.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB, {len(cells)} cells)")
