"""Executes the BUILT notebook (notebooks/chapter6_v4.ipynb) on synthetic data, so the tests check
exactly the code that runs on Kaggle. Data-loading blocks (4-8b) are replaced by a synthetic ToN-IoT-like
dataset; the pip cell and the experiment-launch cell (18b) are skipped."""
import json, os, pathlib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "chapter6_v4.ipynb"

SKIP_PREFIXES = ("# BLOCK 1 ", "import glob", "# BLOCK 4 ", "# BLOCK 5 ", "# BLOCK 6 ", "# BLOCK 7 ",
                 "# BLOCK 8 ", "# BLOCK 8b ", "# BLOCK 18b ")
CLASS_NAMES = ['backdoor', 'ddos', 'dos', 'injection', 'mitm', 'normal', 'password', 'ransomware', 'scanning', 'xss']
SIZES = [300, 500, 400, 600, 150, 3000, 500, 80, 300, 400]      # imbalanced like ToN-IoT, scaled down
N_FEAT = 29

SYNTHETIC = r'''
import pandas as pd
_rng = np.random.default_rng(123)
class_names = __CLASS_NAMES__
_sizes = __SIZES__
_means = _rng.normal(0, 1.5, size=(len(class_names), __N_FEAT__))
def _make(scale):
    xs, ys = [], []
    for c, n in enumerate(_sizes):
        n = max(3, int(n * scale))
        xs.append(_means[c] + _rng.normal(0, 1.0, size=(n, __N_FEAT__))); ys.append(np.full(n, c))
    X_, y_ = np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)
    p = _rng.permutation(len(y_)); return X_[p], y_[p]
X_train, y_train = _make(1.0)
X_val,   y_val   = _make(0.15)
X_test,  y_test  = _make(0.3)
selected_features = [f"f{i}" for i in range(__N_FEAT__)]
X = pd.DataFrame(X_train, columns=selected_features)
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val).long())
test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test).long())
'''.replace("__CLASS_NAMES__", repr(CLASS_NAMES)).replace("__SIZES__", repr(SIZES)).replace("__N_FEAT__", str(N_FEAT))


def notebook_cells():
    nb = json.loads(NB.read_text())
    return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="session")
def nb(tmp_path_factory):
    ns = {"__name__": "__nb__"}
    synthetic_done = False
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("kaggle_working"))
    try:
        for src in notebook_cells():
            if src.startswith(SKIP_PREFIXES):
                if src.startswith("# BLOCK 4 ") and not synthetic_done:
                    exec(SYNTHETIC, ns); synthetic_done = True
                continue
            exec(compile(src, "<cell>", "exec"), ns)
    finally:
        os.chdir(cwd)
    assert synthetic_done
    return ns


@pytest.fixture
def small_cfg(nb, tmp_path):
    """Tiny, fast settings; results go to a fresh temp dir. CFG restored afterwards."""
    saved = dict(nb["CFG"])
    nb["CFG"].update(ROUNDS=3, LOCAL_STEPS=4, BATCH=64, CH5_ROUNDS=2, CH5_LOCAL_EPOCHS=1,
                     RESULTS_DIR=str(tmp_path / "results"))
    yield nb["CFG"]
    nb["CFG"].clear(); nb["CFG"].update(saved)
