import numpy as np
import pytest

from torre2025_fl_cnn_pp.config import Config
from torre2025_fl_cnn_pp.data import prepare
from torre2025_fl_cnn_pp.fl import run_federated
from torre2025_fl_cnn_pp.metrics import overall_from_per_class, per_class_metrics
from torre2025_fl_cnn_pp.model import build_cnn
from torre2025_fl_cnn_pp.paper_results import PAPER_OVERALL, PAPER_PER_CLASS

SMALL = Config().with_overrides(epochs=1, batch_size=64, max_iterations=2, synthetic_rows=4000)


@pytest.fixture(scope="module")
def fd():
    return prepare("synthetic", None, SMALL)


@pytest.mark.parametrize("use_pp", [False, True])
def test_all_mechanisms_execute(fd, use_pp):
    res = run_federated(fd, SMALL, use_pp)
    n_it = SMALL.max_iterations
    assert len(res["history"]) == SMALL.n_clients * n_it
    for c in res["clients"]:
        expect = n_it if use_pp else 0
        assert c["mechanisms"] == {"dp": expect, "dk": expect, "he_encrypt": expect, "he_decrypt": expect}
    s = res["server_mechanisms"]
    assert s["fedavg"] == n_it
    assert s["he_decrypt"] == s["he_encrypt"] == (SMALL.n_clients * n_it if use_pp else 0)
    assert max(res["offset_cancellation_max_err"]) < 1e-6      # DH offsets cancel in the average


def test_dropout_after_convergence(fd):
    cfg = SMALL.with_overrides(tolerance=1e9)                   # everything "converges" at iteration 1
    res = run_federated(fd, cfg, use_pp=True)
    assert all(c["I#"] == 1 and c["converged"] for c in res["clients"])
    assert {h["iteration"] for h in res["history"]} == {1}


def test_without_noise_pp_is_lossless(fd):
    """With lambda = 0 the DK + HE layers must reproduce plain FedAvg exactly (encryption is invertible)."""
    cfg = SMALL.with_overrides(dp_fixed_scale=0.0, max_iterations=1)
    a, b = run_federated(fd, cfg, False), run_federated(fd, cfg, True)
    for ha, hb in zip(a["history"], b["history"]):
        assert ha["local_acc"] == hb["local_acc"]
        assert ha["fed_acc"] == pytest.approx(hb["fed_acc"])
        assert ha["max_abs_diff_weights"] == pytest.approx(hb["max_abs_diff_weights"], abs=1e-5)


def test_architectures():
    cfg = Config()
    names = [l.__class__.__name__ for l in build_cnn(20, 4, cfg).layers]
    assert names == ["InputLayer", "Conv1D", "MaxPooling1D", "Flatten", "Dense", "Dropout",
                     "BatchNormalization", "Dense"]
    lit = [l.__class__.__name__ for l in build_cnn(20, 4, cfg.with_overrides(architecture="algorithm1_literal")).layers]
    assert lit == ["InputLayer", "Conv1D", "MaxPooling1D", "Flatten", "Dense", "Dense"]


def test_table10_is_macro_average_of_tables_3_to_9():
    for ds, rows in PAPER_PER_CLASS.items():
        recs = [{"class": k, "support": 1, "A": v[0], "P": v[1], "R": v[2], "F1": v[3]} for k, v in rows.items()]
        o = overall_from_per_class(recs)
        assert np.allclose([o[m] for m in ("A", "P", "R", "F1")], PAPER_OVERALL[ds], atol=0.015), ds


def test_per_class_metrics():
    rows = per_class_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
    assert rows[0]["P"] == 100 and rows[0]["R"] == 50 and rows[1]["A"] == 75
    assert rows[1]["F1"] == pytest.approx(100 * 2 * (2 / 3) * 1 / (2 / 3 + 1))
