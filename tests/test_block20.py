"""Block 20: binary IDS metrics, per-class AUC, validation-only calibration of the Q-value decision."""
import numpy as np
import pytest

from conftest import notebook_cells


def test_binary_metrics_formula(nb):
    n = nb["NORMAL_IDX"]; a = (n + 1) % 10; b = (n + 2) % 10
    y = np.array([n, n, n, n, a, a, b, b])
    p = np.array([n, n, n, a, a, b, n, b])      # 1 false alarm, 1 missed attack, 1 attack->other attack
    m = nb["metrics_from_preds"](y, p)
    assert m["FAR"] == pytest.approx(1 / 4) and m["DR"] == pytest.approx(3 / 4)
    assert m["BinAcc"] == pytest.approx(6 / 8) and m["Acc"] == pytest.approx(5 / 8)
    assert m["BinPrec"] == pytest.approx(3 / 4)

def test_calibration_never_hurts_validation_objective(nb):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 2000)
    Q = rng.normal(0, 1, (2000, 10)); Q[np.arange(2000), y] += 0.8; Q[:, 3] -= 1.5   # class 3 under-predicted
    before = nb["calib_objective"](nb["metrics_from_preds"](y, Q.argmax(1)), 0.10)
    b = nb["fit_offsets"](Q, y, 0.10)
    after = nb["calib_objective"](nb["metrics_from_preds"](y, (Q + b).argmax(1)), 0.10)
    assert after >= before and b[3] > 0

def test_block20_runs_on_saved_models(nb, small_cfg, tmp_path, monkeypatch):
    small_cfg.update(CLIP_MULT=0.5, REWARD_FA=0.5, ADP_ACTIONS="pacing", TAG="final")
    nb["run_all"]([("dp_fixed", 2.0), ("dp_adpmeta", 2.0)], [0, 1])
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 20 "))
    exec(src, nb)
    cdf = nb["cdf"]
    assert set(cdf["split"]) == {"val", "test"} and set(cdf["mode"]) == {"raw", "calibrated"}
    assert len(cdf) == 4 * 2 * 2
    v = cdf[cdf["split"] == "val"].pivot_table(index=["run", "seed"], columns="mode", values="MacroF1")
    assert (v["calibrated"] >= v["raw"] - 10 * 100).all()
    assert set(nb["adf"]["run"]) == {"dp_fixed_eps2__final", "dp_adpmeta_eps2__final"}
