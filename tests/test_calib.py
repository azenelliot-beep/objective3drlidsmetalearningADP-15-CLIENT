"""Block 18d: weak-class confusion check + detection-threshold calibration (evaluation only) and the 'calib' stage."""
import json, os, re
import numpy as np
import pytest
import torch

from conftest import notebook_cells


def block(prefix):
    return next(s for s in notebook_cells() if s.startswith(prefix))


def calib_src(tag="final", test=True):
    """Block 18d with the tag / test flag used by these tests (the notebook default is v3q, no test)."""
    s = re.sub(r'^CALIB_TAG = "[\w,]*"', f'CALIB_TAG = "{tag}"', block("# BLOCK 18d "), flags=re.M)
    return re.sub(r'^CALIB_TEST = \w+', f'CALIB_TEST = {test}', s, flags=re.M)


def test_tau_zero_reproduces_argmax_and_choose_tau_respects_cap(nb):
    rng = np.random.default_rng(0)
    q = rng.normal(size=(500, nb["action_dim"])).astype(np.float32)
    y = rng.integers(0, nb["action_dim"], 500)
    exec(block("# BLOCK 18d ").split("cal_df, weak_df, pcls_df = run_calibration()")[0], nb)   # helpers only
    assert np.array_equal(nb["calibrated_pred"](q, 0.0), q.argmax(1))
    tau = nb["choose_tau"](q, y, 0.10)
    m = nb["metrics_from_pred"](nb["calibrated_pred"](q, tau), y)
    assert m["FAR"] <= 0.10 + 1e-9
    m0 = nb["metrics_from_pred"](q.argmax(1), y)
    assert m["DR"] >= m0["DR"] - 1e-9 or m0["FAR"] > 0.10      # uses the false-alarm budget for detection


def test_calibration_end_to_end_is_evaluation_only(nb, small_cfg, tmp_path, monkeypatch):
    small_cfg.update(CLIP_MULT=0.5, REWARD_FA=0.5, ADP_ACTIONS="pacing", ADP_META_EPISODES=1,
                     PROXY_MIN_PER_CLIENT=5, TAG="final")
    nb["run_all"]([("dp_fixed", 2.0), ("dp_adpmeta", 2.0)], [0])
    d = small_cfg["RESULTS_DIR"]
    before = {f: open(os.path.join(d, f), "rb").read() for f in os.listdir(d)}
    monkeypatch.chdir(tmp_path)
    exec(calib_src(), nb)
    cal = nb["cal_df"]
    assert set(cal["run"]) == {"dp_fixed_eps2__final", "dp_adpmeta_eps2__final"}
    for f, b in before.items():                                  # nothing trained or overwritten
        assert open(os.path.join(d, f), "rb").read() == b
    for _, r in cal.iterrows():
        res = json.load(open(os.path.join(d, f"{r['run']}__seed0.json")))
        assert r["val_argmax_Macro_F1"] == pytest.approx(100 * res["val_at_best"]["Macro_F1"], abs=1e-6)
        assert r["val_argmax_FAR"] == pytest.approx(100 * res["val_at_best"]["FAR"], abs=1e-6)
        c = json.load(open(os.path.join(d, f"{r['run']}__seed0_CALIB.json")))
        assert c["final_epsilon"] == res["final_epsilon"]           # epsilon unchanged (post-processing)
        assert "test_calibrated" in c and "test_argmax" in c
    w = nb["weak_df"]
    assert {"backdoor", "mitm", "ransomware"} <= set(w["class"])
    s = w.dropna(subset=["correct_%"])
    assert np.allclose(s["correct_%"] + s["as_normal_%"] + s["as_other_attack_%"], 100.0)
    assert os.path.exists(tmp_path / "calibration_results.zip")


def test_calib_stage_uses_existing_results_and_trains_only_missing(nb, small_cfg, monkeypatch):
    src = block("# BLOCK 18b ")
    assert '"calib"' in src
    small_cfg.update(TAG="final")
    os.makedirs(small_cfg["RESULTS_DIR"], exist_ok=True)
    json.dump({"cfg": {}}, open(nb["result_path"]("dp_fixed_eps8__final", 0), "w"))   # pretend it was imported
    saved = dict(nb["CFG"])
    calls = []
    monkeypatch.setitem(nb, "run_all", lambda exps, seeds: calls.append((dict(nb["CFG"]), exps, seeds)))
    monkeypatch.setitem(nb, "_DEFAULT_CFG", dict(nb["_DEFAULT_CFG"], RESULTS_DIR=small_cfg["RESULTS_DIR"]))
    try:
        exec(re.sub(r'^STAGE = ".*"$', 'STAGE = "calib"', src, flags=re.M), nb)
    finally:
        nb["CFG"].clear(); nb["CFG"].update(saved)
    assert [(e, s) for _, e, s in calls] == [([("dp_adpmeta", 8.0)], [0])]
    cfg = calls[0][0]
    assert cfg["TAG"] == "final" and cfg["ADP_ACTIONS"] == "pacing" and cfg["ADP_META_EPISODES"] == 10
    assert cfg["CLIP_MULT"] == 0.5 and cfg["REWARD_FA"] == 0.5 and cfg["MU"] == 0.01


def test_block18d_trains_nothing():
    src = block("# BLOCK 18d ")
    assert not re.search(r"\b(run_all|run_dp_pipeline|fit_clients|optimizer|\.step\(|backward)\b", src)


def test_summary_and_test_cells_ignore_non_run_json(nb, small_cfg, tmp_path, monkeypatch):
    """Regression: _CALIB.json / foreign JSON files in results/ crashed Block 18c with KeyError 'val_at_best'."""
    small_cfg.update(TAG="final")
    nb["run_all"]([("dp_fixed", 2.0)], [0])
    d = small_cfg["RESULTS_DIR"]
    json.dump({"run": "dp_fixed_eps2__final", "seed": 0, "tau": 0.1}, open(os.path.join(d, "dp_fixed_eps2__final__seed0_CALIB.json"), "w"))
    json.dump({"hello": 1}, open(os.path.join(d, "other__seed0_meta.json"), "w"))
    vdf = nb["summarize_val"]()
    assert list(vdf["run"]) == ["dp_fixed_eps2__final"]
    monkeypatch.chdir(tmp_path)
    exec(block("# BLOCK 18c "), nb)
    exec(calib_src(), nb)
    assert list(nb["cal_df"]["run"]) == ["dp_fixed_eps2__final"]
    exec(block("# BLOCK 19 ").replace("FINAL_TEST = False", "FINAL_TEST = True").replace('TEST_TAG = "bst10,bstbin"', 'TEST_TAG = "final"'), nb)
    assert list(nb["tdf"]["run"]) == ["dp_fixed_eps2__final"]


def test_default_is_v3_quick_with_frozen_v3_design(nb, monkeypatch):
    src = block("# BLOCK 18b ")
    assert '"v3_quick"' in src
    src = re.sub(r'^STAGE = ".*"$', 'STAGE = "v3_quick"', src, flags=re.M)
    saved = dict(nb["CFG"]); calls = []
    monkeypatch.setitem(nb, "run_all", lambda exps, seeds: calls.append((dict(nb["CFG"]), exps, seeds)))
    try:
        exec(src, nb)
    finally:
        nb["CFG"].clear(); nb["CFG"].update(saved)
    ((cfg, exps, seeds),) = calls
    assert cfg["TAG"] == "v3q" and cfg["ADP_ACTIONS"] == "pacing_clip" and cfg["META_EPS"] == (8.0,)
    assert cfg["CLIP_MULT"] == 0.5 and cfg["REWARD_FA"] == 0.5 and cfg["MU"] == 0.01
    assert cfg["CLIP_ACTIONS"] == (0.5, 1.0, 2.0) and cfg["META_TIME_LIMIT_MIN"] == 180
    assert exps == [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)] and seeds == [0]
    b18d = block("# BLOCK 18d ")
    assert 'CALIB_TAG = "bst10,bstbin"' in b18d and "CALIB_TEST = False" in b18d


def test_v3_meta_time_limit(nb, small_cfg):
    small_cfg.update(CLIP_MULT=0.5, ADP_ACTIONS="pacing_clip", ADP_META_EPISODES=3, META_EPS=(2.0,), TWIN_POOL=1,
                     PROXY_MIN_PER_CLIENT=5, META_TIME_LIMIT_MIN=0.0, TAG="v3t")
    nb["meta_train_controller_v3"](1.0)
    log = json.load(open(nb["controller_v3_path"]().replace(".pt", "_meta_log.json")))
    assert log["episodes_run"] == 1 and log["stopped_by_time_limit"] is True
