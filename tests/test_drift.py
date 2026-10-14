"""Drift scenario (Block 17e): attack waves via client.active masks, drift-aware ADP-Meta v5, fixed baseline."""
import json, os, re
import numpy as np
import pytest
import torch

from conftest import notebook_cells


def clients(nb, n_clients=3, n_per=40, clip=0.5, seed=0):
    torch.manual_seed(seed)
    cs = []
    for k in range(n_clients):
        ag = nb["SmallDDQN"](nb["input_dim"], nb["action_dim"], nb["device"], nb["class_weights_dp"],
                             gamma=0.0, eps_start=0.0, eps_end=0.0)
        X, y = nb["X_dp"][k * n_per:(k + 1) * n_per + k].copy(), nb["y_dp"][k * n_per:(k + 1) * n_per + k]
        cs.append(nb["DPClient"](ag, X, y, nb["device"], batch_size=10_000, clip_norm=clip))
    return cs


def params_after(nb, cs, fast=None, steps=3, sigma=0.0):
    gp = cs[0].ddqn.get_fl_parameters()
    ps, _, _ = nb["fit_clients"](cs, list(range(len(cs))), gp, {i: sigma for i in range(len(cs))}, steps, 0.0, fast)
    return ps


def same(a, b):
    return all(torch.allclose(x, y, atol=1e-5, rtol=1e-4) for pa, pb in zip(a, b) for x, y in zip(pa, pb))


def test_all_active_mask_equals_no_mask(nb):
    a, b = clients(nb), clients(nb)
    for c in b:
        c.active = torch.ones(c.n, dtype=torch.bool)
    assert same(params_after(nb, a), params_after(nb, b))


def test_inactive_records_never_influence_training(nb):
    a, b = clients(nb), clients(nb)
    for ca, cb in zip(a, b):
        mask = torch.arange(ca.n) % 2 == 0
        ca.active, cb.active = mask, mask.clone()
        cb.X[~mask] = cb.X[~mask] + 100.0                 # change ONLY inactive records
    assert same(params_after(nb, a), params_after(nb, b))


def test_fast_path_equals_loop_with_masks(nb):
    slow, fastc = clients(nb), clients(nb)
    for c1, c2 in zip(slow, fastc):
        m = torch.arange(c1.n) % 3 != 0
        c1.active, c2.active = m, m.clone()
    fast = nb["FastFederation"](fastc)
    assert same(params_after(nb, slow), params_after(nb, fastc, fast))


def test_masks_do_not_change_accounting(nb):
    def eps(mask):
        (c,) = clients(nb, n_clients=1, clip=1.0)
        c.active = mask if mask is None else mask(c.n)
        for _ in range(4):
            c._dp_step(1.1)
        return c.epsilon_spent()
    assert eps(None) == pytest.approx(eps(lambda n: torch.arange(n) % 2 == 0))


def test_waves_parse_and_set_arrivals(nb):
    arr = nb["parse_waves"](nb["CFG"]["DRIFT_WAVES"])
    assert sorted(arr) == list(range(len(nb["class_names"])))
    with pytest.raises(AssertionError, match="without an arrival"):
        nb["parse_waves"]("1:normal")
    cs = clients(nb)
    n1 = nb["set_arrivals"](cs, 1, arr)
    for c in cs:
        allowed = {i for i, r in arr.items() if r <= 1}
        assert set(c.y[c.active].tolist()) <= allowed
    assert n1 == len([1 for r in arr.values() if r <= 1])
    assert nb["set_arrivals"](cs, 99, arr) == len(nb["class_names"]) and all(bool(c.active.all()) for c in cs)
    nb["clear_arrivals"](cs)
    assert all(c.active is None for c in cs)


def test_drift_signals(nb):
    names = nb["class_names"]
    now = {n: (0.9 if n in ("normal", "dos") else 0.0) for n in names}
    prev = {n: 0.0 for n in names}
    unl, burst, spike = nb["drift_signals"](now, prev, 2.0, 1.0)
    assert unl == pytest.approx(8 / 9) and burst == pytest.approx(0.9) and spike == pytest.approx(np.log(2.0))


def test_drift_runs_fixed_and_v5(nb, small_cfg, monkeypatch):
    small_cfg.update(CLIP_MULT=0.5, REWARD_FA=0.5, DRIFT=True, ADP_ACTIONS="drift", ADP_META_EPISODES=2,
                     META_EPS=(2.0,), TWIN_POOL=1, PROXY_MIN_PER_CLIENT=5, ROUNDS=4, TAG="drq",
                     DRIFT_WAVES="1:normal,scanning;2:dos,ddos;3:injection,xss,password;4:backdoor,ransomware,mitm")
    seen = []
    real_init = nb["DPClient"].__init__
    def spy(self, ddqn, X_local, y_local, *a, **k):
        seen.append((k.get("model_name", ""), np.asarray(X_local)))
        real_init(self, ddqn, X_local, y_local, *a, **k)
    monkeypatch.setattr(nb["DPClient"], "__init__", spy)
    fx = nb["run_dp_pipeline"]("dp_fixed", 0, 2.0, C=1.0)
    assert fx["scenario"] == "drift" and fx["adp_version"] == "fixed"
    assert [h["classes_arrived"] for h in fx["history"]] == [2, 4, 7, 10]
    sig = [h["median_sigma"] for h in fx["history"]]
    assert max(sig) == pytest.approx(min(sig), rel=1e-3)                 # fixed schedule = constant noise
    assert all(h["epsilon"] <= 2.0 + 1e-9 for h in fx["history"]) and fx["final_epsilon"] >= 0.95 * 2.0
    ad = nb["run_dp_pipeline"]("dp_adpmeta", 0, 2.0, C=1.0)
    assert ad["adp_version"] == "v5-drift" and ad["run"] == "dp_adpmeta_eps2__drq"
    assert all(h["epsilon"] <= 2.0 + 1e-9 for h in ad["history"])
    assert all(k in ad["history"][0] for k in ("drift_unlearned", "drift_burst", "drift_spike"))
    ia, ib = nb["val_halves"]()
    rows = lambda X: {r.tobytes() for r in X}
    A, B = rows(nb["X_val"][ia]), rows(nb["X_val"][ib])
    proxies = [X for n, X in seen if n.startswith("Proxy")]
    reals = [X for n, X in seen if n.startswith("Client")]
    assert len(proxies) == (2 + 1) * nb["NUM_CLIENTS"] and all(rows(X) <= A for X in proxies)
    assert all(not (rows(X) & (A | B)) for X in reals)
    log = json.load(open(ad["controller_file"].replace(".pt", "_meta_log.json")))
    assert log["episodes_run"] == 2
    ctrl = nb["ADPMetaV5"]((0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
    assert ctrl.act([0.0] * 12, greedy=True) == (1.0, 1.0)              # starts at the fixed schedule


def test_drift_stage_default(nb, monkeypatch):
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 18b "))
    assert 'STAGE = "drift_quick"' in src
    saved = dict(nb["CFG"]); calls = []
    monkeypatch.setitem(nb, "run_all", lambda exps, seeds: calls.append((dict(nb["CFG"]), exps, seeds)))
    try:
        exec(src, nb)
    finally:
        nb["CFG"].clear(); nb["CFG"].update(saved)
    ((cfg, exps, seeds),) = calls
    assert cfg["TAG"] == "drq" and cfg["DRIFT"] is True and cfg["ADP_ACTIONS"] == "drift"
    assert cfg["CLIP_MULT"] == 0.5 and cfg["REWARD_FA"] == 0.5 and cfg["MU"] == 0.01 and cfg["META_EPS"] == (8.0,)
    assert exps == [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)] and seeds == [0]


def test_block17e_never_touches_test_set():
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 17e "))
    assert not re.search(r"\b(test_dataset|X_test|y_test)\b", src)
