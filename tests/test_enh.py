"""Enhanced stage (Block 17g): weights fix, server momentum / EMA (post-processing) and ADP-Meta v6 (lr x clip)."""
import json, os, re
import numpy as np
import pytest
import torch

from conftest import notebook_cells


def enh(small_cfg, **kw):
    small_cfg.update(CLIP_MULT=0.5, REWARD_FA=0.5, WEIGHTS_FROM="val", ENH=True, SERVER_MOMENTUM=0.5,
                     SERVER_EMA=0.8, ADP_ACTIONS="lr_clip", META_EPS=(2.0,), ADP_META_EPISODES=2, TWIN_POOL=1,
                     PROXY_MIN_PER_CLIENT=5, TAG="enht")
    small_cfg.update(kw)
    return small_cfg


def test_server_opt_math(nb):
    S = nb["ServerOpt"]
    w = [torch.zeros(3), torch.full((2,), 2.0)]                          # every client move below is +1
    a1 = [torch.ones(3), torch.full((2,), 3.0)]
    plain = S(0.0, 0.0).step(w, a1)
    assert all(torch.equal(x, y) for x, y in zip(plain, a1))            # off = plain FedAvg/FedProx average
    s = S(0.5, 0.0)
    n1 = s.step(w, a1)                                                   # v = d -> same as plain on round 1
    assert all(torch.allclose(x, y) for x, y in zip(n1, a1))
    a2 = [x + 1 for x in n1]                                             # clients move +1 again
    n2 = s.step(n1, a2)                                                  # v = 0.5 * 1 + 1 = 1.5
    assert all(torch.allclose(x, y + 1.5) for x, y in zip(n2, n1))
    e = S(0.0, 0.8)
    e.step(w, a1); e.step(a1, [x + 1 for x in a1])
    assert all(torch.allclose(t, x + 0.2) for t, x in zip(e.evaluated(None), a1))   # 0.8 a1 + 0.2 (a1 + 1)


def test_fixed_policy_adp_equals_dp_fixed_exactly(nb, small_cfg, monkeypatch):
    enh(small_cfg)
    nb["run_all"]([("dp_fixed", 2.0)], [0])
    fixed = json.load(open(nb["result_path"](nb["run_name"]("dp_fixed", 2.0), 0)))

    class Stub:
        def act(self, s, greedy=False):
            return 1.0, 1.0
    monkeypatch.setitem(nb, "meta_train_controller_v6", lambda C: Stub())
    nb["run_all"]([("dp_adpmeta", 2.0)], [0])
    adp = json.load(open(nb["result_path"](nb["run_name"]("dp_adpmeta", 2.0), 0)))
    for hf, ha in zip(fixed["history"], adp["history"]):
        assert hf["Macro_F1"] == ha["Macro_F1"] and hf["FAR"] == ha["FAR"] and hf["epsilon"] == ha["epsilon"]
    assert fixed["sigma_per_client"] == adp["sigma_per_client"]


def test_budget_spent_and_respected(nb, small_cfg):
    enh(small_cfg, TAG="enhb")
    nb["run_all"]([("dp_fixed", 2.0)], [0])
    r = json.load(open(nb["result_path"](nb["run_name"]("dp_fixed", 2.0), 0)))
    eps = [h["epsilon"] for h in r["history"]]
    assert len(eps) == small_cfg["ROUNDS"] and all(e <= 2.0 + 1e-9 for e in eps)
    assert eps[-1] >= 0.95 * 2.0                                         # the fixed schedule spends the budget
    assert r["cfg"]["WEIGHTS_FROM"] == "val" and r["server_momentum"] == 0.5 and r["server_ema"] == 0.8


def test_lr_and_clip_actions_change_training_not_epsilon(nb, small_cfg):
    enh(small_cfg, TAG="enhl")
    C = nb["dp_clip_norm"]()
    def run(policy):
        nb["set_seed"](0)
        parts = nb["dirichlet_split"](nb["y_dp"], nb["NUM_CLIENTS"], nb["ALPHA"], np.random.default_rng(0),
                                      nb["MIN_PER_CLIENT"])
        w = nb["reward_weights"]()
        mk = lambda: nb["SmallDDQN"](nb["input_dim"], nb["action_dim"], nb["device"], w, gamma=0.0, lr=small_cfg["LR"])
        g = mk()
        cl = [nb["DPClient"](mk(), nb["X_dp"][ci], nb["y_dp"][ci], nb["device"], batch_size=small_cfg["BATCH"],
                             clip_norm=C) for ci in parts]
        sig = [1.0] * len(cl)
        fast = nb["make_fast"](cl)
        h, _ = nb["enh_episode"]("t", g, cl, sig, policy, nb["val_dataset"], rounds=2, steps=3, mu=0.01,
                                 rng=np.random.default_rng(0), C_base=C, eps_budget=1e6, fast=fast, check_budget=True)
        return h, cl, fast
    h1, cl1, _ = run(lambda s: (1.0, 1.0))
    h2, cl2, fast2 = run(lambda s: (2.0, 0.5))
    assert h2[0]["lr"] == 2 * small_cfg["LR"] and abs(h2[0]["clip"] - 0.5 * C) < 1e-9
    assert fast2.C == 0.5 * C and fast2.opt.param_groups[0]["lr"] == 2 * small_cfg["LR"]
    assert [h["epsilon"] for h in h1] == [h["epsilon"] for h in h2]     # same noise multiplier -> same epsilon
    assert h1[-1]["median_update_norm"] != h2[-1]["median_update_norm"]


def test_v6_meta_training_on_validation_only(nb, small_cfg):
    enh(small_cfg, TAG="enhm")
    seen = []
    orig = nb["DPClient"]
    class Spy(orig):
        def __init__(self, ddqn, X, y, *a, **k):
            seen.append(np.asarray(X).copy()); super().__init__(ddqn, X, y, *a, **k)
    nb["DPClient"] = Spy
    try:
        C = nb["dp_clip_norm"]()
        ctrl = nb["meta_train_controller_v6"](C)
    finally:
        nb["DPClient"] = orig
    ia, ib = nb["val_halves"]()
    A = {r.tobytes() for r in nb["X_val"][ia]}
    assert seen and all({r.tobytes() for r in X} <= A for X in seen)  # proxies train on validation half A only
    log = json.load(open(nb["controller_v6_path"]().replace(".pt", "_meta_log.json")))
    assert log["episodes_run"] == 2 and all(len(e["actions"]) == small_cfg["ROUNDS"] for e in log["episodes"])
    fresh = nb["ADPMetaV6"]((0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
    assert fresh.act([0.0] * 8, greedy=True) == (1.0, 1.0)              # starts at the fixed schedule


def test_enh_stage_config(nb, monkeypatch):
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 18b "))
    assert 'STAGE = "enh"' in src
    saved = dict(nb["CFG"]); calls = []
    monkeypatch.setitem(nb, "run_all", lambda exps, seeds: calls.append((dict(nb["CFG"]), nb["NUM_CLIENTS"], exps, seeds)))
    try:
        exec(src, nb)
    finally:
        nb["CFG"].clear(); nb["CFG"].update(saved)
    (cw, nw, ew, sw), (ce, ne, ee, se) = calls
    assert cw["TAG"] == "enhw" and cw["WEIGHTS_FROM"] == "val" and not cw["ENH"] and ew == [("dp_fixed", 8.0)]
    assert ce["TAG"] == "enh" and ce["ENH"] and ce["ADP_ACTIONS"] == "lr_clip" and ce["WEIGHTS_FROM"] == "val"
    assert ce["SERVER_MOMENTUM"] == 0.5 and ce["SERVER_EMA"] == 0.8 and ce["META_EPS"] == (8.0,)
    for c, n in ((cw, nw), (ce, ne)):                                   # required framework
        assert c["ROUNDS"] == 20 and n == 15 and c["NUM_CLIENTS_DP"] is None and c["PUBLIC_FRAC"] == 0.0
        assert c["CLIP_MULT"] == 0.5 and c["REWARD_FA"] == 0.5 and c["MU"] == 0.01 and not c["BINARY"]
    assert ee == [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)] and sw == se == [0]


def test_block17g_never_touches_test_set():
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 17g "))
    assert not re.search(r"\b(test_dataset|X_test|y_test)\b", src)
