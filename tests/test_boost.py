"""Boost scenarios (Block 17f): binary task, fewer/larger clients, 5% public pre-training. Pillars unchanged."""
import json, os, re
import numpy as np
import pytest
import torch

from conftest import notebook_cells

KEYS = ("class_names", "X_dp", "y_dp", "y_val", "y_test", "NUM_CLIENTS", "action_dim", "NORMAL_IDX",
        "class_weights_dp")


def snap(nb):
    return {k: (np.array(nb[k], copy=True) if isinstance(nb[k], np.ndarray) else nb[k]) for k in KEYS}


def same_snap(a, b):
    for k in KEYS:
        if isinstance(a[k], np.ndarray):
            assert np.array_equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k
    return True


@pytest.fixture
def restore(nb):
    yield
    nb["apply_scenario"]()


def test_default_scenario_is_identity(nb, restore):
    before = snap(nb)
    nb["apply_scenario"]()
    assert same_snap(before, snap(nb)) and nb["X_pub"] is None


def test_binary_mapping_and_exact_restore(nb, restore):
    before = snap(nb)
    n = before["class_names"].index("normal")
    nb["apply_scenario"](binary=True)
    assert nb["class_names"] == ["attack", "normal"] and nb["action_dim"] == 2 and nb["NORMAL_IDX"] == 1
    assert np.array_equal(nb["y_val"], (before["y_val"] == n).astype(np.int64))
    assert np.array_equal(nb["y_test"], (before["y_test"] == n).astype(np.int64))
    assert np.array_equal(nb["val_dataset"].tensors[1].numpy(), nb["y_val"])
    assert len(nb["class_weights_dp"]) == 2
    nb["apply_scenario"]()
    assert same_snap(before, snap(nb))
    assert np.array_equal(nb["val_dataset"].tensors[1].numpy(), before["y_val"])


def test_public_slice_disjoint_from_clients(nb, restore):
    X0, y0 = nb["X_dp"].copy(), nb["y_dp"].copy()
    nb["apply_scenario"](public_frac=0.05)
    Xp, Xd = nb["X_pub"], nb["X_dp"]
    assert len(Xp) + len(Xd) == len(X0) and abs(len(Xp) - 0.05 * len(X0)) <= len(np.unique(y0))
    rows = lambda X: {r.tobytes() for r in X}
    assert not (rows(Xp) & rows(Xd))                                   # no record is both public and private
    parts = nb["dp_client_indices"]
    allc = np.concatenate(parts)
    assert len(allc) == len(Xd) and len(np.unique(allc)) == len(Xd)    # clients hold only private rows
    assert set(np.unique(nb["y_pub"])) == set(np.unique(y0))           # stratified


def test_fewer_clients(nb, restore):
    nb["apply_scenario"](num_clients=3)
    assert nb["NUM_CLIENTS"] == 3 and len(nb["dp_client_indices"]) == 3


def test_public_init_keeps_rng_and_is_cached(nb, small_cfg, restore):
    small_cfg.update(PUBLIC_FRAC=0.05, PRETRAIN_STEPS=20)
    nb["apply_scenario"](public_frac=0.05)
    torch.manual_seed(7); np.random.seed(7)
    a = torch.rand(3); b = np.random.rand(3)
    torch.manual_seed(7); np.random.seed(7)
    p1 = nb["public_init"](0)
    assert torch.equal(torch.rand(3), a) and np.allclose(np.random.rand(3), b)
    p2 = nb["public_init"](0)
    assert all(torch.equal(x, y) for x, y in zip(p1, p2))
    ag = nb["SmallDDQN"](nb["input_dim"], nb["action_dim"], nb["device"], nb["reward_weights"](), gamma=0.0)
    nb["maybe_public_init"](ag, 0)
    assert all(torch.equal(x, y) for x, y in zip(ag.get_fl_parameters(), p1))


@pytest.mark.parametrize("binary", [False, True])
def test_boost_pipeline_end_to_end(nb, small_cfg, restore, binary):
    small_cfg.update(CLIP_MULT=0.5, REWARD_FA=0.5, ADP_ACTIONS="pacing", ADP_META_EPISODES=1,
                     PROXY_MIN_PER_CLIENT=5, BINARY=binary, NUM_CLIENTS_DP=5, PUBLIC_FRAC=0.05,
                     PRETRAIN_STEPS=20, TAG="bstbin" if binary else "bst10")
    nb["scenario_from_cfg"](small_cfg)
    nb["run_all"]([("dp_fixed", 2.0), ("dp_adpmeta", 2.0)], [0])
    d = small_cfg["RESULTS_DIR"]
    for m in ("dp_fixed", "dp_adpmeta"):
        r = json.load(open(nb["result_path"](nb["run_name"](m, 2.0), 0)))
        assert r["cfg"]["NUM_CLIENTS_DP"] == 5 and r["cfg"]["PUBLIC_FRAC"] == 0.05 and r["cfg"]["BINARY"] is binary
        assert all(h["epsilon"] <= 2.0 + 1e-9 for h in r["history"])
        f1 = r["val_at_best"]["F1_per_class"]
        assert (set(f1) == {"attack", "normal"}) if binary else len(f1) == 10
    nb["apply_scenario"]()
    exec(re.sub(r'^CALIB_TAG = "[\w,]*"', f'CALIB_TAG = "{small_cfg["TAG"]}"',
                next(s for s in notebook_cells() if s.startswith("# BLOCK 18d ")), flags=re.M), nb)
    assert len(nb["cal_df"]) == 2
    assert nb["class_names"][0] == "backdoor"                          # 18d restores the Chapter 6 data


def test_boost_stage_config(nb, monkeypatch):
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 18b "))
    assert 'STAGE = "boost"' in src
    saved = dict(nb["CFG"]); calls = []
    monkeypatch.setitem(nb, "run_all", lambda exps, seeds: calls.append((dict(nb["CFG"]), nb["NUM_CLIENTS"],
                                                                         nb["action_dim"], exps, seeds)))
    try:
        exec(src, nb)
    finally:
        nb["CFG"].clear(); nb["CFG"].update(saved)
    (c10, n10, a10, e10, s10), (cb, nb_, ab, eb, sb) = calls
    assert c10["TAG"] == "bst10" and c10["BINARY"] is False and a10 == 10
    assert cb["TAG"] == "bstbin" and cb["BINARY"] is True and ab == 2
    for c, n in ((c10, n10), (cb, nb_)):
        assert c["NUM_CLIENTS_DP"] == 5 and n == 5 and c["PUBLIC_FRAC"] == 0.05
        assert c["CLIP_MULT"] == 0.5 and c["REWARD_FA"] == 0.5 and c["MU"] == 0.01
    assert e10 == eb == [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)] and s10 == sb == [0]
    assert nb["action_dim"] == 10 and nb["X_pub"] is None              # stage leaves the default data behind


@pytest.mark.parametrize("prefix", ["# BLOCK 17f "])
def test_block17f_never_touches_test_set_for_training(prefix):
    src = next(s for s in notebook_cells() if s.startswith(prefix))
    body = src.split("def public_init")[1]
    assert not re.search(r"\b(test_dataset|X_test|y_test)\b", body)
