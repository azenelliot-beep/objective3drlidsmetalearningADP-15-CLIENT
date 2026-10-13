"""C0 tests: selection rule, privacy guards, budget enforcement, partitioning, seeds, test-set guard."""
import copy, json, os, re
import numpy as np
import pytest
import torch

from conftest import notebook_cells


# ---------------- model-selection rule ----------------
def H(*rows):
    return [{"round": i + 1, "Macro_F1": f, "FAR": far} for i, (f, far) in enumerate(rows)]

def test_selection_prefers_best_f1_under_cap(nb):
    sel = nb["select_best_round"]
    h = H((0.30, 0.09), (0.37, 0.19), (0.32, 0.10), (0.20, 0.05))
    assert sel(h, 0.10) == (2, True)             # 0.37 is excluded by FAR 0.19; FAR == cap is allowed

def test_selection_ties_take_earliest(nb):
    assert nb["select_best_round"](H((0.3, 0.05), (0.3, 0.02)), 0.10) == (0, True)

def test_selection_falls_back_to_lowest_far(nb):
    i, ok = nb["select_best_round"](H((0.5, 0.30), (0.2, 0.12), (0.4, 0.12)), 0.10)
    assert (i, ok) == (2, False)                 # lowest FAR (tie at 0.12) -> higher macro-F1


# ---------------- DP client guards ----------------
def make_client(nb, clip, n=300, batch=64):
    X, y = nb["X_dp"][:n], nb["y_dp"][:n]
    ag = nb["SmallDDQN"](nb["input_dim"], nb["action_dim"], nb["device"], nb["class_weights_dp"], gamma=0.0)
    return nb["DPClient"](ag, X, y, nb["device"], batch_size=batch, clip_norm=clip, delta=1e-5)

def test_nonprivate_client_never_reports_epsilon(nb):
    c = make_client(nb, clip=None)
    params = c.ddqn.get_fl_parameters()
    _, _, m = c.fit(params, {"noise_multiplier": 0.0, "local_steps": 3})
    assert c.epsilon_spent() is None and m["epsilon_spent"] is None and m["dp_enabled"] is False
    assert not c.accountant.history

def test_mixing_private_and_nonprivate_steps_voids_epsilon(nb):
    c = make_client(nb, clip=1.0)
    c._dp_step(1.0)
    assert c.epsilon_spent() > 0
    c._dp_step(0.0)
    assert c.epsilon_spent() is None

def test_noise_without_clipping_is_refused(nb):
    c = make_client(nb, clip=None)
    with pytest.raises(AssertionError, match="unbounded sensitivity"):
        c._dp_step(1.0)

def test_per_sample_clipping_bounds_update(nb):
    C = 0.05
    c = make_client(nb, clip=C)
    captured = {}
    def fake_step():
        captured["g"] = torch.cat([p.grad.view(-1) for p in c.ddqn.online_net.parameters()])
    c.ddqn.optimizer.step = fake_step
    torch.manual_seed(0)
    n_sampled = int((torch.rand(c.n) < c.q).sum())        # same RNG draw the step will make
    torch.manual_seed(0)
    c._dp_step(0.0)                                        # sigma 0 -> grad = sum of clipped grads / expected_bs
    bound = n_sampled * C / (c.q * c.n)
    assert float(captured["g"].norm()) <= bound + 1e-6

def test_projected_epsilon_matches_spent(nb):
    c = make_client(nb, clip=1.0)
    proj = c.projected_epsilon(1.3, 7)
    for _ in range(7):
        c._dp_step(1.3)
    assert c.epsilon_spent() == pytest.approx(proj, rel=1e-9)


# ---------------- end-to-end runner ----------------
def test_fixed_noise_respects_budget_and_records_total_eps(nb, small_cfg):
    out = nb["run_dp_pipeline"]("dp_fixed", 0, 2.0, C=1.0)
    eps = [h["epsilon"] for h in out["history"]]
    assert all(e <= 2.0 + 1e-9 for e in eps) and eps == sorted(eps)
    assert out["final_epsilon"] == eps[-1] and out["rounds_run"] == small_cfg["ROUNDS"]
    assert out["final_epsilon"] > 0.9 * 2.0                 # calibrated to spend (almost) the whole budget

def test_adpmeta_respects_budget(nb, small_cfg):
    out = nb["run_dp_pipeline"]("dp_adpmeta", 0, 1.0, C=1.0)
    assert all(h["epsilon"] <= 1.0 + 1e-9 for h in out["history"])
    assert len(out["controller"]) == out["rounds_run"]

def test_nonprivate_methods_have_no_epsilon_and_no_clip(nb, small_cfg):
    out = nb["run_dp_pipeline"]("fedavg_np", 0)
    assert out["final_epsilon"] is None and out["clip_norm"] is None and out["mu"] == 0.0
    assert all(h["epsilon"] is None for h in out["history"])
    with pytest.raises(AssertionError):
        nb["run_dp_pipeline"]("dp_fixed", 0, None, C=1.0)  # private method without a budget
    with pytest.raises(AssertionError):
        nb["run_dp_pipeline"]("fedprox_np", 0, 8.0)         # non-private method with a budget

def test_every_record_in_exactly_one_client_and_seed_changes_split(nb):
    split = lambda s: nb["dirichlet_split"](nb["y_dp"], nb["NUM_CLIENTS"], nb["ALPHA"],
                                            np.random.default_rng(s), nb["MIN_PER_CLIENT"])
    for s in (0, 1, 2):
        a = np.concatenate(split(s))
        assert len(a) == len(nb["y_dp"]) == len(np.unique(a))
    assert [len(p) for p in split(0)] == [len(p) for p in split(0)]
    assert [len(p) for p in split(0)] != [len(p) for p in split(1)]

def test_same_seed_reproduces_history(nb, small_cfg):
    a = nb["run_dp_pipeline"]("fedprox_np", 5)
    os.remove(nb["result_path"]("fedprox_np", 5))
    b = nb["run_dp_pipeline"]("fedprox_np", 5)
    assert [h["Macro_F1"] for h in a["history"]] == [h["Macro_F1"] for h in b["history"]]

def test_run_all_summary_and_test_guard(nb, small_cfg, tmp_path):
    exps = [("fedprox_np", None), ("dp_fixed", 3.0), ("fedavg_ch5", None)]
    nb["run_all"](exps, [0, 1])
    d = small_cfg["RESULTS_DIR"]
    files = sorted(os.listdir(d))
    assert len([f for f in files if f.endswith(".json")]) == 6 and not any("_TEST" in f for f in files)
    for f in files:
        if f.endswith(".json"):
            assert "test" not in json.load(open(os.path.join(d, f)))   # no test numbers before Block 19
    vdf = nb["summarize_val"]()
    assert set(vdf["run"]) == {"fedprox_np", "dp_fixed_eps3", "fedavg_ch5"} and len(vdf) == 6
    nb["run_all"](exps, [0])                                            # resumes: everything skipped
    assert sorted(os.listdir(d)) == files

    block19 = next(s for s in notebook_cells() if s.startswith("# BLOCK 19 "))
    exec(block19, nb)                                                   # FINAL_TEST = False
    assert not any("_TEST" in f for f in os.listdir(d))
    exec(block19.replace("FINAL_TEST = False", "FINAL_TEST = True").replace('TEST_TAG = "v4"', 'TEST_TAG = ""'), nb)
    assert len([f for f in os.listdir(d) if f.endswith("_TEST.json")]) == 6
    t = nb["table"]
    assert "±" in t.loc["dp_fixed_eps3", "MacroF1"] and t.loc["fedprox_np", "eps"] == "-"


# ---------------- static checks on the notebook ----------------
def test_no_leak_columns_in_features(nb):
    cells = notebook_cells()
    b5 = next(s for s in cells if s.startswith("# BLOCK 5 "))
    for col in ['label', 'ts', 'src_ip', 'dst_ip', 'src_port', 'dst_port']:
        assert f"'{col}'" in b5, f"Block 5 no longer drops {col}"
    assert "assert c not in X.columns" in next(s for s in cells if s.startswith("# BLOCK 7 "))

def test_runner_never_touches_test_set():
    src = next(s for s in notebook_cells() if s.startswith("# ====") and "BLOCK 18a" in s)
    assert not re.search(r"\b(test_dataset|X_test|y_test)\b", src)
