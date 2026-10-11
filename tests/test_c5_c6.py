"""C5/C6 tests: RDP pacing, ADP-Meta v2, meta-training on validation halves, config guard."""
import json, os, re
import numpy as np
import pytest
import torch

from conftest import notebook_cells


# ---------------- pacing ----------------
def pacer(nb, eps=3.0, q=0.05, rounds=5, steps=20):
    return nb["RDPPacer"](q, eps, 1e-5, rounds, steps)

def test_uniform_pacing_reproduces_fixed_noise(nb):
    p = pacer(nb)
    for r in range(5):
        s = p.sigma_for_share(nb["pacing_share"](1.0, 5 - r))
        assert s == pytest.approx(p.sigma0, rel=1e-6)
        p.commit(s)
    assert p.spent == pytest.approx(p.R, rel=1e-6)

@pytest.mark.parametrize("seq_seed", [0, 1, 2, 3])
def test_any_action_sequence_stays_within_budget_and_spends_it(nb, seq_seed):
    from opacus.accountants import RDPAccountant
    eps, q, rounds, steps = 3.0, 0.05, 6, 20
    p, acc = pacer(nb, eps, q, rounds, steps), RDPAccountant()
    ks = np.random.default_rng(seq_seed).choice([0.5, 0.75, 1.0, 1.5, 2.0], size=rounds)
    for r, k in enumerate(ks):
        s = p.sigma_for_share(nb["pacing_share"](k, rounds - r))
        assert s is not None
        p.commit(s); acc.history.append((s, q, steps))
        assert p.spent <= p.R * (1 + 1e-9)
    final = acc.get_epsilon(delta=1e-5)
    assert final <= eps + 1e-9                     # Opacus epsilon (all orders) never above the budget
    assert final >= 0.95 * eps                     # last round spends what is left

def test_pacing_share_last_round_spends_all(nb):
    assert nb["pacing_share"](0.5, 1) == 1.0 and nb["pacing_share"](2.0, 3) == pytest.approx(2 / 3)
    assert nb["pacing_share"](2.0, 1) == 1.0 and nb["pacing_share"](5.0, 2) == 1.0


# ---------------- controller ----------------
def test_policy_is_32_16_and_learns_a_toy_bandit(nb):
    torch.manual_seed(0)
    c = nb["ADPMetaV2"]((0.5, 1.0, 2.0), lr=5e-2, entropy_coef=0.0)
    sizes = [m.out_features for m in c.policy if isinstance(m, torch.nn.Linear)]
    assert sizes == [32, 16, 3]
    state = [0.5, 0.5, 0.3, 0.0, 0.0]
    for _ in range(300):                           # 1-step episodes; action 2 pays most
        a = c.act(state)
        c.finish_episode([[0.0, 0.2, 1.0][a]])
    probs = torch.softmax(c.policy(torch.tensor(state)), -1)
    assert int(probs.argmax()) == 2 and float(probs[2]) > 0.8

def test_greedy_mode_records_nothing(nb):
    c = nb["ADPMetaV2"]((0.5, 1.0), device="cpu")
    c.act([0.1] * 5, greedy=True)
    assert c._logp == [] and c._ent == []


# ---------------- end-to-end ----------------
def test_paced_run_online_respects_budget(nb, small_cfg):
    small_cfg.update(ADP_ACTIONS="pacing", TAG="c5")
    out = nb["run_dp_pipeline"]("dp_adpmeta", 0, 2.0, C=1.0)
    assert out["adp_mode"] == "sample_online" and out["rounds_run"] == small_cfg["ROUNDS"]
    assert all(h["epsilon"] <= 2.0 + 1e-9 for h in out["history"])
    assert out["final_epsilon"] >= 0.95 * 2.0
    assert out["run"] == "dp_adpmeta_eps2__c5"

def test_meta_training_uses_only_validation_and_is_frozen(nb, small_cfg, monkeypatch):
    small_cfg.update(ADP_ACTIONS="pacing", ADP_META_EPISODES=2, PROXY_MIN_PER_CLIENT=5, TAG="c6")
    seen = []
    real_init = nb["DPClient"].__init__
    def spy(self, ddqn, X_local, y_local, *a, **k):
        seen.append((k.get("model_name", ""), np.asarray(X_local)))
        real_init(self, ddqn, X_local, y_local, *a, **k)
    monkeypatch.setattr(nb["DPClient"], "__init__", spy)
    out = nb["run_dp_pipeline"]("dp_adpmeta", 0, 2.0, C=1.0)
    ia, ib = nb["val_halves"]()
    assert len(set(ia) & set(ib)) == 0 and len(ia) + len(ib) == len(nb["y_val"])
    rows = lambda X: {r.tobytes() for r in X}
    A, B, private = rows(nb["X_val"][ia]), rows(nb["X_val"][ib]), rows(nb["X_dp"])
    proxies = [X for n, X in seen if n.startswith("Proxy")]
    reals = [X for n, X in seen if n.startswith("Client")]
    assert len(proxies) == 2 * nb["NUM_CLIENTS"] and len(reals) == nb["NUM_CLIENTS"]
    assert all(rows(X) <= A for X in proxies)                 # proxies: validation half A only
    assert all(rows(X) <= private and not (rows(X) & (A | B)) for X in reals)   # real detector: no validation data
    assert out["adp_mode"] == "greedy" and os.path.exists(out["controller_file"])
    log = json.load(open(out["controller_file"].replace(".pt", "_meta_log.json")))
    assert len(log["episodes"]) == 2
    assert all(h["epsilon"] <= 2.0 + 1e-9 for h in out["history"])
    # second seed reuses the cached controller (no new practice episodes)
    seen.clear()
    nb["run_dp_pipeline"]("dp_adpmeta", 1, 2.0, C=1.0)
    assert not [n for n, _ in seen if n.startswith("Proxy")]

def test_config_change_without_new_tag_is_refused(nb, small_cfg):
    nb["run_all"]([("dp_fixed", 2.0)], [0])
    small_cfg["BATCH"] = 32
    with pytest.raises(ValueError, match="DIFFERENT config"):
        nb["run_all"]([("dp_fixed", 2.0)], [0])
    small_cfg["TAG"] = "b32"
    nb["run_all"]([("dp_fixed", 2.0)], [0])
    assert os.path.exists(nb["result_path"]("dp_fixed_eps2__b32", 0))

def test_block17b_never_touches_test_set():
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 17b "))
    assert not re.search(r"\b(test_dataset|X_test|y_test)\b", src)


def test_block18b_stages_apply_overrides_on_fresh_defaults(nb, monkeypatch):
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 18b "))
    calls = []
    saved = dict(nb["CFG"])
    monkeypatch.setitem(nb, "run_all", lambda exps, seeds: calls.append((dict(nb["CFG"]), exps, seeds)))
    try:
        exec(src.replace('STAGE = "C0_baseline"', 'STAGE = "C6_meta"'), nb)
        exec(src.replace('STAGE = "C0_baseline"', 'STAGE = "C0_baseline"'), nb)
    finally:
        nb["CFG"].clear(); nb["CFG"].update(saved)
    c6, c0 = calls
    assert c6[0]["ADP_META_EPISODES"] == 10 and c6[0]["TAG"] == "C6" and c6[1] == [("dp_adpmeta", 8.0)]
    assert c0[0]["ADP_META_EPISODES"] == 0 and c0[0]["TAG"] == "" and c0[0]["ADP_ACTIONS"] == "mult"
