"""ADP-Meta v3 (Block 17c): joint pacing + adaptive clip, twin-baseline meta-training."""
import json, os, re
import numpy as np
import pytest
import torch

from conftest import notebook_cells


def test_policy_starts_at_fixed_schedule_and_uses_own_rng(nb):
    c = nb["ADPMetaV3"]((0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
    s = [0.0] * 9
    assert c.act(s, greedy=True) == (1.0, 1.0)              # initial bias -> fixed schedule
    lin = [m for m in c.trunk if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in lin] == [32, 16]
    torch.manual_seed(7); before = torch.rand(3)
    torch.manual_seed(7); c.act(s); after = torch.rand(3)
    assert torch.equal(before, after)                       # sampling actions does not move the global RNG

def test_policy_learns_toy_advantage(nb):
    torch.manual_seed(0)
    c = nb["ADPMetaV3"]((0.5, 1.0, 2.0), (0.5, 1.0, 2.0))   # default lr 1e-2, entropy 0.01
    s = [0.1] * 9
    for _ in range(300):                                    # clip 0.5 and pace 2.0 pay off
        k, m = c.act(s)
        c.finish_episode((m == 0.5) * 1.0 + (k == 2.0) * 0.5)
    assert c.act(s, greedy=True) == (2.0, 0.5)

def test_episode_score_uses_selection_rule(nb):
    h = [{"round": 1, "Macro_F1": 0.5, "FAR": 0.3}, {"round": 2, "Macro_F1": 0.3, "FAR": 0.05}]
    assert nb["episode_score"](h) == pytest.approx(0.3)
    h2 = [{"round": 1, "Macro_F1": 0.5, "FAR": 0.3}, {"round": 2, "Macro_F1": 0.4, "FAR": 0.2}]
    assert nb["episode_score"](h2) == pytest.approx(0.4 - 0.1)

def test_v3_meta_training_and_real_run(nb, small_cfg, monkeypatch):
    small_cfg.update(CLIP_MULT=0.5, REWARD_FA=0.5, ADP_ACTIONS="pacing_clip", ADP_META_EPISODES=4,
                     META_EPS=(2.0, 1.0), TWIN_POOL=1, PROXY_MIN_PER_CLIENT=5, TAG="v3")
    seen = []
    real_init = nb["DPClient"].__init__
    def spy(self, ddqn, X_local, y_local, *a, **k):
        seen.append((k.get("model_name", ""), np.asarray(X_local)))
        real_init(self, ddqn, X_local, y_local, *a, **k)
    monkeypatch.setattr(nb["DPClient"], "__init__", spy)
    out = nb["run_dp_pipeline"]("dp_adpmeta", 0, 2.0, C=1.0)
    ia, ib = nb["val_halves"]()
    rows = lambda X: {r.tobytes() for r in X}
    A, B = rows(nb["X_val"][ia]), rows(nb["X_val"][ib])
    proxies = [X for n, X in seen if n.startswith("Proxy")]
    reals = [X for n, X in seen if n.startswith("Client")]
    K = nb["NUM_CLIENTS"]
    assert len(proxies) == (4 + 2) * K                      # 4 practice episodes + 2 cached twins (1 seed x 2 eps)
    assert all(rows(X) <= A for X in proxies)
    assert all(not (rows(X) & (A | B)) for X in reals)
    log = json.load(open(out["controller_file"].replace(".pt", "_meta_log.json")))
    assert [e["eps"] for e in log["episodes"]] == [2.0, 1.0, 2.0, 1.0]
    assert out["adp_version"] == "v3-pacing-clip"
    for h in out["history"]:
        assert h["epsilon"] <= 2.0 + 1e-9
        assert h["clip"] == pytest.approx(1.0 * h["clip_mult"]) and h["clip_mult"] in (0.5, 1.0, 2.0)
    assert out["final_epsilon"] >= 0.95 * 2.0
    seen.clear()                                            # other eps + other seed reuse the shared controller
    out1 = nb["run_dp_pipeline"]("dp_adpmeta", 1, 1.0, C=1.0)
    assert not [n for n, _ in seen if n.startswith("Proxy")] and out1["final_epsilon"] <= 1.0 + 1e-9

def test_clip_changes_do_not_change_accounting(nb):
    def run(ms):
        torch.manual_seed(0)
        cs = [nb["DPClient"](nb["SmallDDQN"](nb["input_dim"], nb["action_dim"], nb["device"], nb["class_weights_dp"],
                                             gamma=0.0), nb["X_dp"][:200], nb["y_dp"][:200], nb["device"],
                             batch_size=32, clip_norm=1.0)]
        for m in ms:
            cs[0].C = m
            cs[0].fit(cs[0].ddqn.get_fl_parameters(), {"noise_multiplier": 1.2, "local_steps": 3})
        return cs[0].epsilon_spent()
    assert run([1.0, 1.0, 1.0]) == pytest.approx(run([0.5, 2.0, 1.0]))

def test_v3_stage_and_block19_tag():
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 18b "))
    assert 'STAGE = "v3_val"' in src and '"final_v3"' in src
    b19 = next(s for s in notebook_cells() if s.startswith("# BLOCK 19 "))
    assert 'TEST_TAG = "v3"' in b19 and "FINAL_TEST = False" in b19


def test_eps16_stage_uses_frozen_final_settings(nb, monkeypatch):
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 18b "))
    calls = []
    saved = dict(nb["CFG"])
    monkeypatch.setitem(nb, "run_all", lambda exps, seeds: calls.append((dict(nb["CFG"]), exps, seeds)))
    try:
        exec(re.sub(r'^STAGE = ".*"$', 'STAGE = "eps16"', src, flags=re.M), nb)
    finally:
        nb["CFG"].clear(); nb["CFG"].update(saved)
    (cfg, exps, seeds), = calls
    assert cfg["TAG"] == "final" and cfg["ADP_ACTIONS"] == "pacing" and cfg["ADP_META_EPISODES"] == 10
    assert cfg["CLIP_MULT"] == 0.5 and cfg["REWARD_FA"] == 0.5
    assert exps == [("dp_fixed", 16.0), ("dp_adpmeta", 16.0)] and seeds == [0, 1, 2]
