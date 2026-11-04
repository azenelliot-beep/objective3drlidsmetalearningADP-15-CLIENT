"""Mechanism tests required by candidate/RUN_WITH_CLAUDE.md (step 6), executed on the candidate notebook
(candidate/chapter6_ddqn_adp_candidate_repaired.ipynb: received notebook + one vmap-safe indexing repair) with the synthetic ToN-IoT-like data of conftest.py."""
import json, os, pathlib, re
import numpy as np
import pytest
import torch

from conftest import SKIP_PREFIXES, SYNTHETIC

CAND = pathlib.Path(__file__).resolve().parents[1] / "candidate" / "chapter6_ddqn_adp_candidate_repaired.ipynb"


def cand_cells():
    nb = json.loads(CAND.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def cand(tmp_path_factory):
    ns = {"__name__": "__cand__"}
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("cand_working"))
    done = False
    try:
        for src in cand_cells():
            if src.startswith(SKIP_PREFIXES):
                if src.startswith("# BLOCK 4 ") and not done:
                    exec(SYNTHETIC, ns); done = True
                continue
            exec(compile(src, "<cand-cell>", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns


@pytest.fixture
def ccfg(cand, tmp_path):
    saved = dict(cand["CFG"])
    cand["CFG"].update(ROUNDS=3, LOCAL_STEPS=4, BATCH=64, RESULTS_DIR=str(tmp_path / "results"))
    yield cand["CFG"]
    cand["CFG"].clear(); cand["CFG"].update(saved)


def client(ns, seed=0, n=300, clip=0.5, batch=64):
    torch.manual_seed(seed)
    ag = ns["SmallDDQN"](ns["input_dim"], ns["action_dim"], ns["device"], ns["reward_weights"](), gamma=0.0, lr=1e-3)
    return ns["DPClient"](ag, ns["X_dp"][:n], ns["y_dp"][:n], ns["device"], batch_size=batch, clip_norm=clip)


def params(c):
    return [p.detach().clone() for p in c.ddqn.online_net.parameters()]


def test_patch_is_active(cand):
    assert cand["DPClient"]._dp_step is cand["patched_private_step"]
    assert cand["DPClient"].fit is cand["patched_fit"]
    assert cand["ADPMetaV2"].finish_episode is cand["patched_finish_episode"]
    assert cand["CFG"]["FAST"] is False and cand["CFG"]["WEIGHTS_FROM"] == "val" and cand["CFG"]["DDQN_GAMMA"] == 0.1
    ia, _ = cand["val_halves"]()
    assert torch.equal(cand["PUBLIC_DDQN_ANCHORS"].cpu(), torch.as_tensor(cand["X_val"][ia]))   # anchors = val half A


def test_one_private_step(cand):
    c = client(cand)
    before = params(c)
    c._dp_step(1.0)
    after = params(c)
    assert len(c.accountant.history) == 1 and c.nonprivate_steps == 0
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    assert all(torch.isfinite(p.grad).all() for p in c.ddqn.online_net.parameters())
    assert c.epsilon_spent() > 0


def test_empty_batch_still_adds_calibrated_noise(cand):
    c = client(cand, clip=0.5)
    c.q = 1e-9                                                    # Poisson sampling selects nobody
    torch.manual_seed(1)
    c._dp_step(2.0)
    assert len(c.accountant.history) == 1                         # the step is still accounted
    g = torch.cat([p.grad.reshape(-1) for p in c.ddqn.online_net.parameters()]) * (c.q * c.n)
    assert torch.isfinite(g).all() and g.abs().sum() > 0
    assert abs(float(g.std()) / (2.0 * 0.5) - 1.0) < 0.05         # pure noise with std sigma * C


def test_noise_is_positive_gaussian_and_absent_at_sigma_zero(cand):
    c0 = client(cand); c0.q = 1e-9
    c0._dp_step(0.0)
    g0 = torch.cat([p.grad.reshape(-1) for p in c0.ddqn.online_net.parameters()])
    assert float(g0.abs().sum()) == 0.0 and c0.nonprivate_steps == 1 and c0.epsilon_spent() is None
    stds = []
    for s in (0.5, 1.0, 4.0):
        c = client(cand); c.q = 1e-9
        c._dp_step(s)
        stds.append(float((torch.cat([p.grad.reshape(-1) for p in c.ddqn.online_net.parameters()]) * c.q * c.n).std()))
    assert stds[0] < stds[1] < stds[2] and abs(stds[2] / stds[1] - 4.0) < 0.3


def _two_steps(cand, gamma, perturb_target, seed=3):
    cand["CFG"]["DDQN_GAMMA"] = gamma
    c = client(cand, seed=0)
    if perturb_target:
        with torch.no_grad():
            for p in c.ddqn.target_net.parameters():
                p.add_(0.5)
    torch.manual_seed(seed)
    c._dp_step(0.0); c._dp_step(0.0)
    return params(c)


def test_target_network_influence_gamma_positive_vs_zero(cand):
    saved = cand["CFG"]["DDQN_GAMMA"]
    try:
        same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a, b))
        assert same(_two_steps(cand, 0.1, False), _two_steps(cand, 0.1, False))      # matched randomness reproduces
        assert not same(_two_steps(cand, 0.1, False), _two_steps(cand, 0.0, False))  # gamma changes the update
        assert not same(_two_steps(cand, 0.1, False), _two_steps(cand, 0.1, True))   # target net matters if gamma > 0
        assert same(_two_steps(cand, 0.0, False), _two_steps(cand, 0.0, True))       # ... and not at all if gamma = 0
    finally:
        cand["CFG"]["DDQN_GAMMA"] = saved


def test_fedprox_term_is_active(cand):
    grads = {}
    for mu in (0.0, 0.5):
        c = client(cand, seed=0)
        c.global_params = {k: v.detach().clone() + 0.1 for k, v in c.ddqn.online_net.named_parameters()}
        c.mu = mu
        p0 = {k: v.detach().clone() for k, v in c.ddqn.online_net.named_parameters()}
        torch.manual_seed(5)
        c._dp_step(1.0)
        grads[mu] = {k: p.grad.detach().clone() for k, p in c.ddqn.online_net.named_parameters()}
    for k in grads[0.0]:
        expected = 0.5 * (p0[k] - (p0[k] + 0.1))                   # mu * (w - w_global)
        assert torch.allclose(grads[0.5][k] - grads[0.0][k], expected, atol=1e-6)


def test_gradients_finite_over_many_steps(cand):
    c = client(cand)
    for _ in range(25):
        c._dp_step(1.1)
        assert all(torch.isfinite(p.grad).all() and torch.isfinite(p).all() for p in c.ddqn.online_net.parameters())


def test_first_episode_policy_gradient_without_entropy(cand):
    torch.manual_seed(0)
    ctrl = cand["ADPMetaV2"](cand["CFG"]["PACE_ACTIONS"], lr=1e-2, entropy_coef=0.0, device=cand["device"])
    before = [p.detach().clone() for p in ctrl.parameters()]
    for t in range(3):
        ctrl.act([t / 3, 1 - t / 3, 0.2 * t, 0.1, 0.5])
    ctrl.finish_episode([0.1, 0.2, 0.3])                          # first episode: baseline 0 -> nonzero advantage
    assert ctrl.last_policy_grad_norm > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, ctrl.parameters()))


def test_frozen_policy_output_varies_with_state(cand):
    torch.manual_seed(0)
    ctrl = cand["ADPMetaV2"](cand["CFG"]["PACE_ACTIONS"], device=cand["device"])
    states = [[t, 1 - t, f1, 0.0, u] for t in (0.05, 0.5, 0.95) for f1 in (0.1, 0.6) for u in (0.0, 3.0)]
    with torch.no_grad():
        probs = torch.stack([torch.softmax(ctrl.policy(torch.tensor(s)), -1) for s in states])
    assert float((probs.max(0).values - probs.min(0).values).max()) > 1e-3   # outputs depend on the state


def test_repair_stage_runs_end_to_end(cand, ccfg):
    ccfg.update(CLIP_MULT=0.5, REWARD_FA=0.5, ADP_ACTIONS="pacing", ADP_META_EPISODES=1, PROXY_MIN_PER_CLIENT=5,
                TAG="repair_t")
    cand["run_all"]([("dp_fixed", 2.0), ("dp_adpmeta", 2.0)], [0])
    for m in ("dp_fixed", "dp_adpmeta"):
        r = json.load(open(cand["result_path"](cand["run_name"](m, 2.0), 0)))
        eps = [h["epsilon"] for h in r["history"]]
        assert len(eps) == ccfg["ROUNDS"] and max(eps) <= 2.0 + 1e-9 and eps[-1] >= 0.95 * 2.0
        assert r["cfg"]["DDQN_GAMMA"] == 0.1 and r["cfg"]["FAST"] is False


def test_repair_check_stage_config(cand, monkeypatch):
    src = next(s for s in cand_cells() if s.startswith("# BLOCK 18b "))
    assert 'STAGE = "repair_check"' in src
    saved = dict(cand["CFG"]); calls = []
    monkeypatch.setitem(cand, "run_all", lambda e, s: calls.append((dict(cand["CFG"]), cand["NUM_CLIENTS"], e, s)))
    try:
        exec(src, cand)
    finally:
        cand["CFG"].clear(); cand["CFG"].update(saved)
    (cr, nr, er, sr), (cc, nc, ec, sc) = calls
    for c, n in ((cr, nr), (cc, nc)):
        assert c["ROUNDS"] == 20 and n == 15 and c["DDQN_GAMMA"] == 0.1 and c["FAST"] is False
        assert c["WEIGHTS_FROM"] == "val" and c["CLIP_MULT"] == 0.5 and c["REWARD_FA"] == 0.5
        assert c["ADP_ACTIONS"] == "pacing" and c["ADP_META_EPISODES"] == 10
    assert (cr["BATCH"], cr["LOCAL_STEPS"], cr["LR"]) == (256, 400, 1e-3)
    assert (cc["BATCH"], cc["LOCAL_STEPS"], cc["LR"]) == (512, 200, 3e-4)
    assert er == ec == [("dp_fixed", 8.0), ("dp_adpmeta", 8.0)] and sr == sc == [0]
