"""ADP-Meta v4 / Option 2 (Block 17d + projection in Blocks 16/16b): public-subspace projection, meta-learned rank,
fair baseline dp_fixed_proj. Existing components (DDQN, FedProx, DP-SGD, RDP accounting, pacing, clip) must stay."""
import json, os, re
import numpy as np
import pytest
import torch

from conftest import notebook_cells


def clients(nb, n_clients=3, n_per=40, clip=0.5, seed=0, eps_greedy=0.0):
    """q = 1 and greedy actions -> deterministic sampling/actions (same trick as test_fast.py)."""
    torch.manual_seed(seed)
    cs = []
    for k in range(n_clients):
        ag = nb["SmallDDQN"](nb["input_dim"], nb["action_dim"], nb["device"], nb["class_weights_dp"],
                             gamma=0.0, eps_start=eps_greedy, eps_end=eps_greedy)
        X, y = nb["X_dp"][k * n_per:(k + 1) * n_per + k], nb["y_dp"][k * n_per:(k + 1) * n_per + k]
        cs.append(nb["DPClient"](ag, X, y, nb["device"], batch_size=10_000, clip_norm=clip))
    return cs


def dim(nb):
    return sum(p.numel() for p in nb["SmallQNetwork"](nb["input_dim"], nb["action_dim"]).parameters())


def rand_basis(d, k, seed=0):
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=torch.Generator().manual_seed(seed)))
    return q


def fit_round(cs, basis, sigma, steps=3, mu=0.0, fast=None, nb=None):
    for c in cs:
        c.basis = basis
    if fast is not None:
        fast.basis = basis
    gp = cs[0].ddqn.get_fl_parameters()
    return nb["fit_clients"](cs, list(range(len(cs))), gp, {i: sigma for i in range(len(cs))}, steps, mu, fast)


# ---------------- the projection itself ----------------
@pytest.mark.parametrize("clip", [0.5, None])
def test_identity_basis_equals_full_space_step(nb, clip):
    d = dim(nb)
    a, b = clients(nb, clip=clip), clients(nb, clip=clip)
    pa, _, _ = fit_round(a, None, 0.0, nb=nb)
    pb, _, _ = fit_round(b, torch.eye(d), 0.0, nb=nb)
    for x, y in zip(pa, pb):
        for tx, ty in zip(x, y):
            assert torch.allclose(tx, ty, atol=1e-5, rtol=1e-4)

@pytest.mark.parametrize("mu", [0.0, 0.1])
def test_fast_path_equals_loop_with_basis(nb, mu):
    V = rand_basis(dim(nb), 16)
    slow, fastc = clients(nb), clients(nb)
    fast = nb["FastFederation"](fastc)
    ps, _, _ = fit_round(slow, V, 0.0, mu=mu, nb=nb)
    pf, _, _ = fit_round(fastc, V, 0.0, mu=mu, fast=fast, nb=nb)
    for x, y in zip(ps, pf):
        for tx, ty in zip(x, y):
            assert torch.allclose(tx, ty, atol=1e-5, rtol=1e-4)

def test_noise_lives_in_subspace_with_scale_sigma_C(nb):
    d, k = dim(nb), 32
    V = rand_basis(d, k, seed=3)
    cs = clients(nb, clip=0.5)
    fast = nb["FastFederation"](cs)
    fast.basis = V
    gp = {n: v.detach().clone() for n, v in zip(fast.names, cs[0].ddqn.get_fl_parameters())}
    def grad_at(sig):
        for n in fast.names:
            with torch.no_grad():
                fast.params[n].copy_(gp[n].unsqueeze(0).expand_as(fast.params[n]))
        fast.opt = torch.optim.Adam([fast.params[n] for n in fast.names], lr=1e-3); fast.steps = 0
        torch.manual_seed(1); fast._step(sig, 0.0, gp)
        return torch.cat([fast.last_grad[n].reshape(3, -1) for n in fast.names], 1)
    sig = torch.tensor([1.0, 2.0, 4.0])
    noise = (grad_at(sig) - grad_at(torch.zeros(3))) * (fast.q * fast.n.float())[:, None]   # undo / expected batch
    outside = noise - (noise @ V) @ V.T
    assert float(outside.norm()) < 1e-3 * float(noise.norm())            # all noise inside span(V)
    expected = sig * 0.5 * np.sqrt(k)                                     # ||N(0,(sigma C)^2 I_k)||
    assert torch.allclose(noise.norm(dim=1), expected, rtol=0.35)
    assert float(noise.norm(dim=1)[0]) < 0.5 * 1.0 * 0.5 * np.sqrt(d)    # far below the full-space noise norm

def test_projection_clips_in_subspace(nb):
    V = rand_basis(dim(nb), 16, seed=5)
    (c,) = clients(nb, n_clients=1, clip=0.01)
    c.basis = V
    captured = {}
    orig = c.ddqn.optimizer.step
    def spy():
        captured["g"] = torch.cat([p.grad.reshape(-1) for p in c.ddqn.online_net.parameters()]); orig()
    c.ddqn.optimizer.step = spy
    c._dp_step(0.0)
    assert float(captured["g"].norm()) <= c.n * 0.01 / (c.q * c.n) + 1e-6   # q = 1: every row sampled once
    assert float((captured["g"] - V @ (V.T @ captured["g"])).norm()) < 1e-5

def test_projection_does_not_change_accounting(nb):
    def eps(basis):
        (c,) = clients(nb, n_clients=1, clip=1.0)
        c.basis = basis
        for _ in range(4):
            c._dp_step(1.1)
        return c.epsilon_spent()
    assert eps(None) == pytest.approx(eps(rand_basis(dim(nb), 8)))

def test_noise_without_clipping_refused_with_basis(nb):
    (c,) = clients(nb, n_clients=1, clip=None)
    c.basis = rand_basis(dim(nb), 8)
    with pytest.raises(AssertionError, match="unbounded"):
        c._dp_step(1.0)


# ---------------- public basis ----------------
def test_public_basis_orthonormal_own_rng_and_full_space_option(nb, small_cfg):
    ag = nb["SmallDDQN"](nb["input_dim"], nb["action_dim"], nb["device"], nb["class_weights_dp"], gamma=0.0)
    ia, _ = nb["val_halves"]()
    XA, yA = nb["X_val"][ia], nb["y_val"][ia]
    torch.manual_seed(9); before = torch.rand(3)
    torch.manual_seed(9); V = nb["public_basis"](ag, XA, yA, 32, 0.5, 7); after = torch.rand(3)
    assert torch.equal(before, after)                                     # training RNG stream untouched
    assert V.shape == (dim(nb), 32)
    assert torch.allclose(V.T @ V, torch.eye(32), atol=1e-4)
    V2 = nb["public_basis"](ag, XA, yA, 32, 0.5, 7)
    assert torch.allclose(V.abs(), V2.abs(), atol=1e-5)                   # deterministic given the round seed
    assert nb["public_basis"](ag, XA, yA, 0, 0.5, 7) is None              # rank 0 = full space

def test_anchor_rows_are_class_balanced(nb):
    y = np.array([0] * 500 + [1] * 3 + [2] * 200)
    idx = nb["proj_anchor_indices"](y, 90, np.random.default_rng(0))
    assert len(idx) == 90 and len(np.unique(idx)) == 90
    assert (y[idx] == 1).sum() == 3                                       # every row of the rare class
    assert (y[idx] == 2).sum() >= 30


# ---------------- controller ----------------
def test_v4_policy_starts_at_fair_baseline_and_learns_rank(nb):
    torch.manual_seed(0)
    c = nb["ADPMetaV4"]((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (32, 64, 128, 0), 64)
    s = [0.0] * 10
    assert c.act(s, greedy=True) == (1.0, 1.0, 64)
    assert [m.out_features for m in c.trunk if isinstance(m, torch.nn.Linear)] == [32, 16]
    for _ in range(400):
        k, m, r = c.act(s)
        c.finish_episode((r == 32) * 1.0)
    assert c.act(s, greedy=True)[2] == 32


# ---------------- end-to-end ----------------
def test_fixed_proj_baseline_equals_fixed_noise_schedule(nb, small_cfg):
    small_cfg.update(CLIP_MULT=0.5, REWARD_FA=0.5, ADP_ACTIONS="pacing_clip_rank", PROJ_RANK=32, TAG="v4")
    out = nb["run_dp_pipeline"]("dp_fixed_proj", 0, 2.0, C=1.0)
    assert out["adp_version"] == "fixed-proj" and out["run"] == "dp_fixed_proj_eps2__v4"
    assert all(h["rank"] == 32 and h["clip_mult"] == 1.0 and h["action_k"] == 1.0 for h in out["history"])
    sig = [h["median_sigma"] for h in out["history"]]
    assert max(sig) == pytest.approx(min(sig), rel=1e-3)                  # constant noise = fixed-noise schedule
    assert all(h["epsilon"] <= 2.0 + 1e-9 for h in out["history"]) and out["final_epsilon"] >= 0.95 * 2.0

def test_v4_meta_training_and_real_run(nb, small_cfg, monkeypatch):
    small_cfg.update(CLIP_MULT=0.5, REWARD_FA=0.5, ADP_ACTIONS="pacing_clip_rank", ADP_META_EPISODES=4,
                     META_EPS=(2.0, 1.0), TWIN_POOL=1, PROXY_MIN_PER_CLIENT=5, RANK_ACTIONS=(16, 32, 0),
                     PROJ_RANK=32, PROJ_ANCHORS=128, TAG="v4")
    seen, basis_rows = [], []
    real_init = nb["DPClient"].__init__
    def spy(self, ddqn, X_local, y_local, *a, **k):
        seen.append((k.get("model_name", ""), np.asarray(X_local)))
        real_init(self, ddqn, X_local, y_local, *a, **k)
    monkeypatch.setattr(nb["DPClient"], "__init__", spy)
    real_basis = nb["public_basis"]
    def spy_basis(agent, X_pub, y_pub, *a):
        basis_rows.append(np.asarray(X_pub)); return real_basis(agent, X_pub, y_pub, *a)
    monkeypatch.setitem(nb, "public_basis", spy_basis)
    out = nb["run_dp_pipeline"]("dp_adpmeta", 0, 2.0, C=1.0)
    ia, ib = nb["val_halves"]()
    rows = lambda X: {r.tobytes() for r in X}
    A, B, private = rows(nb["X_val"][ia]), rows(nb["X_val"][ib]), rows(nb["X_dp"])
    proxies = [X for n, X in seen if n.startswith("Proxy")]
    reals = [X for n, X in seen if n.startswith("Client")]
    assert len(proxies) == (4 + 2) * nb["NUM_CLIENTS"]                   # 4 practice episodes + 2 cached twins
    assert all(rows(X) <= A for X in proxies)
    assert all(rows(X) <= private and not (rows(X) & (A | B)) for X in reals)   # detector never trains on val
    assert basis_rows and all(rows(X) <= A for X in basis_rows)            # subspace from public half A only
    assert out["adp_version"] == "v4-pacing-clip-rank" and out["adp_mode"] == "greedy"
    log = json.load(open(out["controller_file"].replace(".pt", "_meta_log.json")))
    assert [e["eps"] for e in log["episodes"]] == [2.0, 1.0, 2.0, 1.0]
    d = dim(nb)
    for h in out["history"]:
        assert h["epsilon"] <= 2.0 + 1e-9
        assert h["rank_action"] in (16, 32, 0) and h["rank"] == (d if h["rank_action"] == 0 else h["rank_action"])
        assert h["clip"] == pytest.approx(1.0 * h["clip_mult"])
    assert out["final_epsilon"] >= 0.95 * 2.0
    seen.clear()                                                          # shared controller reused
    out1 = nb["run_dp_pipeline"]("dp_adpmeta", 1, 1.0, C=1.0)
    assert not [n for n, _ in seen if n.startswith("Proxy")] and out1["final_epsilon"] <= 1.0 + 1e-9

def test_v3_and_v2_paths_untouched_by_v4(nb, small_cfg):
    small_cfg.update(ADP_ACTIONS="pacing", TAG="c5")                      # v2 online still runs, no basis used
    out = nb["run_dp_pipeline"]("dp_adpmeta", 0, 2.0, C=1.0)
    assert out["adp_version"] == "v2-pacing" and "rank" not in out["history"][0]


# ---------------- stages, summaries, static checks ----------------
def test_v4_stages(nb, monkeypatch):
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 18b "))
    assert 'STAGE = "v4_val"' in src
    saved = dict(nb["CFG"])
    def run(name):
        calls = []
        monkeypatch.setitem(nb, "run_all", lambda exps, seeds: calls.append((dict(nb["CFG"]), exps, seeds)))
        try:
            exec(re.sub(r'^STAGE = ".*"$', f'STAGE = "{name}"', src, flags=re.M), nb)
        finally:
            nb["CFG"].clear(); nb["CFG"].update(saved)
        return calls
    (val,) = run("v4_val")
    cfg, exps, seeds = val
    assert cfg["TAG"] == "v4" and cfg["ADP_ACTIONS"] == "pacing_clip_rank" and cfg["ADP_META_EPISODES"] == 24
    assert cfg["CLIP_MULT"] == 0.5 and cfg["REWARD_FA"] == 0.5 and cfg["MU"] == 0.01 and seeds == [0]
    assert exps == [("dp_fixed", 8.0), ("dp_fixed_proj", 8.0), ("dp_adpmeta", 8.0),
                    ("dp_fixed", 1.0), ("dp_fixed_proj", 1.0), ("dp_adpmeta", 1.0)]
    ((cfg_f, exps_f, seeds_f),) = run("final_v4")
    assert seeds_f == [0, 1, 2] and ("fedprox_np", None) in exps_f and ("dp_fixed_proj", 3.0) in exps_f
    assert run("v3_val")[0][0]["ADP_ACTIONS"] == "pacing_clip"            # v3 stage still available

def test_paired_tables_include_fair_baseline(nb, small_cfg, tmp_path, monkeypatch):
    small_cfg.update(CLIP_MULT=0.5, REWARD_FA=0.5, ADP_ACTIONS="pacing_clip_rank", ADP_META_EPISODES=1,
                     META_EPS=(2.0,), TWIN_POOL=1, PROXY_MIN_PER_CLIENT=5, RANK_ACTIONS=(16, 0), PROJ_RANK=16,
                     PROJ_ANCHORS=64, TAG="v4")
    nb["run_all"]([("dp_fixed", 2.0), ("dp_fixed_proj", 2.0), ("dp_adpmeta", 2.0)], [0])
    monkeypatch.chdir(tmp_path)
    exec(next(s for s in notebook_cells() if s.startswith("# BLOCK 18c ")), nb)
    assert set(nb["pdf_"]["run"]) == {"dp_adpmeta_eps2__v4", "dp_adpmeta_eps2__v4  vs fixed_proj"}
    b19 = next(s for s in notebook_cells() if s.startswith("# BLOCK 19 "))
    exec(b19.replace("FINAL_TEST = False", "FINAL_TEST = True"), nb)
    assert set(nb["paired"].index) == {"dp_adpmeta_eps2__v4", "dp_adpmeta_eps2__v4  vs fixed_proj"}

def test_old_results_stay_resumable_after_new_cfg_keys(nb, small_cfg):
    nb["run_all"]([("dp_fixed", 2.0)], [0])
    p = nb["result_path"]("dp_fixed_eps2", 0)
    r = json.load(open(p))
    for k in ("RANK_ACTIONS", "PROJ_RANK", "PROJ_ANCHORS"):              # simulate a run saved before v4
        r["cfg"].pop(k)
    json.dump(r, open(p, "w"))
    nb["run_all"]([("dp_fixed", 2.0)], [0])                              # skipped, not refused
    small_cfg["PROJ_RANK"] = 16
    with pytest.raises(ValueError, match="DIFFERENT config"):
        nb["run_all"]([("dp_fixed", 2.0)], [0])

def test_block17d_never_touches_test_set():
    src = next(s for s in notebook_cells() if s.startswith("# BLOCK 17d "))
    assert not re.search(r"\b(test_dataset|X_test|y_test)\b", src)

def test_unique_model_components_all_present_and_active():
    """DDQN + FedProx + record-level DP (noise really added) + ADP pacing/clip/rank + meta-learning all in the build."""
    cells = notebook_cells()
    joined = "\n".join(cells)
    for marker in ["# BLOCK 12 ", "# BLOCK 16 ", "# BLOCK 16b ", "# BLOCK 17 ", "# BLOCK 17b ", "# BLOCK 17c ",
                   "# BLOCK 17d ", "BLOCK 18a"]:
        assert marker in joined
    b16 = next(s for s in cells if s.startswith("# BLOCK 16 "))
    assert "p.grad = p.grad + torch.normal(0.0, sigma * self.C" in b16                 # full-space noise
    assert "e = e + torch.randn_like(e) * (sig * C)" in b16                            # subspace noise
    assert "self.mu * (p.detach() - self.global_params[k])" in b16                     # FedProx
    assert not re.search(r"#\s*\+\s*noise", joined)                                    # nothing commented out
    b17d = next(s for s in cells if s.startswith("# BLOCK 17d "))
    assert "class ADPMetaV4(ADPMetaV3)" in b17d and "finish_episode(adv)" in b17d       # meta-learning active
    assert "RDPPacer" in b17d and "pacing_share" in b17d                               # ADP pacing active
