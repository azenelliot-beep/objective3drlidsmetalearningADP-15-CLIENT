# BLOCK 17d (v4 - ADP-Meta v4, Option 2): public-subspace gradient projection with a meta-learned rank
# Same DDQN detector, 15-client FedProx, record-level local DP-SGD, RDP accounting, budget pacing (17b), adaptive
# clipping (17c) and meta-training as before. ADDED: noise is applied inside a low-dimensional public subspace.
#  (1) Public basis (GEP / PDP-SGD style): every round the server takes the CURRENT global DDQN, computes per-sample
#      TD gradients on PROJ_ANCHORS class-balanced PUBLIC rows (validation half A) and keeps the top-k right singular
#      vectors V (d x k). V never touches private training data, so it costs no privacy (post-processing of the
#      already-private global model + public data). Clients clip V^T g_i to C and add N(0, (sigma C)^2 I_k): noise
#      norm sigma*C*sqrt(k) instead of sigma*C*sqrt(d) (d = all DDQN parameters).
#  (2) ADP-Meta v4 controller: each round picks THREE actions - budget pacing k (17b), clip multiplier m (17c) and
#      subspace rank r in RANK_ACTIONS (0 = full space, i.e. the v3 behaviour). Policy 32-16 trunk, three heads,
#      initial bias towards the FAIR fixed baseline (k = 1, m = 1, r = PROJ_RANK). State = v3 state + log rank share.
#  (3) Meta-training exactly as v3 (twin-baseline REINFORCE on validation half A, reward on half B, one controller for
#      every eps). The twin is the fair fixed baseline: fixed noise + fixed rank PROJ_RANK.
#  (4) Fair baseline method "dp_fixed_proj": fixed noise (k = 1 every round == constant sigma) + the same projection
#      with constant rank PROJ_RANK and constant clip. ADP-Meta v4 is compared against it AND against "dp_fixed".
# THREAT MODEL (amended for this block, disclose in the thesis): validation half A is server-held public data and may
# now also shape the gradient subspace. It is never used as training records for the reported detector (no gradient
# step of the detector is taken on it); the test set is never used.

def proj_anchor_indices(y, m, rng):
    """Class-balanced anchor rows: up to m / n_classes per class (all rows of a rare class), rest filled at random."""
    classes = np.unique(y)
    per = max(1, m // len(classes))
    take = []
    for c in classes:
        idx = np.where(y == c)[0]
        take.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
    rest = np.setdiff1d(np.arange(len(y)), take)
    extra = min(len(rest), max(0, m - len(take)))
    if extra:
        take.extend(rng.choice(rest, size=extra, replace=False).tolist())
    return np.array(sorted(take))


def public_basis(agent, X_pub, y_pub, rank, explore_eps, round_seed):
    """Top-`rank` right singular vectors of per-sample TD gradients of `agent` on public rows. None = full space.
    Uses its own RNGs, so the training random streams (and v3/v4 twin alignment) are not shifted."""
    net = agent.online_net
    d = sum(p.numel() for p in net.parameters())
    if not rank or rank >= d:
        return None
    rng = np.random.default_rng(round_seed)
    idx = proj_anchor_indices(y_pub, CFG["PROJ_ANCHORS"], rng)
    x = torch.as_tensor(X_pub[idx], dtype=torch.float32, device=device)
    yy = torch.as_tensor(y_pub[idx], dtype=torch.long, device=device)
    gen = torch.Generator(device=x.device).manual_seed(int(round_seed))
    with torch.no_grad():                                   # epsilon-greedy actions as in training, own generator
        net.eval(); greedy = net(x).argmax(1); net.train()
        explore = torch.rand(greedy.shape, generator=gen, device=x.device) < explore_eps
        rand = torch.randint(0, action_dim, greedy.shape, generator=gen, device=x.device)
        a = torch.where(explore, rand, greedy)
        r = agent.reward(a, yy)
    params = {k: v.detach() for k, v in net.named_parameters()}
    buffers = {k: v.detach() for k, v in net.named_buffers()}
    def loss_fn(p, b, xi, ai, ri):
        qv = functional_call(net, (p, b), (xi.unsqueeze(0),)).squeeze(0)
        return F.smooth_l1_loss(qv.gather(0, ai.unsqueeze(0)).squeeze(0), ri)
    g = vmap(grad(loss_fn), in_dims=(None, None, 0, 0, 0))(params, buffers, x, a, r)
    G = torch.cat([g[k].reshape(len(idx), -1) for k in params], 1)          # [M, d], parameter order
    _, _, Vh = torch.linalg.svd(G, full_matrices=False)
    k = min(int(rank), Vh.shape[0])
    return Vh[:k].T.contiguous()                                            # [d, k], orthonormal columns


class ADPMetaV4(ADPMetaV3):
    STATE_DIM = 10

    def __init__(self, pace_actions, clip_actions, rank_actions, fixed_rank, lr=1e-2, entropy_coef=0.01,
                 baseline_beta=0.8, init_bias=1.0):
        super().__init__(pace_actions, clip_actions, lr=lr, entropy_coef=entropy_coef,
                         baseline_beta=baseline_beta, init_bias=init_bias)
        self.rank = list(rank_actions)
        assert fixed_rank in self.rank, "PROJ_RANK must be one of RANK_ACTIONS"
        self.head_r = nn.Linear(16, len(self.rank))
        with torch.no_grad():                               # start at the fair fixed baseline (k=1, m=1, r=PROJ_RANK)
            self.head_r.bias[self.rank.index(fixed_rank)] += init_bias
        self.opt = optim.Adam(self.parameters(), lr=lr)     # rebuilt so the rank head is trained too

    @staticmethod
    def make_state(t, T, left, f1, df1, far, upd, prev_sigma, prev_m, eps_budget, prev_rank_share):
        return ADPMetaV3.make_state(t, T, left, f1, df1, far, upd, prev_sigma, prev_m, eps_budget) + \
               [math.log(max(prev_rank_share, 1e-3))]

    def _dists(self, state):
        z = self.trunk(torch.as_tensor(state, dtype=torch.float32))
        return (torch.softmax(self.head_k(z), -1), torch.softmax(self.head_m(z), -1),
                torch.softmax(self.head_r(z), -1))

    def act(self, state, greedy=False):
        pk, pm, pr = self._dists(state)
        if greedy:
            return self.pace[int(pk.argmax())], self.clip[int(pm.argmax())], self.rank[int(pr.argmax())]
        ik = int(torch.multinomial(pk.detach(), 1, generator=self.gen))
        im = int(torch.multinomial(pm.detach(), 1, generator=self.gen))
        ir = int(torch.multinomial(pr.detach(), 1, generator=self.gen))
        self._logp.append(torch.log(pk[ik]) + torch.log(pm[im]) + torch.log(pr[ir]))
        self._ent.append(-sum((p * torch.log(p + 1e-12)).sum() for p in (pk, pm, pr)))
        return self.pace[ik], self.clip[im], self.rank[ir]


def fixed_proj_policy(state):
    return 1.0, 1.0, CFG["PROJ_RANK"]


def _explore_eps(clients, fast):
    return fast._epsilon() if fast is not None else clients[0].ddqn.epsilon


def paced_clip_rank_episode(name, g, clients, pacers, policy, eval_ds, X_pub, y_pub, *, rounds, steps, mu, rng,
                            C_base, eps_budget, fast=None, check_budget=False, keep_states=False, basis_seed=0):
    """One federated run; each round policy(state) -> (k, m, r): noise share, clip C_base * m, subspace rank r."""
    history, states = [], []
    d = sum(p.numel() for p in g.online_net.parameters())
    m0 = eval_metrics(g, eval_ds)
    f1_prev, df1_prev, far_prev, upd, sig_prev, m_prev = m0["Macro_F1"], 0.0, m0["FAR"], 0.0, 1.0, 1.0
    rshare_prev = 1.0
    for r in range(1, rounds + 1):
        left = float(np.mean([p.left_frac for p in pacers]))
        k, m, rank = policy(ADPMetaV4.make_state(r, rounds, left, f1_prev, df1_prev, far_prev, upd, sig_prev, m_prev,
                                                 eps_budget, rshare_prev))
        share = pacing_share(k, rounds - r + 1)
        active = rng.choice(len(clients), size=max(1, math.ceil(CFG["CLIENT_FRAC"] * len(clients))), replace=False)
        sig = {int(i): pacers[i].sigma_for_share(share) for i in active}
        if any(s is None for s in sig.values()):
            print(f"[{name}] Round {r}: budget exhausted - stopping."); break
        if check_budget and max(clients[i].projected_epsilon(sig[i], steps) for i in sig) > eps_budget + 1e-9:
            print(f"[{name}] Round {r}: privacy budget would be exceeded - stopping before spending."); break
        C_t = C_base * m
        V = public_basis(g, X_pub, y_pub, rank, _explore_eps(clients, fast), basis_seed * 1000 + r)
        for c in clients:
            c.C, c.basis = C_t, V                               # clip norm + public subspace for this round
        if fast is not None:
            fast.C, fast.basis = C_t, V
        gp = g.get_fl_parameters()
        params, sizes, mets = fit_clients(clients, list(sig), gp, sig, steps, mu, fast)
        for i, s in sig.items():
            pacers[i].commit(s)
        g.set_fl_parameters(weighted_average(params, sizes))
        h = eval_metrics(g, eval_ds)
        eps_now = None
        if check_budget:
            eps_now = max(c.epsilon_spent() for c in clients)
            assert eps_now <= eps_budget + 1e-9, "DP guard: budget exceeded"
        rank_used = d if V is None else int(V.shape[1])
        h.update({"round": r, "action_k": k, "clip_mult": m, "clip": C_t, "rank": rank_used, "rank_action": rank,
                  "share": share, "median_sigma": float(np.median(list(sig.values()))), "epsilon": eps_now,
                  "budget_left": float(np.mean([p.left_frac for p in pacers])),
                  "median_update_norm": float(np.median([x["delta_l2"] for x in mets]))})
        history.append(h)
        if keep_states:
            states.append([t.clone() for t in g.get_fl_parameters()])
        _log_round(name, r, h, f" | k={k} m={m} C={C_t:.2f} rank={rank_used}/{d} sigma~{h['median_sigma']:.2f}"
                               + ("" if eps_now is None else f" | eps={eps_now:.3f}"))
        f1_prev, df1_prev, far_prev = h["Macro_F1"], h["Macro_F1"] - f1_prev, h["FAR"]
        upd, sig_prev, m_prev, rshare_prev = h["median_update_norm"], h["median_sigma"], m, rank_used / d
    for c in clients:
        c.basis = None                                          # leave clients in the default full-space state
    if fast is not None:
        fast.basis = None
    return history, states


def controller_v4_path():
    tag = f"__{CFG['TAG']}" if CFG.get("TAG") else ""
    return os.path.join(CFG["RESULTS_DIR"], f"controller_v4_shared{tag}.pt")


def _new_v4_controller():
    return ADPMetaV4(CFG["PACE_ACTIONS"], CFG["CLIP_ACTIONS"], CFG["RANK_ACTIONS"], CFG["PROJ_RANK"],
                     lr=CFG["ADP_LR"], entropy_coef=CFG["ADP_ENTROPY"])


def _practice_episode_v4(ep_seed, eps_budget, C, policy, XA, yA, val_B, q_real, label):
    set_seed(ep_seed)
    parts = dirichlet_split(yA, NUM_CLIENTS, ALPHA, np.random.default_rng(ep_seed), CFG["PROXY_MIN_PER_CLIENT"])
    w = reward_weights()
    mk = lambda: SmallDDQN(input_dim, action_dim, device, w, gamma=0.0, lr=CFG["LR"])
    g = mk()
    maybe_public_init(g, SPLIT_SEED)                               # Block 17f: no-op unless PUBLIC_FRAC > 0
    proxies = [DPClient(mk(), XA[ci], yA[ci], device, model_name=f"Proxy{k+1}", batch_size=CFG["BATCH"],
                        clip_norm=C, delta=CFG["DELTA"]) for k, ci in enumerate(parts)]
    R, STEPS = CFG["ROUNDS"], CFG["LOCAL_STEPS"]
    pacers = [RDPPacer(q_real[k], eps_budget, CFG["DELTA"], R, STEPS) for k in range(len(proxies))]
    hist, _ = paced_clip_rank_episode(label, g, proxies, pacers, policy, val_B, XA, yA, rounds=R, steps=STEPS,
                                      mu=CFG["MU"], rng=np.random.default_rng(ep_seed), C_base=C,
                                      eps_budget=eps_budget, fast=make_fast(proxies), basis_seed=ep_seed)
    return episode_score(hist), hist


def meta_train_controller_v4(C):
    """Meta-train ONE ADP-Meta v4 controller for all eps in CFG['META_EPS'] (cached). Twin = fair fixed baseline.
    Checkpointed after every episode (resumes after a Kaggle timeout). CFG['META_TIME_LIMIT_MIN'] (None = off): no new
    episode is started if it would not finish within the limit; the number of episodes actually run is logged."""
    path = controller_v4_path()
    if os.path.exists(path):
        ctrl = _new_v4_controller()
        ctrl.load_state_dict(torch.load(path)); print(f"Loaded meta-trained controller {path}")
        return ctrl
    torch.manual_seed(SPLIT_SEED)
    ctrl = _new_v4_controller()
    ia, ib = val_halves()
    XA, yA = X_val[ia], y_val[ia]
    val_B = TensorDataset(torch.from_numpy(X_val[ib]), torch.from_numpy(y_val[ib]).long())
    q_real = [min(1.0, CFG["BATCH"] / len(ci)) for ci in dp_client_indices]
    pool = [30_000 + i for i in range(CFG["TWIN_POOL"])]
    twins, log, done_min = {}, [], 0.0
    ckpt = path.replace(".pt", "_ckpt.pt")
    if os.path.exists(ckpt):                                  # resume after an interrupted session
        ck = torch.load(ckpt, weights_only=False)
        ctrl.load_state_dict(ck["ctrl"]); ctrl.opt.load_state_dict(ck["opt"])
        ctrl.baseline = ck["baseline"]; ctrl.gen.set_state(ck["gen"])
        twins = {(es, eps): sc for es, eps, sc in ck["twins"]}
        log, done_min = ck["log"], ck["minutes"]
        print(f"Resuming v4 meta-training from checkpoint: {len(log)} episodes done, {done_min:.0f} min used")
    t0 = time.time() - 60 * done_min
    limit = CFG.get("META_TIME_LIMIT_MIN")
    stopped_early = False
    for e in range(len(log), CFG["ADP_META_EPISODES"]):
        used = (time.time() - t0) / 60
        per_ep = used / len(log) if log else 0.0              # measured minutes per episode (incl. its twin)
        if limit is not None and log and used + per_ep > limit:
            print(f"=== v4 meta-training stopped by META_TIME_LIMIT_MIN={limit}: {len(log)} of "
                  f"{CFG['ADP_META_EPISODES']} episodes in {used:.0f} min")
            stopped_early = True
            break
        eps_b = CFG["META_EPS"][e % len(CFG["META_EPS"])]
        es = pool[(e // len(CFG["META_EPS"])) % len(pool)]
        if (es, eps_b) not in twins:
            twins[(es, eps_b)] = _practice_episode_v4(es, eps_b, C, fixed_proj_policy, XA, yA, val_B, q_real,
                                                      f"twin s{es} eps{eps_b:g}")[0]
        score, hist = _practice_episode_v4(es, eps_b, C, lambda s: ctrl.act(s), XA, yA, val_B, q_real,
                                           f"meta ep{e+1} eps{eps_b:g}")
        adv = score - twins[(es, eps_b)]
        ctrl.finish_episode(adv)
        log.append({"episode": e + 1, "eps": eps_b, "seed": es, "score": score, "twin": twins[(es, eps_b)],
                    "advantage": adv, "actions": [(h["action_k"], h["clip_mult"], h["rank_action"]) for h in hist]})
        print(f"=== v4 meta-episode {e+1}/{CFG['ADP_META_EPISODES']} eps={eps_b:g} | score={100*score:.1f} "
              f"twin={100*twins[(es, eps_b)]:.1f} adv={100*adv:+.1f} | {(time.time()-t0)/60:.0f} min")
        os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
        torch.save({"ctrl": ctrl.state_dict(), "opt": ctrl.opt.state_dict(), "baseline": ctrl.baseline,
                    "gen": ctrl.gen.get_state(), "twins": [[es_, eps_, sc] for (es_, eps_), sc in twins.items()],
                    "log": log, "minutes": (time.time() - t0) / 60}, ckpt)
    os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
    torch.save(ctrl.state_dict(), path)
    with open(path.replace(".pt", "_meta_log.json"), "w") as f:
        json.dump({"episodes": log, "episodes_run": len(log), "stopped_by_time_limit": stopped_early,
                   "minutes": (time.time() - t0) / 60, "val_halves_seed": SPLIT_SEED, "cfg": cfg_json()}, f, indent=2)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    return ctrl


def _run_v4(name, method, seed, eps_budget, C, g, clients, parts, rng, mu, fast=None, adaptive=True):
    """dp_adpmeta with ADP_ACTIONS = 'pacing_clip_rank' (adaptive=True) or the fair baseline dp_fixed_proj."""
    if adaptive:
        ctrl = meta_train_controller_v4(C)
        policy = lambda s: ctrl.act(s, greedy=True)
    else:
        policy = fixed_proj_policy
    set_seed(seed); rng = np.random.default_rng(seed)        # meta-training must not shift the run's RNG
    ia, _ = val_halves()
    R, STEPS, delta = CFG["ROUNDS"], CFG["LOCAL_STEPS"], CFG["DELTA"]
    pacers = [RDPPacer(c.q, eps_budget, delta, R, STEPS) for c in clients]
    print(f"[{name} | seed {seed}] {'ADP-Meta v4 (frozen, greedy)' if adaptive else 'fixed noise + fixed projection'}"
          f" | C_base={C:.3f} | mu={mu} | rank actions={CFG['RANK_ACTIONS']} fixed rank={CFG['PROJ_RANK']} | "
          f"sigma0 median={np.median([p.sigma0 for p in pacers]):.3f}")
    t0 = time.time()
    history, states = paced_clip_rank_episode(name, g, clients, pacers, policy, val_dataset, X_val[ia], y_val[ia],
                                              rounds=R, steps=STEPS, mu=mu, rng=rng, C_base=C, eps_budget=eps_budget,
                                              fast=fast, check_budget=True, keep_states=True, basis_seed=seed)
    return _save(name, seed, history, states, "small", {
        "method": method, "private": True, "eps_budget": eps_budget,
        "final_epsilon": history[-1]["epsilon"] if history else 0.0, "delta": delta, "clip_norm": C,
        "mu": mu, "adp_version": "v4-pacing-clip-rank" if adaptive else "fixed-proj",
        "adp_mode": "greedy" if adaptive else "fixed",
        "controller_file": controller_v4_path() if adaptive else None,
        "public_basis_rows": "validation half A", "proj_rank_fixed": CFG["PROJ_RANK"],
        "sigma0_per_client": [p.sigma0 for p in pacers], "rounds_run": len(history),
        "minutes": (time.time() - t0) / 60})
