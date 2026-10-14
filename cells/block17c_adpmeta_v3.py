# BLOCK 17c (v4 - ADP-Meta v3): joint noise pacing + adaptive clipping, twin-baseline meta-training
# Same DDQN, FedProx, record-level local DP-SGD and RDP accounting as before. What changes is the controller:
#  (1) Each round it picks BOTH a budget-pacing action k (share of the remaining RDP budget, Block 17b) AND a clip
#      multiplier m: C_t = C_base * m. Privacy depends only on the noise multiplier sigma (noise std = sigma * C_t),
#      so changing C costs no privacy. The decision uses only validation metrics and already-noised updates.
#  (2) Meta-training is paired: each practice episode has a TWIN that uses the fixed schedule (k = 1, m = 1) with
#      the same proxy data, initialisation and sampling seed. The controller is rewarded for beating its twin.
#      Twins are cached per (seed, eps), so 6 twin runs serve all practice episodes.
#  (3) Reward = the episode's selected-round score under the SAME rule used for model selection
#      (best val macro-F1 with FAR <= cap; if the cap is never met, F1 minus the FAR excess).
#  (4) Richer state: t/T, budget left, macro-F1, d macro-F1, FAR, log(1 + update norm), previous log sigma,
#      previous log clip multiplier, log10(eps budget).
#  (5) More practice episodes (cheap with Block 16b). ONE controller for all eps budgets (eps is in the state).
# The policy is 32-16 with two heads. Its initial bias favours (k = 1, m = 1), i.e. it starts at the fixed
# schedule and explores around it. The real run uses the frozen policy greedily.
# Practice data = validation half A (proxy clients follow the real clients' noise schedule), reward on half B.
# The reported detector never trains on validation data, and the test set is never used.


class ADPMetaV3(nn.Module):
    STATE_DIM = 9

    def __init__(self, pace_actions, clip_actions, lr=1e-2, entropy_coef=0.01, baseline_beta=0.8, init_bias=1.0):
        super().__init__()
        self.pace, self.clip = list(pace_actions), list(clip_actions)
        self.trunk = nn.Sequential(nn.Linear(self.STATE_DIM, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh())
        self.head_k, self.head_m = nn.Linear(16, len(self.pace)), nn.Linear(16, len(self.clip))
        with torch.no_grad():                                   # start at the fixed schedule (k = 1, m = 1)
            if 1.0 in self.pace:
                self.head_k.bias[self.pace.index(1.0)] += init_bias
            if 1.0 in self.clip:
                self.head_m.bias[self.clip.index(1.0)] += init_bias
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.entropy_coef, self.beta, self.baseline = entropy_coef, baseline_beta, None   # set by 1st episode
        self.gen = torch.Generator().manual_seed(SPLIT_SEED)    # own RNG: does not shift the training RNG streams
        self._logp, self._ent = [], []

    @staticmethod
    def make_state(t, T, left, f1, df1, far, upd, prev_sigma, prev_m, eps_budget):
        return [t / T, left, f1, df1, far, math.log1p(max(0.0, upd)), math.log(max(prev_sigma, 1e-3)),
                math.log(prev_m), math.log10(eps_budget)]

    def _dists(self, state):
        z = self.trunk(torch.as_tensor(state, dtype=torch.float32))
        return torch.softmax(self.head_k(z), -1), torch.softmax(self.head_m(z), -1)

    def act(self, state, greedy=False):
        pk, pm = self._dists(state)
        if greedy:
            return self.pace[int(pk.argmax())], self.clip[int(pm.argmax())]
        ik = int(torch.multinomial(pk.detach(), 1, generator=self.gen))
        im = int(torch.multinomial(pm.detach(), 1, generator=self.gen))
        self._logp.append(torch.log(pk[ik]) + torch.log(pm[im]))
        self._ent.append(-(pk * torch.log(pk + 1e-12)).sum() - (pm * torch.log(pm + 1e-12)).sum())
        return self.pace[ik], self.clip[im]

    def finish_episode(self, advantage):
        """Terminal reward: every decision of the episode shares the advantage over the twin."""
        if not self._logp:
            return
        if self.baseline is None:
            self.baseline = float(advantage)                    # first episode: no update signal, sets the baseline
        a = float(advantage) - self.baseline
        loss = -a * torch.stack(self._logp).sum() - self.entropy_coef * torch.stack(self._ent).sum()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.baseline = self.beta * self.baseline + (1 - self.beta) * float(advantage)
        self._logp, self._ent = [], []


def episode_score(history):
    """Score of an episode under the model-selection rule (same rule as the reported runs)."""
    if not history:
        return -1.0
    i, ok = select_best_round(history, CFG["FAR_CAP"])
    h = history[i]
    return h["Macro_F1"] - (0.0 if ok else (h["FAR"] - CFG["FAR_CAP"]))


def paced_clip_episode(name, g, clients, pacers, policy, eval_ds, *, rounds, steps, mu, rng, C_base, eps_budget,
                       fast=None, check_budget=False, keep_states=False):
    """One federated run; each round policy(state) -> (k, m) sets the noise share and the clip norm C_base * m."""
    history, states = [], []
    m0 = eval_metrics(g, eval_ds)
    f1_prev, df1_prev, far_prev, upd, sig_prev, m_prev = m0["Macro_F1"], 0.0, m0["FAR"], 0.0, 1.0, 1.0
    for r in range(1, rounds + 1):
        left = float(np.mean([p.left_frac for p in pacers]))
        k, m = policy(ADPMetaV3.make_state(r, rounds, left, f1_prev, df1_prev, far_prev, upd, sig_prev, m_prev,
                                           eps_budget))
        share = pacing_share(k, rounds - r + 1)
        active = rng.choice(len(clients), size=max(1, math.ceil(CFG["CLIENT_FRAC"] * len(clients))), replace=False)
        sig = {int(i): pacers[i].sigma_for_share(share) for i in active}
        if any(s is None for s in sig.values()):
            print(f"[{name}] Round {r}: budget exhausted - stopping."); break
        if check_budget and max(clients[i].projected_epsilon(sig[i], steps) for i in sig) > eps_budget + 1e-9:
            print(f"[{name}] Round {r}: privacy budget would be exceeded - stopping before spending."); break
        C_t = C_base * m
        for c in clients:
            c.C = C_t                                           # clip norm for this round (sigma * C_t noise)
        if fast is not None:
            fast.C = C_t
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
        h.update({"round": r, "action_k": k, "clip_mult": m, "clip": C_t, "share": share,
                  "median_sigma": float(np.median(list(sig.values()))), "epsilon": eps_now,
                  "budget_left": float(np.mean([p.left_frac for p in pacers])),
                  "median_update_norm": float(np.median([x["delta_l2"] for x in mets]))})
        history.append(h)
        if keep_states:
            states.append([t.clone() for t in g.get_fl_parameters()])
        _log_round(name, r, h, f" | k={k} m={m} C={C_t:.2f} sigma~{h['median_sigma']:.2f}"
                               + ("" if eps_now is None else f" | eps={eps_now:.3f}"))
        f1_prev, df1_prev, far_prev = h["Macro_F1"], h["Macro_F1"] - f1_prev, h["FAR"]
        upd, sig_prev, m_prev = h["median_update_norm"], h["median_sigma"], m
    return history, states


def controller_v3_path():
    tag = f"__{CFG['TAG']}" if CFG.get("TAG") else ""
    return os.path.join(CFG["RESULTS_DIR"], f"controller_v3_shared{tag}.pt")


def _practice_episode(ep_seed, eps_budget, C, policy, XA, yA, val_B, q_real, label):
    set_seed(ep_seed)
    parts = dirichlet_split(yA, NUM_CLIENTS, ALPHA, np.random.default_rng(ep_seed), CFG["PROXY_MIN_PER_CLIENT"])
    w = reward_weights()
    mk = lambda: SmallDDQN(input_dim, action_dim, device, w, gamma=0.0, lr=CFG["LR"])
    g = mk()
    proxies = [DPClient(mk(), XA[ci], yA[ci], device, model_name=f"Proxy{k+1}", batch_size=CFG["BATCH"],
                        clip_norm=C, delta=CFG["DELTA"]) for k, ci in enumerate(parts)]
    R, STEPS = CFG["ROUNDS"], CFG["LOCAL_STEPS"]
    pacers = [RDPPacer(q_real[k], eps_budget, CFG["DELTA"], R, STEPS) for k in range(len(proxies))]
    hist, _ = paced_clip_episode(label, g, proxies, pacers, policy, val_B, rounds=R, steps=STEPS, mu=CFG["MU"],
                                 rng=np.random.default_rng(ep_seed), C_base=C, eps_budget=eps_budget,
                                 fast=make_fast(proxies))
    return episode_score(hist), hist


def meta_train_controller_v3(C):
    """Meta-train ONE ADP-Meta v3 controller for all eps in CFG['META_EPS'] (cached)."""
    path = controller_v3_path()
    ctrl = ADPMetaV3(CFG["PACE_ACTIONS"], CFG["CLIP_ACTIONS"], lr=CFG["ADP_LR"], entropy_coef=CFG["ADP_ENTROPY"])
    if os.path.exists(path):
        ctrl.load_state_dict(torch.load(path)); print(f"Loaded meta-trained controller {path}")
        return ctrl
    torch.manual_seed(SPLIT_SEED)
    ctrl = ADPMetaV3(CFG["PACE_ACTIONS"], CFG["CLIP_ACTIONS"], lr=CFG["ADP_LR"], entropy_coef=CFG["ADP_ENTROPY"])
    ia, ib = val_halves()
    XA, yA = X_val[ia], y_val[ia]
    val_B = TensorDataset(torch.from_numpy(X_val[ib]), torch.from_numpy(y_val[ib]).long())
    q_real = [min(1.0, CFG["BATCH"] / len(ci)) for ci in dp_client_indices]
    pool = [20_000 + i for i in range(CFG["TWIN_POOL"])]
    twins, log, t0 = {}, [], time.time()
    fixed = lambda s: (1.0, 1.0)
    limit, stopped_early = CFG.get("META_TIME_LIMIT_MIN"), False     # None (default) = no limit, behaviour unchanged
    for e in range(CFG["ADP_META_EPISODES"]):
        used = (time.time() - t0) / 60
        if limit is not None and log and used + used / len(log) > limit:
            print(f"=== v3 meta-training stopped by META_TIME_LIMIT_MIN={limit}: {len(log)} episodes in {used:.0f} min")
            stopped_early = True
            break
        eps_b = CFG["META_EPS"][e % len(CFG["META_EPS"])]
        es = pool[(e // len(CFG["META_EPS"])) % len(pool)]
        if (es, eps_b) not in twins:
            twins[(es, eps_b)] = _practice_episode(es, eps_b, C, fixed, XA, yA, val_B, q_real,
                                                   f"twin s{es} eps{eps_b:g}")[0]
        score, hist = _practice_episode(es, eps_b, C, lambda s: ctrl.act(s), XA, yA, val_B, q_real,
                                        f"meta ep{e+1} eps{eps_b:g}")
        adv = score - twins[(es, eps_b)]
        ctrl.finish_episode(adv)
        log.append({"episode": e + 1, "eps": eps_b, "seed": es, "score": score, "twin": twins[(es, eps_b)],
                    "advantage": adv, "actions": [(h["action_k"], h["clip_mult"]) for h in hist]})
        print(f"=== meta-episode {e+1}/{CFG['ADP_META_EPISODES']} eps={eps_b:g} | score={100*score:.1f} "
              f"twin={100*twins[(es, eps_b)]:.1f} adv={100*adv:+.1f} | {(time.time()-t0)/60:.0f} min")
    os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
    torch.save(ctrl.state_dict(), path)
    with open(path.replace(".pt", "_meta_log.json"), "w") as f:
        json.dump({"episodes": log, "episodes_run": len(log), "stopped_by_time_limit": stopped_early,
                   "minutes": (time.time() - t0) / 60, "val_halves_seed": SPLIT_SEED,
                   "cfg": cfg_json()}, f, indent=2)
    return ctrl


def _run_paced_clip(name, method, seed, eps_budget, C, g, clients, parts, rng, mu, fast=None):
    ctrl = meta_train_controller_v3(C)
    set_seed(seed); rng = np.random.default_rng(seed)        # meta-training must not shift the run's RNG
    R, STEPS, delta = CFG["ROUNDS"], CFG["LOCAL_STEPS"], CFG["DELTA"]
    pacers = [RDPPacer(c.q, eps_budget, delta, R, STEPS) for c in clients]
    print(f"[{name} | seed {seed}] ADP-Meta v3 (frozen, greedy) | C_base={C:.3f} | mu={mu} | "
          f"sigma0 median={np.median([p.sigma0 for p in pacers]):.3f}")
    t0 = time.time()
    history, states = paced_clip_episode(name, g, clients, pacers, lambda s: ctrl.act(s, greedy=True), val_dataset,
                                         rounds=R, steps=STEPS, mu=mu, rng=rng, C_base=C, eps_budget=eps_budget,
                                         fast=fast, check_budget=True, keep_states=True)
    return _save(name, seed, history, states, "small", {
        "method": method, "private": True, "eps_budget": eps_budget,
        "final_epsilon": history[-1]["epsilon"] if history else 0.0, "delta": delta, "clip_norm": C,
        "mu": mu, "adp_version": "v3-pacing-clip", "adp_mode": "greedy", "controller_file": controller_v3_path(),
        "sigma0_per_client": [p.sigma0 for p in pacers], "rounds_run": len(history),
        "minutes": (time.time() - t0) / 60})
