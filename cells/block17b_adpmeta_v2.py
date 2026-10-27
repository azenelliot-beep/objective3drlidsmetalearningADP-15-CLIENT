# BLOCK 17b (v4 - C5/C6): ADP-Meta v2 - budget pacing + macro-F1 reward + meta-training on validation
# C5 (CFG["ADP_ACTIONS"] = "pacing"):
#   - Budget pacing in RDP at a fixed order alpha* per client. RDP at a fixed order ADDS UP over rounds, so
#     "spend a share of what is left" is exact. alpha* and the budget R come from the constant-noise plan
#     that spends exactly eps_budget over all rounds -> final Opacus epsilon <= eps_budget.
#   - Action k in PACE_ACTIONS: this round spends min(1, k / rounds_left) of the remaining RDP budget.
#     k = 1 every round reproduces fixed noise exactly; the last round always spends what is left.
#     sigma for the round is found by bisection, always rounding towards MORE noise.
#   - Policy network 32-16 (as in Chapter 6). State = [t/T, budget left, val macro-F1, d macro-F1,
#     log(1 + median client update norm)]. Reward r_t = validation macro-F1 gain of round t.
# C6 (CFG["ADP_META_EPISODES"] > 0):
#   - Practice episodes on SERVER-HELD VALIDATION data only (threat model): validation is split once into two
#     stratified halves (seed 42). Proxy detectors train on half A with the same DP-SGD client, batch size,
#     clip norm, local steps, rounds, FedProx mu and reward weights as the real run. The controller's reward
#     is measured on half B.
#   - "Same noise levels": proxy client k uses the noise schedule of REAL client k (its sampling rate q and
#     budget), so per-step noise sigma*C/B is identical to the real run for the same actions. The proxy
#     clients' own epsilon is meaningless (validation is public) and is never reported.
#   - REINFORCE with Monte-Carlo returns and a per-round baseline; one update per episode.
#   - The trained controller is frozen and used greedily in the real run; one file per eps budget,
#     shared by every seed (it is a fixed artifact, like a hyperparameter).
#   - The reported detector never trains on validation data; the test set is never used.
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from sklearn.model_selection import train_test_split

RDP_ORDERS = RDPAccountant.DEFAULT_ALPHAS


class RDPPacer:
    """Budget pacing for ONE client: RDP spent at a fixed order alpha* is additive across rounds."""
    def __init__(self, q, eps_budget, delta, rounds, steps):
        self.q, self.steps = float(q), int(steps)
        self.sigma0 = get_noise_multiplier(target_epsilon=eps_budget, target_delta=delta, sample_rate=self.q,
                                           steps=rounds * steps, accountant="rdp")
        rdp = compute_rdp(q=self.q, noise_multiplier=self.sigma0, steps=rounds * steps, orders=RDP_ORDERS)
        _, self.alpha = get_privacy_spent(orders=RDP_ORDERS, rdp=rdp, delta=delta)
        self.R = self.round_rdp(self.sigma0) * rounds              # RDP budget at alpha*
        self.spent = 0.0

    def round_rdp(self, sigma):
        return float(compute_rdp(q=self.q, noise_multiplier=sigma, steps=self.steps, orders=[self.alpha])[0])

    @property
    def left_frac(self):
        return max(0.0, 1.0 - self.spent / self.R)

    def sigma_for_share(self, share):
        """Smallest sigma (up to bisection precision, rounded towards more noise) whose round cost <= share * left."""
        target = share * (self.R - self.spent)
        if target <= 1e-12 * self.R:
            return None
        lo, hi = 1e-2, max(1.0, self.sigma0)
        while self.round_rdp(hi) > target:
            hi *= 2.0
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            if self.round_rdp(mid) > target:
                lo = mid
            else:
                hi = mid
        return hi

    def commit(self, sigma):
        self.spent += self.round_rdp(sigma)


def pacing_share(k, rounds_left):
    return 1.0 if rounds_left <= 1 else min(1.0, k / rounds_left)


class ADPMetaV2(nn.Module):
    STATE_DIM = 5

    def __init__(self, actions, lr=1e-2, entropy_coef=0.01, baseline_beta=0.8, device="cpu"):
        super().__init__()
        self.actions, self.device = list(actions), torch.device(device)
        self.policy = nn.Sequential(nn.Linear(self.STATE_DIM, 32), nn.Tanh(),
                                    nn.Linear(32, 16), nn.Tanh(),
                                    nn.Linear(16, len(self.actions))).to(self.device)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.entropy_coef, self.beta = entropy_coef, baseline_beta
        self.baseline = {}                   # per-round baseline of the return (episodic mode)
        self.online_baseline = 0.0           # moving baseline of the reward (online mode)
        self._logp, self._ent = [], []
        self.online_mode = False             # set by the runner: True -> update_online after every round

    @staticmethod
    def make_state(t, T, left_frac, f1, df1, upd_norm):
        return [t / T, left_frac, f1, df1, math.log1p(max(0.0, upd_norm))]

    def act(self, state, greedy=False):
        logits = self.policy(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        if greedy:
            with torch.no_grad():
                return int(logits.argmax())
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        self._logp.append(dist.log_prob(a)); self._ent.append(dist.entropy())
        return int(a)

    def finish_episode(self, rewards):
        """REINFORCE with Monte-Carlo returns G_t = sum_{s>=t} r_s and a per-round EMA baseline."""
        assert len(rewards) == len(self._logp), "one reward per sampled action"
        G = np.cumsum(np.asarray(rewards, dtype=np.float64)[::-1])[::-1]
        adv = [g - self.baseline.get(t, g) for t, g in enumerate(G)]       # first episode: advantage 0
        loss = -sum(a * lp for a, lp in zip(adv, self._logp)) - self.entropy_coef * sum(self._ent)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        for t, g in enumerate(G):
            self.baseline[t] = self.beta * self.baseline.get(t, g) + (1 - self.beta) * g
        self._logp, self._ent = [], []
        return float(G[0]) if len(G) else 0.0

    def update_online(self, reward):
        """Online variant (C5 without meta-training): one REINFORCE step per round."""
        adv = float(reward) - self.online_baseline
        loss = -adv * self._logp[-1] - self.entropy_coef * self._ent[-1]
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.online_baseline = self.beta * self.online_baseline + (1 - self.beta) * float(reward)
        self._logp, self._ent = [], []


def _drop_unrewarded(ctrl, rewards):
    """The round that is not run gets no reward -> forget the action just sampled for it (none if greedy)."""
    if len(ctrl._logp) > (len(rewards) if not ctrl.online_mode else 0):
        ctrl._logp.pop(); ctrl._ent.pop()


def paced_episode(name, g, clients, pacers, ctrl, eval_ds, *, rounds, steps, mu, rng, mode,
                  eps_budget=None, keep_states=False, fast=None):
    """One federated run whose noise each round is set by ADP-Meta v2 pacing.
    mode: "sample_episodic" (practice), "sample_online" (C5 real run), "greedy" (frozen, C6 real run).
    eps_budget: if given (real run), every client's Opacus epsilon is checked BEFORE spending and after."""
    assert mode in ("sample_episodic", "sample_online", "greedy")
    ctrl.online_mode = mode == "sample_online"
    history, states, rewards = [], [], []
    m = eval_metrics(g, eval_ds)
    f1_prev, df1_prev, upd = m["Macro_F1"], 0.0, 0.0
    for r in range(1, rounds + 1):
        left = float(np.mean([p.left_frac for p in pacers]))
        a = ctrl.act(ADPMetaV2.make_state(r, rounds, left, f1_prev, df1_prev, upd), greedy=(mode == "greedy"))
        k = ctrl.actions[a]
        share = pacing_share(k, rounds - r + 1)
        active = rng.choice(len(clients), size=max(1, math.ceil(CFG["CLIENT_FRAC"] * len(clients))), replace=False)
        sig = {int(i): pacers[i].sigma_for_share(share) for i in active}
        if any(s is None for s in sig.values()):
            print(f"[{name}] Round {r}: budget exhausted - stopping."); _drop_unrewarded(ctrl, rewards); break
        if eps_budget is not None and max(clients[i].projected_epsilon(sig[i], steps) for i in sig) > eps_budget + 1e-9:
            print(f"[{name}] Round {r}: privacy budget would be exceeded - stopping before spending.")
            _drop_unrewarded(ctrl, rewards); break
        gp = g.get_fl_parameters()
        params, sizes, mets = fit_clients(clients, list(sig), gp, sig, steps, mu, fast)
        for i, s in sig.items():
            pacers[i].commit(s)
        norms = [m["delta_l2"] for m in mets]
        g.set_fl_parameters(weighted_average(params, sizes))
        h = eval_metrics(g, eval_ds)
        reward = h["Macro_F1"] - f1_prev
        if mode == "sample_online":
            ctrl.update_online(reward)
        rewards.append(reward)
        eps_now = None
        if eps_budget is not None:
            eps_now = max(c.epsilon_spent() for c in clients)
            assert eps_now <= eps_budget + 1e-9, "DP guard: budget exceeded"
        h.update({"round": r, "action_k": k, "share": share, "median_sigma": float(np.median(list(sig.values()))),
                  "budget_left": float(np.mean([p.left_frac for p in pacers])), "reward": reward,
                  "epsilon": eps_now, "median_update_norm": float(np.median(norms))})
        history.append(h)
        if keep_states:
            states.append([t.clone() for t in g.get_fl_parameters()])
        _log_round(name, r, h, f" | k={k} share={share:.3f} sigma~{h['median_sigma']:.2f}"
                               + ("" if eps_now is None else f" | eps={eps_now:.3f}"))
        f1_prev, df1_prev, upd = h["Macro_F1"], reward, h["median_update_norm"]
    return history, states, rewards


def val_halves():
    """Fixed stratified split of the server-held validation set (seed 42): A = proxy training, B = reward."""
    ia, ib = train_test_split(np.arange(len(y_val)), test_size=0.5, stratify=y_val, random_state=SPLIT_SEED)
    return ia, ib


def controller_path(eps_budget):
    tag = f"__{CFG['TAG']}" if CFG.get("TAG") else ""
    return os.path.join(CFG["RESULTS_DIR"], f"controller_eps{eps_budget:g}{tag}.pt")


def meta_train_controller(eps_budget, C, mk_agent=None, client_cls=None, fast_factory=None):
    """C6: train ADP-Meta v2 over practice episodes on validation half A (reward on half B). Cached per eps.
    mk_agent / client_cls / fast_factory choose the proxy detector (default: SmallDDQN + DPClient + make_fast).
    A checkpoint is written after every episode, so an interrupted meta-training resumes where it stopped
    (episodes are seeded individually, so the resumed result is the same as an uninterrupted one)."""
    mk_agent = mk_agent or (lambda: SmallDDQN(input_dim, action_dim, device, reward_weights(), gamma=0.0, lr=CFG["LR"]))
    client_cls, fast_factory = client_cls or DPClient, fast_factory or make_fast
    path = controller_path(eps_budget)
    ctrl = ADPMetaV2(CFG["PACE_ACTIONS"], lr=CFG["ADP_LR"], entropy_coef=CFG["ADP_ENTROPY"], device=device)
    if os.path.exists(path):
        ctrl.load_state_dict(torch.load(path)); print(f"Loaded meta-trained controller {path}")
        return ctrl
    ia, ib = val_halves()
    XA, yA = X_val[ia], y_val[ia]
    val_B = TensorDataset(torch.from_numpy(X_val[ib]), torch.from_numpy(y_val[ib]).long())
    q_real = [min(1.0, CFG["BATCH"] / len(ci)) for ci in dp_client_indices]      # real clients, seed-42 split
    R, STEPS, mu = CFG["ROUNDS"], CFG["LOCAL_STEPS"], CFG["MU"]
    set_seed(SPLIT_SEED)
    ctrl = ADPMetaV2(CFG["PACE_ACTIONS"], lr=CFG["ADP_LR"], entropy_coef=CFG["ADP_ENTROPY"], device=device)
    log, t0, start = [], time.time(), 0
    ckpt = path.replace(".pt", "_partial.pt")
    if os.path.exists(ckpt):
        st = torch.load(ckpt, weights_only=False)
        ctrl.load_state_dict(st["policy"]); ctrl.opt.load_state_dict(st["opt"]); ctrl.baseline = st["baseline"]
        log, start = st["log"], st["episodes_done"]
        print(f"Resuming meta-training {path} after episode {start}")
    for e in range(start, CFG["ADP_META_EPISODES"]):
        es = 10_000 + e
        set_seed(es)
        parts = dirichlet_split(yA, NUM_CLIENTS, ALPHA, np.random.default_rng(es), CFG["PROXY_MIN_PER_CLIENT"])
        g = mk_agent()
        proxies = [client_cls(mk_agent(), XA[ci], yA[ci], device, model_name=f"Proxy{k+1}", batch_size=CFG["BATCH"],
                              clip_norm=C, delta=CFG["DELTA"]) for k, ci in enumerate(parts)]
        pacers = [RDPPacer(q_real[k], eps_budget, CFG["DELTA"], R, STEPS) for k in range(len(proxies))]
        hist, _, rewards = paced_episode(f"meta eps{eps_budget:g} ep{e+1}", g, proxies, pacers, ctrl, val_B,
                                         rounds=R, steps=STEPS, mu=mu, rng=np.random.default_rng(es),
                                         mode="sample_episodic", fast=fast_factory(proxies))
        ret = ctrl.finish_episode(rewards) if rewards else 0.0
        best = max((h["Macro_F1"] for h in hist), default=float("nan"))
        log.append({"episode": e + 1, "return": ret, "best_val_B_F1": best, "actions": [h["action_k"] for h in hist]})
        print(f"=== meta-episode {e+1}/{CFG['ADP_META_EPISODES']} | return={ret:+.4f} | best F1(B)={100*best:.1f} "
              f"| actions={log[-1]['actions']} | {(time.time()-t0)/60:.0f} min")
        os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
        torch.save({"policy": ctrl.state_dict(), "opt": ctrl.opt.state_dict(), "baseline": ctrl.baseline,
                    "log": log, "episodes_done": e + 1}, ckpt)
    os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
    torch.save(ctrl.state_dict(), path)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    with open(path.replace(".pt", "_meta_log.json"), "w") as f:
        json.dump({"eps_budget": eps_budget, "episodes": log, "minutes": (time.time() - t0) / 60,
                   "val_halves_seed": SPLIT_SEED, "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                                                          for k, v in CFG.items()}}, f, indent=2)
    return ctrl
