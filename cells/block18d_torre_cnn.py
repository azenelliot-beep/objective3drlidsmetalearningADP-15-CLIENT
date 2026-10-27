# BLOCK 18d (NEW - comparison): Torre et al. (TOSEM 2025) 1D-CNN detector inside the Chapter 6 pipeline
# Purpose: compare DETECTORS under identical conditions. Everything except the detector and its loss is the
# Chapter 6 protocol: same leakage-free data (Blocks 5-8b), same 15 Dirichlet clients (alpha 2, run seed),
# same 20 rounds x 400 local DP-SGD steps, batch 256, Adam lr 1e-3, FedProx mu (CFG), per-sample clipping
# with C = CLIP_MULT x median per-sample grad norm on VALIDATION (rule of dp_clip_norm, applied to this model),
# Gaussian noise calibrated with the RDP accountant to the same (eps, delta), same pre-round budget check,
# same model selection (max val macro-F1 with val FAR <= FAR_CAP) and the test set used once (Block 19).
# Detector = Torre et al. Algorithm 1 as printed: Conv1D(32, k=3, ReLU) -> MaxPool1D(2) -> Flatten ->
#   Dense(128, ReLU) -> Dense(softmax). Dropout/BatchNorm are created but not connected in Algorithm 1; they are
#   left out here because BatchNorm mixes samples in a batch and is incompatible with per-sample DP-SGD.
#   Keras initialisers (Glorot-uniform weights, zero biases). Loss = cross-entropy (Algorithm 1, unweighted by
#   default; CNN_CE_WEIGHTS = True uses the same sqrt-balanced weights as the DDQN reward, as an ablation).
# Runs are tagged "torre_cnn" (CFG TAG), so their result files never collide with the DDQN runs.
CNN_STAGE = None                  # None = define only | "smoke" | "cnn_final" (fedavg/fedprox + fixed DP eps 8/3/1,
                                  # seeds 0/1/2) | "cnn_adpmeta" (ADP-Meta v2 meta-learned, eps 8/3/1, seeds 0/1/2)
CNN_CE_WEIGHTS = False            # ablation: class-weighted cross-entropy (same weights as the DDQN reward)


class TorreCNN(nn.Module):
    def __init__(self, input_dim, n_classes, filters=32, kernel=3, pool=2, dense=128):
        super().__init__()
        self.conv = nn.Conv1d(1, filters, kernel)                                   # L2
        self.pool = nn.MaxPool1d(pool)                                              # L3
        flat = filters * ((input_dim - kernel + 1) // pool)                         # L4
        self.dense = nn.Linear(flat, dense)                                         # L5
        self.out = nn.Linear(dense, n_classes)                                      # L8 (softmax inside the loss)
        for m in (self.conv, self.dense, self.out):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)               # Keras defaults

    def forward(self, x, mode="q_value"):                                           # x: [B, features] (L1)
        h = self.pool(F.relu(self.conv(x.unsqueeze(1))))
        return self.out(F.relu(self.dense(h.flatten(1))))                           # logits


class CNNAgent:
    """Same interface as SmallDDQN where the federation code needs it; trained by cross-entropy, not reward."""
    def __init__(self, input_dim, action_dim, device, class_weights=None, lr=1e-3, **_):
        self.device, self.action_dim, self.steps = device, action_dim, 0
        self.online_net = TorreCNN(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        w = class_weights if (CNN_CE_WEIGHTS and class_weights is not None) else np.ones(action_dim, np.float32)
        self.ce_w = torch.as_tensor(w, dtype=torch.float32, device=device)
        self.eps_start = self.eps_end = 0.0; self.eps_decay_steps = 1                # unused (no exploration)

    def update_target(self):
        pass                                                                        # no target network

    @torch.no_grad()
    def predict(self, x):
        self.online_net.eval()
        out = self.online_net(x.to(self.device)).argmax(1)
        self.online_net.train()
        return out

    def get_fl_parameters(self):
        return [p.detach().cpu().clone() for p in self.online_net.parameters()]

    @torch.no_grad()
    def set_fl_parameters(self, params):
        for p, new in zip(self.online_net.parameters(), params):
            p.data.copy_(new.to(p.device))

    @torch.no_grad()
    def param_vector(self):
        return torch.cat([p.detach().view(-1) for p in self.online_net.parameters()])


def _ce_one(net, w):
    """Loss of ONE record (needed for per-sample clipping): weighted cross-entropy on the logits."""
    def loss_one(p, b, xi, yi):                       # gather (not w[yi]) so that vmap can batch it
        logits = functional_call(net, (p, b), (xi.unsqueeze(0),)).squeeze(0)
        t = yi.unsqueeze(0)
        return -(w.gather(0, t) * F.log_softmax(logits, 0).gather(0, t)).squeeze(0)
    return loss_one


class SupervisedDPClient(DPClient):
    """DPClient._dp_step with the cross-entropy loss; sampling, clipping, noise, accounting, FedProx unchanged."""
    def _dp_step(self, sigma):
        sigma = float(sigma)
        assert sigma >= 0.0, "noise multiplier must be >= 0"
        if sigma > 0:
            assert self.C is not None, "DP guard: noise without clipping has unbounded sensitivity"
            self.accountant.step(noise_multiplier=sigma, sample_rate=self.q)
        else:
            self.nonprivate_steps += 1
        net = self.ddqn.online_net
        idx = (torch.rand(self.n, device=self.device) < self.q).nonzero().squeeze(1)
        params = {k: v.detach() for k, v in net.named_parameters()}
        buffers = {k: v.detach() for k, v in net.named_buffers()}
        if len(idx) > 0:
            x, y = self.X[idx], self.y[idx]
            net.train()
            g = vmap(grad(_ce_one(net, self.ddqn.ce_w)), in_dims=(None, None, 0, 0))(params, buffers, x, y)
            B = len(idx)
            if self.C is not None:
                norms = torch.sqrt(sum(gi.reshape(B, -1).pow(2).sum(1) for gi in g.values()))
                factor = (self.C / (norms + 1e-12)).clamp(max=1.0)
            else:
                factor = torch.ones(B, device=self.device)
            summed = {k: (gi * factor.view(-1, *([1] * (gi.dim() - 1)))).sum(0) for k, gi in g.items()}
        else:
            summed = {k: torch.zeros_like(v) for k, v in params.items()}
        expected_bs = self.q * self.n
        for k, p in net.named_parameters():
            p.grad = summed[k] / expected_bs
            if sigma > 0:
                p.grad = p.grad + torch.normal(0.0, sigma * self.C, size=p.shape, device=self.device) / expected_bs
            if self.global_params is not None and self.mu > 0:
                p.grad = p.grad + self.mu * (p.detach() - self.global_params[k])
        self.ddqn.optimizer.step()
        self.ddqn.optimizer.zero_grad(set_to_none=True)
        self.ddqn.steps += 1


class SupervisedFastFederation(FastFederation):
    """FastFederation._step with the cross-entropy loss (all clients in parallel; same algorithm per client)."""
    def __init__(self, clients):
        super().__init__(clients)
        self._g_ce = vmap(vmap(grad(_ce_one(self.base, clients[0].ddqn.ce_w)), in_dims=(None, None, 0, 0)),
                          in_dims=(0, 0, 0, 0))

    def _clipped_sum(self, x, y, valid):
        """Sum over each client's batch of per-record gradients clipped to C (vmap: one gradient per record)."""
        K, B = valid.shape
        p_det = {k: v.detach() for k, v in self.params.items()}
        g = self._g_ce(p_det, self.buffers, x, y)                                     # [K, B, ...]
        if self.C is not None:
            norms = torch.sqrt(sum(gi.reshape(K, B, -1).pow(2).sum(-1) for gi in g.values()))
            factor = (self.C / (norms + 1e-12)).clamp(max=1.0)
        else:
            factor = torch.ones(K, B, device=self.dev)
        factor = factor * valid
        return {k: (gi * factor.view(K, B, *([1] * (gi.dim() - 2)))).sum(1) for k, gi in g.items()}

    def _step(self, sigma, mu, gp):
        K, dev = self.K, self.dev
        take = (torch.rand(self.row_ok.shape, device=dev) < self.q[:, None]) & self.row_ok
        cnt = take.sum(1)
        B = int(cnt.max())
        if B > 0:
            order = torch.argsort((~take).to(torch.int8), dim=1, stable=True)[:, :B]
            valid = torch.arange(B, device=dev)[None, :] < cnt[:, None]
            x = torch.gather(self.X, 1, order[..., None].expand(-1, -1, self.X.shape[2]))
            y = torch.gather(self.Y, 1, order)
            summed = self._clipped_sum(x, y, valid)
        else:
            summed = {k: torch.zeros_like(v) for k, v in self.params.items()}
        ebs = self.q * self.n
        for k in self.names:
            p = self.params[k]
            shape = (K,) + (1,) * (p.dim() - 1)
            gk = summed[k]
            if bool((sigma > 0).any()):
                gk = gk + torch.randn_like(p) * (sigma * self.C).view(shape)
            gk = gk / ebs.view(shape)
            if mu > 0:
                gk = gk + mu * (p.detach() - gp[k])
            p.grad = gk
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.steps += 1


class GhostCNNFederation(SupervisedFastFederation):
    """Same computation as SupervisedFastFederation, written in closed form for TorreCNN.
    One batched backward pass gives every record's gradient w.r.t. each layer OUTPUT (records are independent:
    no BatchNorm). Per-record parameter gradients follow exactly: Linear W -> outer(dz, h), b -> dz;
    Conv1d W -> sum_t dz[c, t] * x[t + j], b -> sum_t dz[c, t]. Linear norms use ||outer(dz, h)||^2 =
    ||dz||^2 ||h||^2 ("ghost" norm), so the [records x 53k] Dense gradients are never materialised.
    Clipping, noise, sampling, accounting and FedProx are unchanged (inherited)."""
    def _clipped_sum(self, x, y, valid):
        K, B = valid.shape
        P = {k: v.detach().requires_grad_(True) for k, v in self.params.items()}
        u = x.unfold(-1, self.base.conv.kernel_size[0], 1)                            # [K, B, T, k]
        z1 = torch.einsum("kbtj,kcj->kbct", u, P["conv.weight"][:, :, 0, :]) + P["conv.bias"][:, None, :, None]
        T2 = z1.shape[-1] // 2
        f = F.relu(z1)[..., :2 * T2].reshape(K, B, z1.shape[2], T2, 2).amax(-1).reshape(K, B, -1)
        z2 = torch.einsum("kbi,koi->kbo", f, P["dense.weight"]) + P["dense.bias"][:, None]
        h2 = F.relu(z2)
        z3 = torch.einsum("kbi,koi->kbo", h2, P["out.weight"]) + P["out.bias"][:, None]
        w = self.clients[0].ddqn.ce_w
        nll = -(w[y] * F.log_softmax(z3, -1).gather(-1, y[..., None]).squeeze(-1))
        dz1, dz2, dz3 = torch.autograd.grad((nll * valid).sum(), [z1, z2, z3])
        g1w = torch.einsum("kbct,kbtj->kbcj", dz1, u)
        g1b = dz1.sum(-1)
        f, h2 = f.detach(), h2.detach()
        if self.C is not None:
            sq = (g1w.pow(2).sum((-1, -2)) + g1b.pow(2).sum(-1)
                  + dz2.pow(2).sum(-1) * (f.pow(2).sum(-1) + 1) + dz3.pow(2).sum(-1) * (h2.pow(2).sum(-1) + 1))
            factor = (self.C / (sq.sqrt() + 1e-12)).clamp(max=1.0)
        else:
            factor = torch.ones(K, B, device=self.dev)
        a = factor * valid
        return {"conv.weight": torch.einsum("kbcj,kb->kcj", g1w, a).unsqueeze(2),
                "conv.bias": torch.einsum("kbc,kb->kc", g1b, a),
                "dense.weight": torch.einsum("kbo,kbi->koi", dz2 * a[..., None], f),
                "dense.bias": (dz2 * a[..., None]).sum(1),
                "out.weight": torch.einsum("kbo,kbi->koi", dz3 * a[..., None], h2),
                "out.bias": (dz3 * a[..., None]).sum(1)}


CNN_FAST_IMPL = GhostCNNFederation        # exact closed form; SupervisedFastFederation = generic vmap version


ARCH["torre_cnn"] = lambda i, a, d, cw=None, **kw: CNNAgent(i, a, d, cw, **kw)       # Block 19 rebuilds models


def cnn_clip_norm():
    """Same rule as dp_clip_norm: CLIP_MULT x median per-sample grad norm at a seed-42 init, VALIDATION data."""
    set_seed(SPLIT_SEED)
    ag = CNNAgent(input_dim, action_dim, device, reward_weights(), lr=CFG["LR"])
    net = ag.online_net; net.train()
    idx = np.random.default_rng(SPLIT_SEED).choice(len(y_val), size=min(1024, len(y_val)), replace=False)
    x = torch.as_tensor(X_val[idx], device=device); yy = torch.as_tensor(y_val[idx], device=device)
    params = {k: v.detach() for k, v in net.named_parameters()}
    buffers = {k: v.detach() for k, v in net.named_buffers()}
    g = vmap(grad(_ce_one(net, ag.ce_w)), in_dims=(None, None, 0, 0))(params, buffers, x, yy)
    norms = torch.sqrt(sum(gi.reshape(len(idx), -1).pow(2).sum(1) for gi in g.values())).cpu().numpy()
    return CFG["CLIP_MULT"] * float(np.median(norms))


def run_cnn_pipeline(method, seed, eps_budget=None, C=None):
    """run_dp_pipeline (fixed noise / non-private paths) with the CNN detector."""
    mu, use_clip, private, adaptive = METHODS_DP_PIPELINE[method]
    assert not adaptive or CFG["ADP_ACTIONS"] == "pacing", "CNN + ADP-Meta: only the frozen v2 (pacing) is supported"
    mu = CFG["MU"] if mu == "MU" else mu
    assert private == (eps_budget is not None), f"{method}: eps_budget must be given iff the method is private"
    name = run_name(method, eps_budget)
    R, STEPS, delta = CFG["ROUNDS"], CFG["LOCAL_STEPS"], CFG["DELTA"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    parts = dirichlet_split(y_dp, NUM_CLIENTS, ALPHA, np.random.default_rng(seed), MIN_PER_CLIENT)
    _a = np.concatenate(parts)
    assert len(_a) == len(y_dp) and len(np.unique(_a)) == len(y_dp), "A record is in more than one client"
    w = reward_weights()
    mk = lambda: CNNAgent(input_dim, action_dim, device, w, lr=CFG["LR"])
    g = mk()
    clients = [SupervisedDPClient(mk(), X_dp[ci], y_dp[ci], device, model_name=f"Client{k+1}",
                                  batch_size=CFG["BATCH"], clip_norm=(C if use_clip else None), delta=delta)
               for k, ci in enumerate(parts)]
    base = ([get_noise_multiplier(target_epsilon=eps_budget, target_delta=delta, sample_rate=c.q,
                                  steps=R * STEPS, accountant="rdp") for c in clients] if private
            else [0.0] * len(clients))
    fast = CNN_FAST_IMPL(clients) if CFG["FAST"] and CFG["CLIENT_FRAC"] >= 1.0 else None
    if adaptive:
        return _run_cnn_paced(name, method, seed, eps_budget, C, g, clients, parts, rng, mu, fast)
    print(f"[{name} | seed {seed}] Torre CNN ({sum(p.numel() for p in g.online_net.parameters())} params) | "
          f"clip={C if use_clip else 'none'} | mu={mu} | sigma median={np.median(base):.3f}")
    history, states, eps_now, t0 = [], [], None, time.time()
    for r in range(1, R + 1):
        active = rng.choice(len(clients), size=max(1, math.ceil(CFG["CLIENT_FRAC"] * len(clients))), replace=False)
        if private and max(clients[i].projected_epsilon(base[i], STEPS) for i in active) > eps_budget:
            print(f"[{name}] Round {r}: privacy budget would be exceeded - stopping before spending."); break
        gp = g.get_fl_parameters()
        params, sizes, _ = fit_clients(clients, active, gp, {int(i): base[i] for i in active}, STEPS, mu, fast)
        g.set_fl_parameters(weighted_average(params, sizes))
        h = eval_metrics(g, val_dataset)
        eps_now = max(c.epsilon_spent() for c in clients) if private else None
        if private:
            assert eps_now <= eps_budget + 1e-9, "DP guard: budget exceeded"
        h.update({"round": r, "noise_mult": 1.0, "forced": False, "epsilon": eps_now,
                  "median_sigma": float(np.median(base))})
        history.append(h); states.append([t.clone() for t in g.get_fl_parameters()])
        _log_round(name, r, h, f" | eps={eps_now if eps_now is None else round(eps_now, 3)}")
    return _save(name, seed, history, states, "torre_cnn", {
        "method": method, "private": private, "eps_budget": eps_budget, "final_epsilon": eps_now,
        "delta": delta if private else None, "clip_norm": C if use_clip else None, "mu": mu,
        "base_sigma_per_client": base, "rounds_run": len(history), "minutes": (time.time() - t0) / 60,
        "detector": "Torre et al. 2025 Algorithm 1 1D-CNN", "ce_weights": CNN_CE_WEIGHTS})


def _cnn_fast(clients):
    return CNN_FAST_IMPL(clients) if CFG["FAST"] and CFG["CLIENT_FRAC"] >= 1.0 else None


def _run_cnn_paced(name, method, seed, eps_budget, C, g, clients, parts, rng, mu, fast):
    """_run_paced (ADP-Meta v2, Block 18a) with the CNN detector. With ADP_META_EPISODES > 0 the controller is
    meta-trained on validation (half A proxies, half B reward) with CNN proxies, then frozen and used greedily."""
    R, STEPS, delta = CFG["ROUNDS"], CFG["LOCAL_STEPS"], CFG["DELTA"]
    meta = CFG["ADP_META_EPISODES"] > 0
    if meta:
        ctrl = meta_train_controller(eps_budget, C, mk_agent=lambda: CNNAgent(input_dim, action_dim, device,
                                                                               reward_weights(), lr=CFG["LR"]),
                                     client_cls=SupervisedDPClient, fast_factory=_cnn_fast)
        mode = "greedy"
        set_seed(seed); rng = np.random.default_rng(seed)              # meta-training must not shift the run's RNG
    else:
        ctrl, mode = ADPMetaV2(CFG["PACE_ACTIONS"], lr=CFG["ADP_LR"], entropy_coef=CFG["ADP_ENTROPY"],
                               device=device), "sample_online"
    pacers = [RDPPacer(c.q, eps_budget, delta, R, STEPS) for c in clients]
    print(f"[{name} | seed {seed}] Torre CNN + ADP-Meta v2 ({mode}) | clip={C} | mu={mu} | "
          f"sigma0 median={np.median([p.sigma0 for p in pacers]):.3f}")
    t0 = time.time()
    history, states, _ = paced_episode(name, g, clients, pacers, ctrl, val_dataset, rounds=R, steps=STEPS, mu=mu,
                                       rng=rng, mode=mode, eps_budget=eps_budget, keep_states=True, fast=fast)
    return _save(name, seed, history, states, "torre_cnn", {
        "method": method, "private": True, "eps_budget": eps_budget,
        "final_epsilon": history[-1]["epsilon"] if history else 0.0,
        "delta": delta, "clip_norm": C, "mu": mu, "adp_version": "v2-pacing", "adp_mode": mode,
        "controller_file": controller_path(eps_budget) if meta else None,
        "sigma0_per_client": [p.sigma0 for p in pacers], "rounds_run": len(history),
        "minutes": (time.time() - t0) / 60, "detector": "Torre et al. 2025 Algorithm 1 1D-CNN"})


def run_cnn_all(experiments, seeds):
    """Like run_all, for the CNN detector. Finished runs are skipped; a changed config under the same tag fails."""
    C = None
    for seed in seeds:
        for method, eps in experiments:
            name = run_name(method, eps)
            if os.path.exists(result_path(name, seed)):
                with open(result_path(name, seed)) as f:
                    old = json.load(f)["cfg"]
                diff = {k for k in set(old) | set(cfg_json()) if k != "RESULTS_DIR" and old.get(k) != cfg_json().get(k)}
                if diff:
                    raise ValueError(f"{name} seed {seed} exists with a DIFFERENT config ({sorted(diff)}).")
                print(f">>> SKIP {name} seed {seed} (done)"); continue
            if METHODS_DP_PIPELINE[method][1] and C is None:
                C = cnn_clip_norm(); print(f"CNN clip norm C (rule of dp_clip_norm, validation): {C:.3f}")
            print(f"\n{'='*70}\n>>> START {name} | seed {seed}\n{'='*70}")
            run_cnn_pipeline(method, seed, eps, C)
            gc.collect()


# Same frozen settings as the DDQN final stage (FINAL_CFG without the ADP-Meta and reward-only keys, which the
# CNN does not use): CLIP_MULT 0.5. REWARD_FA is a DDQN reward term and has no counterpart in cross-entropy.
CNN_CFG = dict(CLIP_MULT=0.5, TAG="torre_cnn" + ("_cew" if CNN_CE_WEIGHTS else ""))
CNN_EXPS = [("fedavg_np", None), ("fedprox_np", None), ("dp_fixed", 8.0), ("dp_fixed", 3.0), ("dp_fixed", 1.0)]
CNN_ADP_CFG = dict(ADP_ACTIONS="pacing", ADP_META_EPISODES=10)   # the frozen DDQN final stage's ADP-Meta settings
CNN_STAGES = {
    "smoke": (dict(ROUNDS=2, LOCAL_STEPS=5), [0], [("fedprox_np", None), ("dp_fixed", 8.0)]),
    "cnn_final": ({}, [0, 1, 2], CNN_EXPS),
    "cnn_adpmeta": (CNN_ADP_CFG, [0, 1, 2], [("dp_adpmeta", 8.0), ("dp_adpmeta", 3.0), ("dp_adpmeta", 1.0)]),
    "smoke_adpmeta": (CNN_ADP_CFG | dict(ROUNDS=2, LOCAL_STEPS=3, ADP_META_EPISODES=2), [0], [("dp_adpmeta", 8.0)]),
}
if CNN_STAGE is not None:
    over, seeds, exps = CNN_STAGES[CNN_STAGE]
    CFG.clear(); CFG.update(_DEFAULT_CFG); CFG.update(CNN_CFG); CFG.update(over)
    if CNN_STAGE.startswith("smoke"):
        CFG["TAG"] += "_smoke"
    run_cnn_all(exps, seeds)
    CFG.clear(); CFG.update(_DEFAULT_CFG)
