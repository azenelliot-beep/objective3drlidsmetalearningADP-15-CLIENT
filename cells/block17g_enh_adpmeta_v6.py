# BLOCK 17g (enhanced stage + ADP-Meta v6): more accuracy for the SAME epsilon, privacy claim tightened
# Required framework unchanged: 15 clients, 20 rounds, agreed preprocessing, DDQN (TD loss on reward), FedProx,
# record-level local DP-SGD + Opacus RDP accounting, ADP with a meta-learned controller.
# Changes (all CFG keys; defaults = previous behaviour, used only when CFG["ENH"] is True):
#  (1) Privacy: WEIGHTS_FROM = "val" in the stage -> reward class weights come from server-held validation counts,
#      so no statistic of the private training data is used outside DP.
#  (2) SERVER_MOMENTUM = m: FedAvgM on the server, v <- m v + (avg - w), w <- w + v. Post-processing of the
#      already-privatised client models: epsilon is unchanged.
#  (3) SERVER_EMA = b: the evaluated / selected / saved model is an exponential average of the global models,
#      e <- b e + (1 - b) w. Training continues from w. Post-processing: epsilon is unchanged.
#  (4) ADP-Meta v6 (ADP_ACTIONS = "lr_clip"): every round the frozen meta-learned controller picks a learning-rate
#      multiplier and a clip multiplier (C_t = C * m). The noise multiplier stays the fixed-schedule sigma of each
#      client, so the RDP accounting is identical to dp_fixed (noise std = sigma * C_t scales with the clip).
#      Neither the learning rate nor the clip norm changes epsilon. Meta-training is the v3 twin-baseline scheme on
#      validation half A (reward on half B, score = model-selection rule), starting at (1, 1) = the fixed schedule.
# dp_fixed and dp_adpmeta both run through enh_episode() with the same seed reset, so they are exactly paired
# (a controller that always answers (1, 1) reproduces dp_fixed bit for bit). The test set is never used here.


class ADPMetaV6(ADPMetaV3):
    """Same 32-16 two-head REINFORCE policy as v3; heads = learning-rate multiplier and clip multiplier."""
    STATE_DIM = 8

    @staticmethod
    def make_state(t, T, f1, df1, far, upd, prev_lr, prev_m):
        return [t / T, f1, df1, far, math.log1p(max(0.0, upd)), math.log(prev_lr), math.log(prev_m),
                1.0 - t / T]


class ServerOpt:
    """Server-side post-processing: FedAvgM momentum on the aggregate and an EMA of the global model."""
    def __init__(self, momentum=0.0, ema=0.0):
        self.m, self.b = float(momentum), float(ema)
        self.v, self.e = None, None

    def step(self, w, avg):
        """w: current global params, avg: weighted average of the client models -> new global params."""
        if self.m <= 0:
            new = [a.clone() for a in avg]
        else:
            d = [a - x.to(a.device) for a, x in zip(avg, w)]
            self.v = d if self.v is None else [self.m * v + di for v, di in zip(self.v, d)]
            new = [x.to(a.device) + v for x, a, v in zip(w, avg, self.v)]
        if self.b > 0:
            self.e = [t.clone() for t in new] if self.e is None else \
                     [self.b * e + (1 - self.b) * t for e, t in zip(self.e, new)]
        return new

    def evaluated(self, w):
        """The model that is evaluated, selected and saved (EMA if enabled, else the global model)."""
        return w if (self.b <= 0 or self.e is None) else self.e


def _set_lr(clients, fast, lr):
    for c in clients:
        for pg in c.ddqn.optimizer.param_groups:
            pg["lr"] = lr
    if fast is not None:
        for pg in fast.opt.param_groups:
            pg["lr"] = lr


def enh_episode(name, g, clients, sigmas, policy, eval_ds, *, rounds, steps, mu, rng, C_base, eps_budget,
                fast=None, check_budget=False, keep_states=False):
    """One federated run. sigmas: fixed noise multiplier per client (dp_fixed schedule).
    policy(state) -> (lr_mult, clip_mult). Server momentum / EMA from CFG."""
    history, states = [], []
    srv = ServerOpt(CFG["SERVER_MOMENTUM"], CFG["SERVER_EMA"])
    ev = SmallDDQN(input_dim, action_dim, device, g.reward_w.detach().cpu().numpy(), gamma=0.0, lr=CFG["LR"])
    m0 = eval_metrics(g, eval_ds)
    f1_prev, df1_prev, far_prev, upd, lr_prev, m_prev = m0["Macro_F1"], 0.0, m0["FAR"], 0.0, 1.0, 1.0
    for r in range(1, rounds + 1):
        lr_mult, m = policy(ADPMetaV6.make_state(r, rounds, f1_prev, df1_prev, far_prev, upd, lr_prev, m_prev))
        active = rng.choice(len(clients), size=max(1, math.ceil(CFG["CLIENT_FRAC"] * len(clients))), replace=False)
        sig = {int(i): float(sigmas[i]) for i in active}
        if check_budget and max(clients[i].projected_epsilon(sig[i], steps) for i in sig) > eps_budget + 1e-9:
            print(f"[{name}] Round {r}: privacy budget would be exceeded - stopping before spending."); break
        C_t = C_base * m
        for c in clients:
            c.C = C_t                                           # clip norm for this round (noise sigma * C_t)
        if fast is not None:
            fast.C = C_t
        _set_lr(clients, fast, CFG["LR"] * lr_mult)
        gp = g.get_fl_parameters()
        params, sizes, mets = fit_clients(clients, list(sig), gp, sig, steps, mu, fast)
        g.set_fl_parameters(srv.step(gp, weighted_average(params, sizes)))
        ev.set_fl_parameters(srv.evaluated(g.get_fl_parameters()))
        h = eval_metrics(ev, eval_ds)
        eps_now = None
        if check_budget:
            eps_now = max(c.epsilon_spent() for c in clients)
            assert eps_now <= eps_budget + 1e-9, "DP guard: budget exceeded"
        h.update({"round": r, "lr_mult": lr_mult, "clip_mult": m, "clip": C_t, "lr": CFG["LR"] * lr_mult,
                  "median_sigma": float(np.median(list(sig.values()))), "epsilon": eps_now,
                  "median_update_norm": float(np.median([x["delta_l2"] for x in mets]))})
        history.append(h)
        if keep_states:
            states.append([t.clone() for t in ev.get_fl_parameters()])
        _log_round(name, r, h, f" | lr x{lr_mult} clip x{m} C={C_t:.2f} sigma~{h['median_sigma']:.2f}"
                               + ("" if eps_now is None else f" | eps={eps_now:.3f}"))
        f1_prev, df1_prev, far_prev = h["Macro_F1"], h["Macro_F1"] - f1_prev, h["FAR"]
        upd, lr_prev, m_prev = h["median_update_norm"], lr_mult, m
    return history, states


def controller_v6_path():
    tag = f"__{CFG['TAG']}" if CFG.get("TAG") else ""
    return os.path.join(CFG["RESULTS_DIR"], f"controller_v6_shared{tag}.pt")


def _practice_episode_v6(ep_seed, eps_budget, C, policy, XA, yA, val_B, q_real, label):
    set_seed(ep_seed)
    parts = dirichlet_split(yA, NUM_CLIENTS, ALPHA, np.random.default_rng(ep_seed), CFG["PROXY_MIN_PER_CLIENT"])
    w = reward_weights()
    mk = lambda: SmallDDQN(input_dim, action_dim, device, w, gamma=0.0, lr=CFG["LR"])
    g = mk()
    proxies = [DPClient(mk(), XA[ci], yA[ci], device, model_name=f"Proxy{k+1}", batch_size=CFG["BATCH"],
                        clip_norm=C, delta=CFG["DELTA"]) for k, ci in enumerate(parts)]
    R, STEPS = CFG["ROUNDS"], CFG["LOCAL_STEPS"]
    sig = [get_noise_multiplier(target_epsilon=eps_budget, target_delta=CFG["DELTA"], sample_rate=q_real[k],
                                steps=R * STEPS, accountant="rdp") for k in range(len(proxies))]   # real clients' noise
    hist, _ = enh_episode(label, g, proxies, sig, policy, val_B, rounds=R, steps=STEPS, mu=CFG["MU"],
                          rng=np.random.default_rng(ep_seed), C_base=C, eps_budget=eps_budget, fast=make_fast(proxies))
    return episode_score(hist), hist


def meta_train_controller_v6(C):
    """Meta-train ONE ADP-Meta v6 controller for the eps in CFG['META_EPS'] (twin-baseline, cached)."""
    path = controller_v6_path()
    ctrl = ADPMetaV6(CFG["LR_ACTIONS"], CFG["CLIP_ACTIONS"], lr=CFG["ADP_LR"], entropy_coef=CFG["ADP_ENTROPY"])
    if os.path.exists(path):
        ctrl.load_state_dict(torch.load(path)); print(f"Loaded meta-trained controller {path}")
        return ctrl
    torch.manual_seed(SPLIT_SEED)
    ctrl = ADPMetaV6(CFG["LR_ACTIONS"], CFG["CLIP_ACTIONS"], lr=CFG["ADP_LR"], entropy_coef=CFG["ADP_ENTROPY"])
    ia, ib = val_halves()
    XA, yA = X_val[ia], y_val[ia]
    val_B = TensorDataset(torch.from_numpy(X_val[ib]), torch.from_numpy(y_val[ib]).long())
    q_real = [min(1.0, CFG["BATCH"] / len(ci)) for ci in dp_client_indices]
    pool = [20_000 + i for i in range(CFG["TWIN_POOL"])]
    twins, log, t0 = {}, [], time.time()
    fixed = lambda s: (1.0, 1.0)
    limit, stopped_early = CFG.get("META_TIME_LIMIT_MIN"), False
    for e in range(CFG["ADP_META_EPISODES"]):
        used = (time.time() - t0) / 60
        if limit is not None and log and used + used / len(log) > limit:
            print(f"=== v6 meta-training stopped by META_TIME_LIMIT_MIN={limit}: {len(log)} episodes in {used:.0f} min")
            stopped_early = True
            break
        eps_b = CFG["META_EPS"][e % len(CFG["META_EPS"])]
        es = pool[(e // len(CFG["META_EPS"])) % len(pool)]
        if (es, eps_b) not in twins:
            twins[(es, eps_b)] = _practice_episode_v6(es, eps_b, C, fixed, XA, yA, val_B, q_real,
                                                      f"twin s{es} eps{eps_b:g}")[0]
        score, hist = _practice_episode_v6(es, eps_b, C, lambda s: ctrl.act(s), XA, yA, val_B, q_real,
                                           f"meta ep{e+1} eps{eps_b:g}")
        adv = score - twins[(es, eps_b)]
        ctrl.finish_episode(adv)
        log.append({"episode": e + 1, "eps": eps_b, "seed": es, "score": score, "twin": twins[(es, eps_b)],
                    "advantage": adv, "actions": [(h["lr_mult"], h["clip_mult"]) for h in hist]})
        print(f"=== v6 meta-episode {e+1}/{CFG['ADP_META_EPISODES']} eps={eps_b:g} | score={100*score:.1f} "
              f"twin={100*twins[(es, eps_b)]:.1f} adv={100*adv:+.1f} | {(time.time()-t0)/60:.0f} min")
    os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
    torch.save(ctrl.state_dict(), path)
    with open(path.replace(".pt", "_meta_log.json"), "w") as f:
        json.dump({"episodes": log, "episodes_run": len(log), "stopped_by_time_limit": stopped_early,
                   "minutes": (time.time() - t0) / 60, "val_halves_seed": SPLIT_SEED,
                   "cfg": cfg_json()}, f, indent=2)
    return ctrl


def _run_enh(name, method, seed, eps_budget, C, g, clients, parts, rng, mu, fast=None):
    """Enhanced stage: dp_fixed (policy = (1, 1)) or dp_adpmeta (frozen meta-learned v6 policy)."""
    assert method in ("dp_fixed", "dp_adpmeta"), f"{method} is not defined for the enhanced stage"
    adaptive = method == "dp_adpmeta"
    if adaptive:
        assert CFG["ADP_ACTIONS"] == "lr_clip", "the enhanced stage uses ADP-Meta v6 (ADP_ACTIONS='lr_clip')"
        ctrl = meta_train_controller_v6(C)
        policy, label = (lambda s: ctrl.act(s, greedy=True)), "ADP-Meta v6 (frozen, greedy)"
    else:
        policy, label = (lambda s: (1.0, 1.0)), "fixed noise"
    set_seed(seed); rng = np.random.default_rng(seed)        # same streams for both methods -> exact pairing
    R, STEPS, delta = CFG["ROUNDS"], CFG["LOCAL_STEPS"], CFG["DELTA"]
    sig = [get_noise_multiplier(target_epsilon=eps_budget, target_delta=delta, sample_rate=c.q, steps=R * STEPS,
                                accountant="rdp") for c in clients]
    print(f"[{name} | seed {seed}] {label} | C={C:.3f} | mu={mu} | sigma median={np.median(sig):.3f} | "
          f"server momentum={CFG['SERVER_MOMENTUM']} EMA={CFG['SERVER_EMA']} | weights from {CFG['WEIGHTS_FROM']}")
    t0 = time.time()
    history, states = enh_episode(name, g, clients, sig, policy, val_dataset, rounds=R, steps=STEPS, mu=mu, rng=rng,
                                  C_base=C, eps_budget=eps_budget, fast=fast, check_budget=True, keep_states=True)
    return _save(name, seed, history, states, "small", {
        "method": method, "private": True, "eps_budget": eps_budget,
        "final_epsilon": history[-1]["epsilon"] if history else 0.0, "delta": delta, "clip_norm": C, "mu": mu,
        "adp_version": "v6-lr-clip" if adaptive else None, "adp_mode": "greedy" if adaptive else None,
        "controller_file": controller_v6_path() if adaptive else None, "sigma_per_client": sig,
        "server_momentum": CFG["SERVER_MOMENTUM"], "server_ema": CFG["SERVER_EMA"],
        "rounds_run": len(history), "minutes": (time.time() - t0) / 60})
