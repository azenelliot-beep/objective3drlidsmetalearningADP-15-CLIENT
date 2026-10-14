# BLOCK 17e (v5 - drift scenario): emerging-attack waves + drift-aware ADP-Meta v5
# WHY: in the static setting every round sees the same data mix, so an even (fixed) noise schedule is close to optimal
# and no controller can beat it. Real IDS traffic is not static: attack families appear over time. This block adds a
# SIMULATED emerging-threat scenario and a controller that watches for new attacks and moves budget to those rounds.
#  (1) Attack waves (CFG["DRIFT_WAVES"], e.g. "1:normal,scanning;5:dos,ddos;..."): a class's training records "arrive"
#      at its wave round and stay available afterwards. Records that have not arrived are never used (client.active
#      mask, Blocks 16/16b), so each record's sampling rate is still <= q and the RDP accounting is unchanged.
#      Validation / model selection always cover ALL classes: the final IDS must detect every attack.
#      The real capture order cannot be recovered (timestamps were removed as a leak and rows de-duplicated), so the
#      wave order is a declared scenario (reconnaissance first, advanced attacks last) - disclose it in the thesis.
#  (2) ADP-Meta v5 = the v3 controller (joint budget pacing k + clip multiplier m, starts at the fixed schedule,
#      twin-baseline meta-training) with THREE drift signals added to its state, all free of privacy cost
#      (public validation data or already-noised updates):
#        - unlearned share: fraction of attack classes with validation F1 < 5%  ("there is still something to learn")
#        - learning burst : largest per-class F1 increase since the last round ("a new attack is being learned now")
#        - update spike   : log(median noisy client update norm / its running mean) ("the data changed")
#  (3) The fixed-noise baseline runs in the SAME drift scenario (fixed schedule k = 1, m = 1 == constant sigma).
#      Practice episodes (validation half A) follow the same wave schedule. Test set never used.


def parse_waves(spec):
    """'1:normal,scanning;5:dos' -> {class_index: arrival_round}. Every class must appear exactly once."""
    arrival = {}
    for part in spec.split(";"):
        r, names = part.split(":")
        for n in names.split(","):
            n = n.strip()
            assert n in class_names, f"DRIFT_WAVES: unknown class '{n}'"
            assert class_names.index(n) not in arrival, f"DRIFT_WAVES: class '{n}' listed twice"
            arrival[class_names.index(n)] = int(r)
    missing = [class_names[i] for i in range(len(class_names)) if i not in arrival]
    assert not missing, f"DRIFT_WAVES: classes without an arrival round: {missing}"
    return arrival


def set_arrivals(clients, r, arrival):
    """Activate every record whose class has arrived by round r (all clients, same schedule)."""
    arrived = torch.tensor([arrival[c] <= r for c in range(len(class_names))])
    for c in clients:
        c.active = arrived.to(c.y.device)[c.y]
    return int(arrived.sum())


def clear_arrivals(clients, fast=None):
    for c in clients:
        c.active = None
    if fast is not None:
        fast.active_rows = None


class ADPMetaV5(ADPMetaV3):
    STATE_DIM = 12

    @staticmethod
    def make_state(t, T, left, f1, df1, far, upd, prev_sigma, prev_m, eps_budget, unlearned, burst, spike):
        return ADPMetaV3.make_state(t, T, left, f1, df1, far, upd, prev_sigma, prev_m, eps_budget) + \
               [unlearned, burst, spike]


def drift_signals(f1_now, f1_prev, upd, upd_mean):
    att = [k for k in f1_now if k != "normal"]
    unlearned = float(np.mean([f1_now[k] < 0.05 for k in att]))
    burst = max(0.0, max(f1_now[k] - f1_prev.get(k, 0.0) for k in att))
    spike = math.log((upd + 1e-8) / (upd_mean + 1e-8)) if upd_mean > 0 else 0.0
    return unlearned, burst, float(np.clip(spike, -3.0, 3.0))


def drift_episode(name, g, clients, pacers, policy, eval_ds, *, rounds, steps, mu, rng, C_base, eps_budget,
                  arrival, fast=None, check_budget=False, keep_states=False):
    """One federated run in the attack-wave scenario; policy(state) -> (k, m) as in v3."""
    history, states = [], []
    m0 = eval_metrics(g, eval_ds)
    f1_prev, df1_prev, far_prev, upd, sig_prev, m_prev = m0["Macro_F1"], 0.0, m0["FAR"], 0.0, 1.0, 1.0
    cls_prev, upd_mean = dict(m0["F1_per_class"]), 0.0
    unlearned, burst, spike = drift_signals(m0["F1_per_class"], {}, 0.0, 0.0)
    for r in range(1, rounds + 1):
        n_arrived = set_arrivals(clients, r, arrival)
        left = float(np.mean([p.left_frac for p in pacers]))
        k, m = policy(ADPMetaV5.make_state(r, rounds, left, f1_prev, df1_prev, far_prev, upd, sig_prev, m_prev,
                                           eps_budget, unlearned, burst, spike))
        share = pacing_share(k, rounds - r + 1)
        active = rng.choice(len(clients), size=max(1, math.ceil(CFG["CLIENT_FRAC"] * len(clients))), replace=False)
        sig = {int(i): pacers[i].sigma_for_share(share) for i in active}
        if any(s is None for s in sig.values()):
            print(f"[{name}] Round {r}: budget exhausted - stopping."); break
        if check_budget and max(clients[i].projected_epsilon(sig[i], steps) for i in sig) > eps_budget + 1e-9:
            print(f"[{name}] Round {r}: privacy budget would be exceeded - stopping before spending."); break
        C_t = C_base * m
        for c in clients:
            c.C = C_t
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
        upd_now = float(np.median([x["delta_l2"] for x in mets]))
        unlearned, burst, spike = drift_signals(h["F1_per_class"], cls_prev, upd_now, upd_mean)
        h.update({"round": r, "classes_arrived": n_arrived, "action_k": k, "clip_mult": m, "clip": C_t,
                  "share": share, "median_sigma": float(np.median(list(sig.values()))), "epsilon": eps_now,
                  "budget_left": float(np.mean([p.left_frac for p in pacers])), "median_update_norm": upd_now,
                  "drift_unlearned": unlearned, "drift_burst": burst, "drift_spike": spike})
        history.append(h)
        if keep_states:
            states.append([t.clone() for t in g.get_fl_parameters()])
        _log_round(name, r, h, f" | classes={n_arrived} | k={k} m={m} C={C_t:.2f} sigma~{h['median_sigma']:.2f}"
                               + ("" if eps_now is None else f" | eps={eps_now:.3f}"))
        f1_prev, df1_prev, far_prev = h["Macro_F1"], h["Macro_F1"] - f1_prev, h["FAR"]
        upd, sig_prev, m_prev = upd_now, h["median_sigma"], m
        upd_mean = upd_now if upd_mean == 0 else 0.7 * upd_mean + 0.3 * upd_now
        cls_prev = dict(h["F1_per_class"])
    clear_arrivals(clients, fast)
    return history, states


def controller_v5_path():
    tag = f"__{CFG['TAG']}" if CFG.get("TAG") else ""
    return os.path.join(CFG["RESULTS_DIR"], f"controller_v5_drift{tag}.pt")


def _practice_episode_v5(ep_seed, eps_budget, C, policy, XA, yA, val_B, q_real, arrival, label):
    set_seed(ep_seed)
    parts = dirichlet_split(yA, NUM_CLIENTS, ALPHA, np.random.default_rng(ep_seed), CFG["PROXY_MIN_PER_CLIENT"])
    w = reward_weights()
    mk = lambda: SmallDDQN(input_dim, action_dim, device, w, gamma=0.0, lr=CFG["LR"])
    g = mk()
    proxies = [DPClient(mk(), XA[ci], yA[ci], device, model_name=f"Proxy{k+1}", batch_size=CFG["BATCH"],
                        clip_norm=C, delta=CFG["DELTA"]) for k, ci in enumerate(parts)]
    R, STEPS = CFG["ROUNDS"], CFG["LOCAL_STEPS"]
    pacers = [RDPPacer(q_real[k], eps_budget, CFG["DELTA"], R, STEPS) for k in range(len(proxies))]
    hist, _ = drift_episode(label, g, proxies, pacers, policy, val_B, rounds=R, steps=STEPS, mu=CFG["MU"],
                            rng=np.random.default_rng(ep_seed), C_base=C, eps_budget=eps_budget, arrival=arrival,
                            fast=make_fast(proxies))
    return episode_score(hist), hist


def meta_train_controller_v5(C):
    """Twin-baseline meta-training (as v3) in the drift scenario; cached; optional META_TIME_LIMIT_MIN."""
    path = controller_v5_path()
    new = lambda: ADPMetaV5(CFG["PACE_ACTIONS"], CFG["CLIP_ACTIONS"], lr=CFG["ADP_LR"], entropy_coef=CFG["ADP_ENTROPY"])
    if os.path.exists(path):
        ctrl = new(); ctrl.load_state_dict(torch.load(path)); print(f"Loaded meta-trained controller {path}")
        return ctrl
    torch.manual_seed(SPLIT_SEED)
    ctrl = new()
    arrival = parse_waves(CFG["DRIFT_WAVES"])
    ia, ib = val_halves()
    XA, yA = X_val[ia], y_val[ia]
    val_B = TensorDataset(torch.from_numpy(X_val[ib]), torch.from_numpy(y_val[ib]).long())
    q_real = [min(1.0, CFG["BATCH"] / len(ci)) for ci in dp_client_indices]
    pool = [40_000 + i for i in range(CFG["TWIN_POOL"])]
    twins, log, t0 = {}, [], time.time()
    limit, stopped_early = CFG.get("META_TIME_LIMIT_MIN"), False
    fixed = lambda s: (1.0, 1.0)
    for e in range(CFG["ADP_META_EPISODES"]):
        used = (time.time() - t0) / 60
        if limit is not None and log and used + used / len(log) > limit:
            print(f"=== v5 meta-training stopped by META_TIME_LIMIT_MIN={limit}: {len(log)} episodes in {used:.0f} min")
            stopped_early = True
            break
        eps_b = CFG["META_EPS"][e % len(CFG["META_EPS"])]
        es = pool[(e // len(CFG["META_EPS"])) % len(pool)]
        if (es, eps_b) not in twins:
            twins[(es, eps_b)] = _practice_episode_v5(es, eps_b, C, fixed, XA, yA, val_B, q_real, arrival,
                                                      f"twin s{es} eps{eps_b:g}")[0]
        score, hist = _practice_episode_v5(es, eps_b, C, lambda s: ctrl.act(s), XA, yA, val_B, q_real, arrival,
                                           f"v5 meta ep{e+1} eps{eps_b:g}")
        adv = score - twins[(es, eps_b)]
        ctrl.finish_episode(adv)
        log.append({"episode": e + 1, "eps": eps_b, "seed": es, "score": score, "twin": twins[(es, eps_b)],
                    "advantage": adv, "actions": [(h["action_k"], h["clip_mult"]) for h in hist]})
        print(f"=== v5 meta-episode {e+1}/{CFG['ADP_META_EPISODES']} eps={eps_b:g} | score={100*score:.1f} "
              f"twin={100*twins[(es, eps_b)]:.1f} adv={100*adv:+.1f} | {(time.time()-t0)/60:.0f} min")
    os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
    torch.save(ctrl.state_dict(), path)
    with open(path.replace(".pt", "_meta_log.json"), "w") as f:
        json.dump({"episodes": log, "episodes_run": len(log), "stopped_by_time_limit": stopped_early,
                   "minutes": (time.time() - t0) / 60, "val_halves_seed": SPLIT_SEED, "cfg": cfg_json()}, f, indent=2)
    return ctrl


def _run_drift(name, method, seed, eps_budget, C, g, clients, parts, rng, mu, fast=None, adaptive=True):
    """dp_fixed (fixed schedule) or dp_adpmeta (ADP-Meta v5) in the attack-wave scenario."""
    if adaptive:
        ctrl = meta_train_controller_v5(C)
        policy = lambda s: ctrl.act(s, greedy=True)
    else:
        policy = lambda s: (1.0, 1.0)                            # fixed schedule == constant sigma, constant C
    set_seed(seed); rng = np.random.default_rng(seed)
    arrival = parse_waves(CFG["DRIFT_WAVES"])
    R, STEPS, delta = CFG["ROUNDS"], CFG["LOCAL_STEPS"], CFG["DELTA"]
    pacers = [RDPPacer(c.q, eps_budget, delta, R, STEPS) for c in clients]
    print(f"[{name} | seed {seed}] DRIFT scenario ({CFG['DRIFT_WAVES']}) | "
          f"{'ADP-Meta v5 (frozen, greedy)' if adaptive else 'fixed noise'} | C_base={C:.3f} | mu={mu}")
    t0 = time.time()
    history, states = drift_episode(name, g, clients, pacers, policy, val_dataset, rounds=R, steps=STEPS, mu=mu,
                                    rng=rng, C_base=C, eps_budget=eps_budget, arrival=arrival, fast=fast,
                                    check_budget=True, keep_states=True)
    return _save(name, seed, history, states, "small", {
        "method": method, "private": True, "eps_budget": eps_budget,
        "final_epsilon": history[-1]["epsilon"] if history else 0.0, "delta": delta, "clip_norm": C, "mu": mu,
        "scenario": "drift", "drift_waves": CFG["DRIFT_WAVES"],
        "adp_version": "v5-drift" if adaptive else "fixed", "adp_mode": "greedy" if adaptive else "fixed",
        "controller_file": controller_v5_path() if adaptive else None,
        "sigma0_per_client": [p.sigma0 for p in pacers], "rounds_run": len(history),
        "minutes": (time.time() - t0) / 60})
