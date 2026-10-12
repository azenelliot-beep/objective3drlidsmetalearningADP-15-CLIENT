# =====================================================================================
# BLOCK 18a (v4 - C0): Experiment definitions - ONE runner for every method, VALIDATION ONLY
#   Methods (same seeds for all -> paired comparison):
#     fedavg_np / fedprox_np  : non-private, SAME pipeline as DP (all train data, small DDQN, gamma 0,
#                               Poisson batches, 20 rounds x 400 steps) but no clipping and no noise
#     dp_clip_nonoise         : DP pipeline with clipping but sigma = 0 (diagnostic: cost of clipping alone)
#     dp_fixed                : record-level DP-SGD, constant noise calibrated to the budget over all rounds
#     dp_adpmeta              : record-level DP-SGD, ADP-Meta controller (Block 17) picks the noise each round
#     fedavg_ch5 / fedprox_ch5: Chapter 5 non-private pipeline (resampled data, 42.6k DDQN, gamma 0.1) - reference
#   Model selection (fixed BEFORE any new test result, applied to EVERY method):
#     highest validation macro-F1 among rounds with validation FAR <= FAR_CAP;
#     if no round meets the cap -> round with the lowest validation FAR, and cap_met = False is recorded.
#   The test set is NOT touched here. Block 19 evaluates the selected models on test, once, at the end.
# Seeds: train/val/test split, scaler and feature selection stay fixed (SEED = 42, Blocks 8/8b).
#        run_seed drives model init, Dirichlet client split, Poisson sampling, client order,
#        ADP-Meta policy init and (Chapter 5 runs) the resampling.
# =====================================================================================
import copy, time, json, os, gc, traceback
from opacus.accountants.utils import get_noise_multiplier

SPLIT_SEED = SEED                      # 42 - fixed data split / scaler / feature selection
CFG = dict(
    ROUNDS=20, LOCAL_STEPS=400, BATCH=256, LR=1e-3, CLIENT_FRAC=1.0, MU=0.01, DELTA=1e-5,
    FAR_CAP=0.10,
    LEVEL_MULTS=(0.8, 0.9, 1.0, 1.2, 1.5), LAMBDA=1.0,         # ADP-Meta v1 (legacy, ADP_ACTIONS="mult")
    ADP_ACTIONS="mult",                                          # "mult" = v1 (C0) | "pacing" = v2 (C5, Block 17b)
    PACE_ACTIONS=(0.5, 0.75, 1.0, 1.5, 2.0), ADP_LR=1e-2, ADP_ENTROPY=0.01,
    ADP_META_EPISODES=0,                                         # C6: >0 -> meta-train on validation, then frozen
    PROXY_MIN_PER_CLIENT=50,
    TAG="",                                                      # appended to run names: one tag per ablation
    # Sweep-1 knobs (defaults = previous behaviour)
    WEIGHTS_FROM="train",      # C1: reward class weights from pooled "train" counts or server-held "val" counts
    REWARD_POWER=0.5,          # C4: weights = balanced_weight ** power (0.5 = square root, as before)
    REWARD_FA=0.0,             # C4: extra penalty when a NORMAL flow is flagged as an attack (false alarm)
    REWARD_MISS=0.0,           # C4: extra penalty when an ATTACK is called normal (missed attack)
    CLIP_MULT=1.0,             # C3: C = CLIP_MULT x median per-sample grad norm on validation
    CH5_ROUNDS=50, CH5_FRAC=0.8, CH5_LOCAL_EPOCHS=2, CH5_OVERSAMPLE_CAP=10,
    RESULTS_DIR="results",
    FAST=True,
    # ADP-Meta v3 (Block 17c, ADP_ACTIONS = "pacing_clip")
    CLIP_ACTIONS=(0.5, 1.0, 2.0),   # clip multiplier on C_base chosen each round
    META_EPS=(8.0, 3.0, 1.0),       # one shared controller is meta-trained across these budgets
    TWIN_POOL=4,                    # practice seeds; each has one cached fixed-schedule twin per eps                 # Block 16b: train all clients in parallel on the GPU (same algorithm, much faster)
)
_DEFAULT_CFG = dict(CFG)               # Block 18b resets CFG to these defaults before applying a stage

# ------------------------------ smaller DDQN network for private training (D10) ------------------------------
class SmallQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32),        nn.LayerNorm(32), nn.ReLU())
        self.q_head = nn.Linear(32, action_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)
    def forward(self, x, mode="q_value"):
        return self.q_head(self.feature_extractor(x))

NORMAL_IDX = class_names.index("normal")

class SmallDDQN(DDQN):                       # identical DDQN logic (reward, TD update, target net), smaller network
    def __init__(self, *args, lr=1e-3, **kwargs):
        super().__init__(*args, lr=lr, **kwargs)
        self.online_net = SmallQNetwork(input_dim, action_dim).to(self.device)
        self.target_net = SmallQNetwork(input_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

    def reward(self, actions, labels):
        """+w[y] if correct, -w[y] if wrong (as before), plus optional cost-sensitive IDS penalties (C4).
        Still a pure reward: the detector is trained only with the TD loss on this reward."""
        w = self.reward_w[labels]
        r = torch.where(actions == labels, w, -w)
        wrong = actions != labels
        if CFG["REWARD_FA"]:
            r = r - CFG["REWARD_FA"] * (wrong & (labels == NORMAL_IDX)).float()
        if CFG["REWARD_MISS"]:
            r = r - CFG["REWARD_MISS"] * (wrong & (labels != NORMAL_IDX) & (actions == NORMAL_IDX)).float()
        return r

def reward_weights():
    """Class weights for the DP-pipeline reward. 'val' uses server-held validation counts, so no statistic of the
    private training data is used outside DP (C1)."""
    yy = {"train": y_dp, "val": y_val}[CFG["WEIGHTS_FROM"]]
    cw = compute_class_weight('balanced', classes=np.arange(len(class_names)), y=yy) ** CFG["REWARD_POWER"]
    return (cw / cw.mean()).astype(np.float32)

ARCH = {"small": SmallDDQN, "large": DDQN}

# ------------------------------ helpers ------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def prf_from_cm(cm):
    cm = np.array(cm, dtype=float); tp = np.diag(cm)
    p  = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    r  = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return p, r, f1, cm.sum(1)

def eval_metrics(agent, dataset):
    """Accuracy, IDS FAR/DR, macro P/R/F1 and per-class F1 of `agent` on `dataset`."""
    m = evaluate_global(agent, dataset)
    ids = m.ids_metrics()
    P, R, F1, sup = prf_from_cm(m.cm)
    return {"Accuracy": ids["Accuracy"], "FAR": ids["IDS_FAR"], "DR": ids["IDS_DR"],
            "Macro_P": float(P.mean()), "Macro_R": float(R.mean()), "Macro_F1": float(F1.mean()),
            "Weighted_F1": float((F1 * sup / sup.sum()).sum()),
            "F1_per_class": {n: float(f) for n, f in zip(class_names, F1)}, "cm": m.cm.tolist()}

def select_best_round(history, far_cap):
    """Index into history. Rule: max val macro-F1 among rounds with val FAR <= far_cap (earliest on ties);
    if none qualifies -> lowest val FAR (higher macro-F1, then earliest, on ties). Returns (index, cap_met)."""
    assert history, "empty history"
    ok = [i for i, h in enumerate(history) if h["FAR"] <= far_cap]
    if ok:
        return max(ok, key=lambda i: (history[i]["Macro_F1"], -i)), True
    return min(range(len(history)), key=lambda i: (history[i]["FAR"], -history[i]["Macro_F1"], i)), False

def per_sample_grad_norms(agent, X, y, n=1024):          # used only to pick C (server-held VALIDATION data)
    net = agent.online_net; net.train()
    idx = np.random.default_rng(SPLIT_SEED).choice(len(y), size=min(n, len(y)), replace=False)
    x  = torch.as_tensor(X[idx], device=device); yy = torch.as_tensor(y[idx], device=device)
    a  = agent.act(x); r = agent.reward(a, yy)
    params  = {k: v.detach() for k, v in net.named_parameters()}
    buffers = {k: v.detach() for k, v in net.named_buffers()}
    def loss_fn(p, b, xi, ai, ri):
        qv = functional_call(net, (p, b), (xi.unsqueeze(0),)).squeeze(0)
        return F.smooth_l1_loss(qv.gather(0, ai.unsqueeze(0)).squeeze(0), ri)
    g = vmap(grad(loss_fn), in_dims=(None, None, 0, 0, 0), randomness="different")(params, buffers, x, a, r)
    return torch.sqrt(sum(gi.reshape(len(idx), -1).pow(2).sum(1) for gi in g.values())).cpu().numpy()

def dp_clip_norm():
    """C = median per-sample grad norm at init on VALIDATION data. Fixed seed -> same C for every method and seed."""
    set_seed(SPLIT_SEED)
    ag = SmallDDQN(input_dim, action_dim, device, reward_weights(), gamma=0.0, lr=CFG["LR"])
    return CFG["CLIP_MULT"] * float(np.median(per_sample_grad_norms(ag, X_val, y_val)))

def run_name(method, eps_budget=None):
    base = method if eps_budget is None else f"{method}_eps{eps_budget:g}"
    return f"{base}__{CFG['TAG']}" if CFG.get("TAG") else base

def cfg_json():
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()}

def result_path(name, seed, suffix=".json"):
    return os.path.join(CFG["RESULTS_DIR"], f"{name}__seed{seed}{suffix}")

def _save(name, seed, history, states, arch, extra):
    bi, cap_met = select_best_round(history, CFG["FAR_CAP"])
    os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
    torch.save(states[bi], result_path(name, seed, "_best.pt"))
    out = {"run": name, "seed": seed, "split_seed": SPLIT_SEED, "arch": arch,
           "selection_rule": f"max val Macro_F1 s.t. val FAR <= {CFG['FAR_CAP']}; else min val FAR",
           "best_round": history[bi]["round"], "cap_met": cap_met, "val_at_best": history[bi],
           "history": [{k: v for k, v in h.items() if k != "cm"} for h in history],
           "cfg": cfg_json(),
           "features": list(selected_features), **extra}
    with open(result_path(name, seed), "w") as f:
        json.dump(out, f, indent=2)
    h = history[bi]
    print(f"[{name} | seed {seed}] selected round {h['round']} (cap met: {cap_met}) | VAL Acc={100*h['Accuracy']:.2f}% "
          f"MacroF1={100*h['Macro_F1']:.1f} FAR={100*h['FAR']:.2f}% DR={100*h['DR']:.2f}%")
    return out

def _log_round(name, r, h, extra=""):
    print(f"[{name}] Round {r:02d}{extra} | VAL Acc={100*h['Accuracy']:.2f}% F1={100*h['Macro_F1']:.1f} "
          f"FAR={100*h['FAR']:.2f}% DR={100*h['DR']:.2f}%")
    print("          per-class F1: " + " | ".join(f"{n}={100*f:.0f}" for n, f in h["F1_per_class"].items()))

# ------------------------------ DP-pipeline runner (fedavg_np, fedprox_np, dp_*) ------------------------------
METHODS_DP_PIPELINE = {
    #  method            : (mu,          clip,   private, adaptive)
    "fedavg_np"          : (0.0,         False,  False,   False),
    "fedprox_np"         : ("MU",        False,  False,   False),
    "dp_clip_nonoise"    : ("MU",        True,   False,   False),
    "dp_fixed"           : ("MU",        True,   True,    False),
    "dp_adpmeta"         : ("MU",        True,   True,    True),
}

def run_dp_pipeline(method, seed, eps_budget=None, C=None):
    mu, use_clip, private, adaptive = METHODS_DP_PIPELINE[method]
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
    mk = lambda: SmallDDQN(input_dim, action_dim, device, w, gamma=0.0, lr=CFG["LR"])
    g = mk()
    clients = [DPClient(mk(), X_dp[ci], y_dp[ci], device, model_name=f"Client{k+1}", batch_size=CFG["BATCH"],
                        clip_norm=(C if use_clip else None), delta=delta) for k, ci in enumerate(parts)]
    if private:
        base = [get_noise_multiplier(target_epsilon=eps_budget, target_delta=delta, sample_rate=c.q,
                                     steps=R * STEPS, accountant="rdp") for c in clients]
    else:
        base = [0.0] * len(clients)
    fast = make_fast(clients)
    if adaptive and CFG["ADP_ACTIONS"] == "pacing_clip":           # ADP-Meta v3 (Block 17c)
        return _run_paced_clip(name, method, seed, eps_budget, C, g, clients, parts, rng, mu, fast)
    if adaptive and CFG["ADP_ACTIONS"] == "pacing":                # ADP-Meta v2 (C5/C6, Block 17b)
        return _run_paced(name, method, seed, eps_budget, C, g, clients, parts, rng, mu, fast)
    ctrl = ADPMetaController(list(CFG["LEVEL_MULTS"]), device=device) if adaptive else None
    print(f"[{name} | seed {seed}] clip={C if use_clip else 'none'} | mu={mu} | "
          f"sigma median={np.median(base):.3f} | client sizes={[len(p) for p in parts]}")

    history, states, ctrl_hist = [], [], []
    acc_prev, dacc_prev, eps_prev = eval_metrics(g, val_dataset)["Accuracy"], 0.0, 0.0
    t0 = time.time()
    for r in range(1, R + 1):
        mult, forced = 1.0, False
        if adaptive:
            mult = ctrl.select([r / R, acc_prev, dacc_prev, eps_prev / eps_budget])[1]
        active = rng.choice(len(clients), size=max(1, math.ceil(CFG["CLIENT_FRAC"] * len(clients))), replace=False)

        if private:                                         # pre-round budget check: nobody may exceed the budget
            over = lambda m: max(clients[i].projected_epsilon(base[i] * m, STEPS) for i in active) > eps_budget
            if over(mult):
                if adaptive and not over(max(CFG["LEVEL_MULTS"])):
                    mult, forced = max(CFG["LEVEL_MULTS"]), True
                else:
                    print(f"[{name}] Round {r}: privacy budget would be exceeded - stopping before spending.")
                    break

        gp = g.get_fl_parameters()
        params, sizes, _ = fit_clients(clients, active, gp, {int(i): base[i] * mult for i in active}, STEPS, mu, fast)
        g.set_fl_parameters(weighted_average(params, sizes))

        h = eval_metrics(g, val_dataset)
        eps_now = max(c.epsilon_spent() for c in clients) if private else None
        if private:
            assert eps_now <= eps_budget + 1e-9, "DP guard: budget exceeded"
        if adaptive:
            dacc = h["Accuracy"] - acc_prev
            reward = dacc - CFG["LAMBDA"] * (eps_now - eps_prev) / eps_budget
            if not forced:
                ctrl.update(reward, r)
            ctrl_hist.append({"round": r, "mult": mult, "forced": forced, "reward": reward})
            dacc_prev = dacc
        h.update({"round": r, "noise_mult": mult, "forced": forced, "epsilon": eps_now,
                  "median_sigma": float(np.median(base)) * mult})
        history.append(h); states.append([t.clone() for t in g.get_fl_parameters()])
        _log_round(name, r, h, f" | mult={mult:.2f}{'*' if forced else ''} | eps={eps_now if eps_now is None else round(eps_now, 3)}")
        acc_prev = h["Accuracy"]
        if private:
            eps_prev = eps_now

    return _save(name, seed, history, states, "small", {
        "method": method, "private": private, "eps_budget": eps_budget,
        "final_epsilon": eps_prev if private else None,   # TOTAL spent over all rounds (server saw every update)
        "delta": delta if private else None, "clip_norm": C if use_clip else None, "mu": mu,
        "base_sigma_per_client": base, "rounds_run": len(history), "minutes": (time.time() - t0) / 60,
        "controller": ctrl_hist})

def make_fast(clients):
    return FastFederation(clients) if CFG["FAST"] and CFG["CLIENT_FRAC"] >= 1.0 else None

def _run_paced(name, method, seed, eps_budget, C, g, clients, parts, rng, mu, fast=None):
    R, STEPS, delta = CFG["ROUNDS"], CFG["LOCAL_STEPS"], CFG["DELTA"]
    meta = CFG["ADP_META_EPISODES"] > 0
    if meta:
        ctrl, mode = meta_train_controller(eps_budget, C), "greedy"           # frozen, trained on validation only
        set_seed(seed); rng = np.random.default_rng(seed)                    # meta-training must not shift the run's RNG
    else:
        ctrl, mode = ADPMetaV2(CFG["PACE_ACTIONS"], lr=CFG["ADP_LR"], entropy_coef=CFG["ADP_ENTROPY"],
                               device=device), "sample_online"
    pacers = [RDPPacer(c.q, eps_budget, delta, R, STEPS) for c in clients]
    print(f"[{name} | seed {seed}] ADP-Meta v2 ({mode}) | clip={C} | mu={mu} | "
          f"sigma0 median={np.median([p.sigma0 for p in pacers]):.3f} | client sizes={[len(p) for p in parts]}")
    t0 = time.time()
    history, states, _ = paced_episode(name, g, clients, pacers, ctrl, val_dataset, rounds=R, steps=STEPS, mu=mu,
                                       rng=rng, mode=mode, eps_budget=eps_budget, keep_states=True, fast=fast)
    return _save(name, seed, history, states, "small", {
        "method": method, "private": True, "eps_budget": eps_budget,
        "final_epsilon": history[-1]["epsilon"] if history else 0.0,
        "delta": delta, "clip_norm": C, "mu": mu, "adp_version": "v2-pacing", "adp_mode": mode,
        "controller_file": controller_path(eps_budget) if meta else None,
        "sigma0_per_client": [p.sigma0 for p in pacers], "rounds_run": len(history),
        "minutes": (time.time() - t0) / 60})

# ------------------------------ Chapter 5 non-private reference (fedavg_ch5, fedprox_ch5) ------------------------------
def ch5_clients(seed):
    """Resampling + Dirichlet split + loaders exactly as Blocks 9/10, but seeded with run_seed."""
    counts = Counter(y_train); tgt = int(np.median(list(counts.values())))
    over  = {c: min(tgt, n * CFG["CH5_OVERSAMPLE_CAP"]) for c, n in counts.items() if n < tgt}
    under = {c: tgt for c, n in counts.items() if n > tgt}
    Xr, yr = RandomOverSampler(sampling_strategy=over, random_state=seed).fit_resample(X_train, y_train)
    Xr, yr = RandomUnderSampler(sampling_strategy=under, random_state=seed).fit_resample(Xr, yr)
    Xr, yr = Xr.astype(np.float32), yr.astype(np.int64)
    cw = compute_class_weight('balanced', classes=np.arange(len(class_names)), y=yr)
    w = (np.sqrt(cw) / np.sqrt(cw).mean()).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(Xr), torch.from_numpy(yr).long())
    parts = dirichlet_split(yr, NUM_CLIENTS, ALPHA, np.random.default_rng(seed), MIN_PER_CLIENT)
    loaders = [DataLoader(Subset(ds, ci.tolist()), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(seed + k)) for k, ci in enumerate(parts)]
    return loaders, w

def run_ch5(method, seed):
    assert DP_ENABLED is False, "Chapter 5 runs need DP_ENABLED = False (Block 13)"
    mu = {"fedavg_ch5": 0.0, "fedprox_ch5": CFG["MU"]}[method]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    loaders, w = ch5_clients(seed)
    g = DDQN(input_dim, action_dim, device, w)
    clients = [FLClient(DDQN(input_dim, action_dim, device, w), loaders[k], device, model_name=f"Client{k+1}")
               for k in range(NUM_CLIENTS)]
    history, states, t0 = [], [], time.time()
    for r in range(1, CFG["CH5_ROUNDS"] + 1):
        active = rng.choice(NUM_CLIENTS, size=max(1, math.ceil(CFG["CH5_FRAC"] * NUM_CLIENTS)), replace=False)
        gp = g.get_fl_parameters()
        params, sizes = [], []
        for i in active:
            p, n, _ = clients[i].fit(gp, {"local_epochs": CFG["CH5_LOCAL_EPOCHS"], "mu": mu})
            params.append(p); sizes.append(n)
        g.set_fl_parameters(weighted_average(params, sizes))
        h = eval_metrics(g, val_dataset); h["round"] = r
        history.append(h); states.append([t.clone() for t in g.get_fl_parameters()])
        _log_round(method, r, h)
    return _save(method, seed, history, states, "large", {
        "method": method, "private": False, "eps_budget": None, "final_epsilon": None, "mu": mu,
        "rounds_run": len(history), "minutes": (time.time() - t0) / 60, "class_weights": w.tolist()})

# ------------------------------ batch runner ------------------------------
def run_all(experiments, seeds):
    """experiments: list of (method, eps_budget or None). Finished (method, eps, seed) runs are skipped."""
    C = None
    T0 = time.time()
    for seed in seeds:
        for method, eps in experiments:
            name = run_name(method, eps)
            if os.path.exists(result_path(name, seed)):
                with open(result_path(name, seed)) as f:
                    old = json.load(f)["cfg"]
                diff = {k for k in set(old) | set(cfg_json()) if k != "RESULTS_DIR" and old.get(k) != cfg_json().get(k)}
                if diff:
                    raise ValueError(f"{name} seed {seed} exists with a DIFFERENT config ({sorted(diff)}). "
                                     f"Set CFG['TAG'] to a new name for this ablation.")
                print(f">>> SKIP {name} seed {seed} (done)"); continue
            print(f"\n{'='*70}\n>>> START {name} | seed {seed} | elapsed {(time.time()-T0)/60:.0f} min\n{'='*70}")
            try:
                if method in METHODS_DP_PIPELINE:
                    if C is None:
                        C = dp_clip_norm(); print(f"DP clip norm C (median per-sample grad norm, validation): {C:.3f}")
                    run_dp_pipeline(method, seed, eps, C)
                else:
                    assert eps is None, f"{method} is non-private"
                    run_ch5(method, seed)
            except Exception:
                print(f"!!! {name} seed {seed} FAILED:"); traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def summarize_val(results_dir=None):
    """Validation-only summary of every finished run (mean +/- std over seeds). Never reads test results."""
    rows = []
    d = results_dir or CFG["RESULTS_DIR"]
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if fn.endswith(".json") and "__seed" in fn and not fn.endswith("_TEST.json"):
            with open(os.path.join(d, fn)) as f:
                r = json.load(f)
            v = r["val_at_best"]
            rows.append({"run": r["run"], "seed": r["seed"], "best_round": r["best_round"], "cap_met": r["cap_met"],
                         "Acc": 100 * v["Accuracy"], "MacroF1": 100 * v["Macro_F1"], "FAR": 100 * v["FAR"],
                         "DR": 100 * v["DR"], "eps": r.get("final_epsilon"),
                         **{f"F1_{k}": 100 * x for k, x in v["F1_per_class"].items()}})
    return pd.DataFrame(rows)
