# BLOCK 18 (UPDATED v3): Run ALL experiments - non-DP FL (Ch.5) + DP / ADP-Meta (Ch.6)
# DP settings locked from validation-only diagnostics D4-D8:
#   batch 256 | 400 local steps | 20 rounds | lr 1e-3 | clip C = median per-sample grad norm on VALIDATION data
#   per-client sigma calibrated to each client's own sampling rate and the eps budget
# - Each run saves its own JSON + model; finished runs are SKIPPED; failures print in full and the loop continues
import time, json, os, gc, traceback
from opacus.accountants.utils import get_noise_multiplier

ROUNDS, CLIENT_FRAC = 50, 0.8                     # non-DP FL runs
DELTA = 1e-5
DP_ROUNDS, DP_STEPS, DP_BATCH, DP_LR = 20, 400, 256, 1e-3
LAMBDA = 1.0                                      # privacy-cost weight in the ADP-Meta reward
LEVEL_MULTS = (0.8, 0.9, 1.0, 1.2, 1.5)           # ADP-Meta actions: multiplier on each client's base sigma

# ---------- per-sample gradient norms (used to choose the clip norm C from server-side VALIDATION data) ----------
def per_sample_grad_norms(agent, X, y, n=1024):
    net = agent.online_net; net.train()
    idx = np.random.default_rng(SEED).choice(len(y), size=min(n, len(y)), replace=False)
    x  = torch.as_tensor(X[idx], device=device); yy = torch.as_tensor(y[idx], device=device)
    a  = agent.act(x); r = agent.reward(a, yy)
    params  = {k: v.detach() for k, v in net.named_parameters()}
    buffers = {k: v.detach() for k, v in net.named_buffers()}
    def loss_fn(p, b, xi, ai, ri):
        qv = functional_call(net, (p, b), (xi.unsqueeze(0),)).squeeze(0)
        return F.smooth_l1_loss(qv.gather(0, ai.unsqueeze(0)).squeeze(0), ri)
    g = vmap(grad(loss_fn), in_dims=(None, None, 0, 0, 0), randomness="different")(params, buffers, x, a, r)
    return torch.sqrt(sum(gi.reshape(len(idx), -1).pow(2).sum(1) for gi in g.values())).cpu().numpy()

torch.manual_seed(SEED)
_ag = DDQN(input_dim, action_dim, device, class_weights_dp, gamma=0.0, lr=DP_LR)
C_DP = float(np.median(per_sample_grad_norms(_ag, X_val, y_val)))
del _ag
print(f"DP clip norm C (median per-sample grad norm on validation): {C_DP:.3f}")

class NoNoiseClient(DPClient):                    # control run only: same pipeline, sigma = 0, NO privacy claim
    def epsilon_spent(self): return 0.0

def _finish(name, history, best_state, best_round, best_acc, extra):
    plot_training_curves(history)
    model = DDQN(input_dim, action_dim, device, class_weights)
    model.set_fl_parameters(best_state)
    print(f"Best validation round: {best_round} (Acc={100*best_acc:.2f}%)")
    res = final_report(model, test_dataset, title=f"TEST - {name}")
    torch.save(best_state, f"{name}_best.pt")
    json.dump({"run": name, "seed": SEED, "client_frac": CLIENT_FRAC,
               "features": selected_features, "best_round": best_round, "history": history,
               "test": res["ids"], "auc": res["auc"], "confusion_matrix": res["cm"].tolist(), **extra},
              open(f"{name}.json", "w"), indent=2)
    print(f"Saved {name}.json")

# ---------- Chapter 5: non-private FL (FedProx / FedAvg) ----------
def run_fl(name, mu, local_epochs=2):
    assert DP_ENABLED is False, "Non-DP runs need DP_ENABLED = False (Block 13)"
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    rng = np.random.default_rng(SEED)
    g = DDQN(input_dim, action_dim, device, class_weights)
    clients = [FLClient(DDQN(input_dim, action_dim, device, class_weights), client_loaders[k], device,
                        model_name=f"Client{k+1}") for k in range(NUM_CLIENTS)]
    comm = CommunicationTracker()
    history, best_acc, best_round, best_state = [], -1.0, 0, None
    for r in range(1, ROUNDS + 1):
        val, _ = federated_round(r, g, clients, rng, client_fraction=CLIENT_FRAC, local_epochs=local_epochs,
                                 mu=mu, epsilon=1.0, clip_norm=1.0, delta=DELTA, comm_tracker=comm)
        history.append({"round": r, **val})
        if val["Accuracy"] > best_acc:
            best_acc, best_round, best_state = val["Accuracy"], r, [p.clone() for p in g.get_fl_parameters()]
    _finish(name, history, best_state, best_round, best_acc,
            {"type": "fl", "dp_enabled": False, "rounds": ROUNDS, "mu": mu, "local_epochs": local_epochs,
             "comm": comm.get_efficiency()})

# ---------- Chapter 6: DP with ADP-Meta / fixed sigma / no-noise control ----------
def run_dp(name, adaptive, eps_budget, noise=True, mu=0.01):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    rng = np.random.default_rng(SEED)
    Client = DPClient if noise else NoNoiseClient
    g = DDQN(input_dim, action_dim, device, class_weights_dp, gamma=0.0, lr=DP_LR)
    clients = [Client(DDQN(input_dim, action_dim, device, class_weights_dp, gamma=0.0, lr=DP_LR),
                      X_dp[ci], y_dp[ci], device, model_name=f"Client{k+1}",
                      batch_size=DP_BATCH, clip_norm=C_DP, delta=DELTA)
               for k, ci in enumerate(dp_client_indices)]

    # Per-client base sigma: keeps EACH client within eps_budget even if picked in every one of DP_ROUNDS
    if noise:
        base = [get_noise_multiplier(target_epsilon=eps_budget, target_delta=DELTA, sample_rate=min(1.0, c.q),
                                     steps=DP_ROUNDS * DP_STEPS, accountant="rdp") for c in clients]
    else:
        base = [0.0] * len(clients)
    budget = eps_budget if noise else 1.0
    print(f"[{name}] base sigma per client: min {min(base):.2f} | median {np.median(base):.2f} | max {max(base):.2f}")

    ctrl = ADPMetaController(list(LEVEL_MULTS), device=device) if adaptive else None
    history, best_acc, best_round, best_state = [], -1.0, 0, None
    acc_prev, dacc_prev, eps_prev = evaluate_global(g, val_dataset).ids_metrics()["Accuracy"], 0.0, 0.0
    for r in range(1, DP_ROUNDS + 1):
        state = [r / DP_ROUNDS, acc_prev, dacc_prev, eps_prev / budget]
        mult = ctrl.select(state)[1] if adaptive else 1.0          # controller picks the noise multiplier
        k = max(1, math.ceil(CLIENT_FRAC * len(clients)))
        active = rng.choice(len(clients), size=k, replace=False)
        gp = g.get_fl_parameters()
        params, sizes = [], []
        for i in active:
            p, n, _ = clients[i].fit(gp, {"noise_multiplier": base[i] * mult, "local_steps": DP_STEPS, "mu": mu})
            params.append(p); sizes.append(n)
        g.set_fl_parameters(weighted_average(params, sizes))
        val = evaluate_global(g, val_dataset).ids_metrics()
        eps_now = max(c.epsilon_spent() for c in clients) if noise else 0.0
        dacc = val["Accuracy"] - acc_prev
        reward = dacc - LAMBDA * (eps_now - eps_prev) / budget
        if adaptive:
            ctrl.update(reward, r)
        history.append({"round": r, **val, "noise_mult": mult, "median_sigma": float(np.median(base)) * mult,
                        "epsilon": eps_now, "reward": reward})
        print(f"[{name}] Round {r:02d} | mult={mult:.2f} | eps={eps_now:.3f}/{eps_budget} | "
              f"Acc={100*val['Accuracy']:.2f}% | FAR={100*val['IDS_FAR']:.2f}% | reward={reward:+.4f}")
        if val["Accuracy"] > best_acc:
            best_acc, best_round, best_state = val["Accuracy"], r, [p.clone() for p in g.get_fl_parameters()]
        acc_prev, dacc_prev, eps_prev = val["Accuracy"], dacc, eps_now
        if noise and eps_now >= eps_budget:
            print(f"[{name}] Privacy budget reached at round {r} - stopping.")
            break
    _finish(name, history, best_state, best_round, best_acc,
            {"type": "dp" if noise else "dp_control_nonoise", "adaptive": adaptive, "eps_budget": eps_budget,
             "final_epsilon": eps_prev if noise else None, "delta": DELTA, "clip_norm": C_DP,
             "base_sigma_per_client": base, "level_mults": list(LEVEL_MULTS) if adaptive else [1.0],
             "rounds": DP_ROUNDS, "local_steps": DP_STEPS, "batch_size": DP_BATCH, "lr": DP_LR,
             "lambda": LAMBDA, "mu": mu, "controller": ctrl.history if adaptive else []})

# ---------- Experiment list (order = priority) ----------
EXPERIMENTS = [
    ("fl_fedprox_nonDP",    run_fl, dict(mu=0.01)),
    ("fl_fedavg_nonDP",     run_fl, dict(mu=0.0)),
    ("dp_control_nonoise",  run_dp, dict(adaptive=False, eps_budget=None, noise=False)),
    ("dp_adpmeta_eps8",     run_dp, dict(adaptive=True,  eps_budget=8.0)),
    ("dp_fixed_eps8",       run_dp, dict(adaptive=False, eps_budget=8.0)),
    ("dp_adpmeta_eps3",     run_dp, dict(adaptive=True,  eps_budget=3.0)),
    ("dp_fixed_eps3",       run_dp, dict(adaptive=False, eps_budget=3.0)),
    ("dp_adpmeta_eps1",     run_dp, dict(adaptive=True,  eps_budget=1.0)),
    ("dp_fixed_eps1",       run_dp, dict(adaptive=False, eps_budget=1.0)),
]

T0 = time.time()
for name, fn, kwargs in EXPERIMENTS:
    if os.path.exists(f"{name}.json"):
        print(f"\n>>> SKIP {name} (already done)"); continue
    print(f"\n{'='*70}\n>>> START {name} | elapsed {(time.time()-T0)/60:.0f} min\n{'='*70}")
    try:
        fn(name, **kwargs)
    except Exception:
        print(f"!!! {name} FAILED:"); traceback.print_exc()
    gc.collect(); torch.cuda.empty_cache()

# ---------- Summary table ----------
rows = []
for name, _, _ in EXPERIMENTS:
    if os.path.exists(f"{name}.json"):
        d = json.load(open(f"{name}.json"))
        fe = d.get("final_epsilon")
        rows.append({"run": name, "Acc %": round(100*d["test"]["Accuracy"], 2), "FAR %": round(100*d["test"]["IDS_FAR"], 2),
                     "DR %": round(100*d["test"]["IDS_DR"], 2), "final eps": round(fe, 3) if fe is not None else "no DP",
                     "best round": d["best_round"]})
summary = pd.DataFrame(rows)
print("\n", summary.to_string(index=False))
summary.to_csv("summary_all_runs.csv", index=False)
print(f"\nTotal time: {(time.time()-T0)/60:.0f} min")
