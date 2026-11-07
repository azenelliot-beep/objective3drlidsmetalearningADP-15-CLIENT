"""Federated training loop: GRU clients + DQN client selection. [PAPER] Algorithm 1 (p.7).

Per round t ([PAPER] Algorithm 1; details tagged inline):
  1. build the state s_t (Eq. 6) and select clients and transmit powers (Eq. 7)
  2. broadcast the global model; each selected client trains locally for 5 epochs, batch 20
  3. aggregate weighted by local dataset size (Eqs. 18, 25)
  4. update data quality C_dq (Eq. 5), energy (Eqs. 1-4), reward (Eq. 8), DQN (Eqs. 12-13)
  5. log; at rounds 1/50/100 compute the full metric set and best/worst client accuracy

Usage (from the repo root):
  python kaur2024/src/fl.py --config kaur2024/configs/e1_iid_dqn.yaml --seed 0
  options: --csv PATH  --rounds N (smoke tests)  --out DIR  --device cuda|cpu
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel import Channel                                    # noqa: E402
from data import CLASS_NAMES, load_prepared, repo_path         # noqa: E402
from dqn_selector import Selector                              # noqa: E402
from evaluate import evaluate, plot_roc                        # noqa: E402
from gru_model import FlatAdam, GRUClassifier, Layout, functional_forward, param_names  # noqa: E402


# ------------------------------------------------------------------ config
def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path) -> dict:
    path = repo_path(path)
    cfg = yaml.safe_load(path.read_text())
    if "inherit" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("inherit")), cfg)
    return cfg


def default_out_dir(cfg, seed) -> Path:
    """Core runs (seed 0, paper experiments) -> results/core; everything else -> results/extra/runs."""
    root = repo_path(cfg["out_root"])
    core = seed == 0 and not cfg["name"].startswith("x2_")
    return (root / "core" if core else root / "extra" / "runs") / cfg["name"] / f"seed{seed}"


# --------------------------------------------------------------- training
def local_train(global_flat, layout, clients, X, y, cfg, gen):
    """Train len(clients) copies of the global model in parallel, each on its own data only.
    Clients are processed in order of decreasing batch count so the active ones at any step are a
    prefix; a client with no batch left in this epoch is not updated (its Adam step is skipped)."""
    lc, mc = cfg["local"], cfg["model"]
    B, E = lc["batch_size"], lc["epochs"]                       # [PAPER] Fig. 4, Table I(B)
    sizes = np.array([len(c) for c in clients])
    order = np.argsort(-sizes, kind="stable")
    C = len(clients)
    nb = np.ceil(sizes[order] / B).astype(int)                 # [GAP-FILL G19] partial last batch
    S = int(nb.max())
    active_at = [(nb > s).sum() for s in range(S)]
    dev = X.device
    idx_lists = [torch.as_tensor(clients[o], device=dev) for o in order]

    flat = global_flat.detach().unsqueeze(0).repeat(C, 1).requires_grad_(True)
    opt = FlatAdam(flat, lc["lr"], tuple(lc["adam_betas"]), lc["adam_eps"])   # [PAPER] Adam
    loss_sum = torch.zeros(C, device=dev)
    n_seen = torch.zeros(C, device=dev)
    for _ in range(E):
        idx = torch.full((C, S * B), -1, dtype=torch.long, device=dev)
        for j, il in enumerate(idx_lists):
            idx[j, :len(il)] = il[torch.randperm(len(il), generator=gen, device=dev)]
        idx = idx.view(C, S, B)
        loss_sum.zero_(); n_seen.zero_()
        for s in range(S):
            a = int(active_at[s])
            bi = idx[:a, s]
            valid = (bi >= 0)
            bc = bi.clamp(min=0)
            logits = functional_forward(layout.views(flat[:a]), X[bc][:, :, None, :],   # [GAP-FILL G3]
                                        mc["gru_layers"], mc["dropout"], True)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[bc].reshape(-1),
                                 reduction="none").view(a, B)                           # [GAP-FILL G4]
            vf = valid.float()
            per = (ce * vf).sum(1) / vf.sum(1)
            per.sum().backward()
            opt.step(a)
            with torch.no_grad():
                loss_sum[:a] += (ce * vf).sum(1)
                n_seen[:a] += vf.sum(1)
    final = torch.empty_like(flat)
    final[torch.as_tensor(order, device=dev)] = flat.detach()
    last_loss = torch.empty_like(loss_sum)
    last_loss[torch.as_tensor(order, device=dev)] = loss_sum / n_seen
    return final, last_loss                                     # original client order


@torch.no_grad()
def predict(flat_rows, layout, X, cfg, chunk=8192):
    """flat_rows [C, total] -> softmax scores [C, n, K] (eval mode, dropout off)."""
    mc = cfg["model"]
    out = []
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk]
        xb = xb[None].expand(flat_rows.shape[0], *xb.shape)[:, :, None, :]
        out.append(torch.softmax(functional_forward(layout.views(flat_rows), xb, mc["gru_layers"],
                                                    mc["dropout"], False), -1))
    return torch.cat(out, 1)


# -------------------------------------------------------------------- run
def run(cfg: dict, seed: int, csv=None, out_dir=None, device=None, rounds=None) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    R = rounds or cfg["fl"]["rounds"]
    out = Path(out_dir) if out_dir else default_out_dir(cfg, seed)
    if rounds and not out_dir:
        out = out.parent / f"{out.name}_smoke{rounds}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = load_prepared(cfg, csv)
    pdir = repo_path(cfg["data"]["partitions_dir"])
    mask_file = pdir / "unseen_test_mask.npy"                   # [GAP-FILL G17] saved once, reused
    if mask_file.exists():
        if not np.array_equal(np.load(mask_file), data.unseen_test_mask):
            raise RuntimeError("unseen-test mask differs from the saved one; preprocessing changed?")
    else:
        np.save(mask_file, data.unseen_test_mask)

    # seeds: only the TRAINING seed varies (model init, batch order, dropout, channel, DQN) [G9]
    ss = np.random.SeedSequence(seed)
    rng_channel, rng_sel = (np.random.default_rng(s) for s in ss.spawn(2))
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    X_tr = torch.as_tensor(data.X_train, device=device)
    y_tr = torch.as_tensor(data.y_train, device=device)
    X_te = torch.as_tensor(data.X_test, device=device)
    y_te = data.y_test
    d, N = X_tr.shape[1], cfg["n_clients"]
    mc = cfg["model"]
    ref = GRUClassifier(d, mc["hidden"], mc["gru_layers"], mc["dropout"], mc["n_classes"])
    names = param_names(mc["gru_layers"])
    layout = Layout(ref.state_dict(), names)
    w_g = layout.flatten(ref.state_dict()).to(device)
    n_par = layout.total
    sizes = np.array([len(c) for c in data.clients])

    channel = Channel(cfg["channel"], N, rng_channel)
    sel = Selector(cfg, N, rng_sel, device)
    k_state = sel.k
    prev_sel = np.zeros(N)
    cdq_sum, cdq_cnt = np.zeros(N), np.zeros(N)
    cdq = np.zeros(N)

    def state(prev, snr_db, cdq_):                              # [PAPER] Eq. 6; scaling [GAP-FILL G6b]
        feats = np.stack([np.full(N, cfg["channel"]["cycles_per_bit"] / 20.0),
                          np.full(N, cfg["channel"]["cpu_hz"] / 1.1e9),
                          prev, snr_db / 100.0, cdq_], 1)
        return feats.reshape(-1).astype(np.float32)

    resolved = copy.deepcopy(cfg)
    resolved.update({"seed": seed, "rounds_run": R, "device": str(device), "fingerprints": data.fingerprints,
                     "dropped_high_cardinality": data.dropped_high_card, "n_params": n_par,
                     "client_sizes": sizes.tolist(), "torch": str(torch.__version__)})
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    log_f = open(out / "log.jsonl", "w")
    summary = {"name": cfg["name"], "seed": seed, "status": "ok", "rounds": {}}

    s_t = state(prev_sel, channel.state_snr_db(k_state), cdq)
    energy_cum = 0.0
    for t in range(1, R + 1):
        tr0 = time.time()
        chosen, pidx, explored = sel.select(s_t, t)
        p_w = sel.power_w(pidx)
        local, loss = local_train(w_g, layout, [data.clients[i] for i in chosen], X_tr, y_tr, cfg, gen)

        n_sel = torch.as_tensor(sizes[chosen], dtype=torch.float32, device=device)
        w_new = (local * n_sel[:, None]).sum(0) / n_sel.sum()  # [PAPER] Eqs. 18, 25
        if not (torch.isfinite(w_new).all() and torch.isfinite(loss).all()):
            summary["status"] = f"diverged at round {t} (non-finite weights or loss)"
            log_f.write(json.dumps({"round": t, "status": summary["status"]}) + "\n")
            print(summary["status"], flush=True)
            break
        w_g = w_new

        # [GAP-FILL G7] C_dq: running mean over participations of ||w_i - w_G|| / Delta
        delta = float(n_sel.sum())
        norms = (local - w_g[None]).norm(dim=1).cpu().numpy()
        cdq_sum[chosen] += norms / delta
        cdq_cnt[chosen] += 1
        cdq = np.where(cdq_cnt > 0, cdq_sum / np.maximum(cdq_cnt, 1), 0.0)

        # [PAPER] Eqs. 1-4 energy; constants [GAP-FILL G10-G12]
        e_data, e_comp, e_comm = channel.energy(sizes[chosen], d, n_par, p_w, len(chosen),
                                                channel.gain[chosen])
        e_dev = e_data + e_comp + e_comm
        c_training = float(e_dev.sum() / N)                    # [PAPER] Eq. 4 (0 J for unselected)
        energy_cum += c_training
        reward = -float(np.mean(c_training + cdq[chosen]))     # [PAPER] Eq. 8

        prev_sel = np.zeros(N); prev_sel[chosen] = 1
        channel.draw()
        s_next = state(prev_sel, channel.state_snr_db(k_state), cdq)
        sel.observe(s_t, chosen, pidx, reward, s_next, done=(t == R))
        s_t = s_next

        scores_g = predict(w_g[None], layout, X_te, cfg)[0].cpu().numpy()
        pred_g = scores_g.argmax(1)
        acc_g = float((pred_g == y_te).mean())
        rec = {"round": t, "selected": chosen.tolist(), "power_dbm": (10 * np.log10(p_w) + 30).round(1).tolist(),
               "explored": bool(explored), "epsilon": sel.epsilon(t) if sel.method == "dqn" else None,
               "train_loss": loss.cpu().numpy().round(5).tolist(), "global_test_acc": acc_g,
               "pred_class_share_max": float(np.bincount(pred_g, minlength=10).max() / len(pred_g)),
               "E_data": float(e_data.sum()), "E_comp": float(e_comp.sum()), "E_comm": float(e_comm.sum()),
               "C_training": c_training, "energy_cum": energy_cum, "cdq_selected": cdq[chosen].tolist(),
               "reward": reward, "dqn_updates": sel.updates,
               "dqn_loss": sel.losses[-1] if sel.losses else None, "sec": round(time.time() - tr0, 2)}

        if t in cfg["fl"]["eval_rounds"] or t == R:
            # [GAP-FILL G14] best/worst client: each selected client's local model on the test set
            sc_loc = predict(local, layout, X_te, cfg).argmax(-1).cpu().numpy()
            acc_loc = (sc_loc == y_te[None]).mean(1)
            m_all = evaluate(y_te, pred_g, scores_g, CLASS_NAMES)
            m_uns = evaluate(y_te, pred_g, scores_g, CLASS_NAMES, mask=data.unseen_test_mask)  # [GAP-FILL G17]
            res = {"round": t, "global": m_all, "global_unseen_subset": m_uns,
                   "client_acc": dict(zip(map(int, chosen), map(float, acc_loc))),
                   "best_client_acc": float(acc_loc.max()), "worst_client_acc": float(acc_loc.min()),
                   "energy_cum": energy_cum, "C_training_round": c_training}
            (out / f"metrics_round{t}.json").write_text(json.dumps(res, indent=1))
            np.savez_compressed(out / f"preds_round{t}.npz", y_true=y_te, y_pred=pred_g,
                                y_score=scores_g.astype(np.float32))
            summary["rounds"][t] = {"acc": m_all["accuracy"], "macro_f1": m_all["macro"]["f1"],
                                    "auc_macro": m_all["auc"]["macro_ovr"],
                                    "best": res["best_client_acc"], "worst": res["worst_client_acc"],
                                    "unseen_acc": m_uns["accuracy"], "energy_cum": energy_cum}
            rec["eval"] = summary["rounds"][t]
            if t in cfg["fl"]["save_model_rounds"] or t == R:
                torch.save(layout.state_dict(w_g.cpu()), out / f"global_round{t}.pt")
                plot_roc(y_te, scores_g, out / f"roc_round{t}.png", f"{cfg['name']} seed {seed} round {t}")
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        print(f"[{cfg['name']} s{seed}] round {t:3d} acc {acc_g:.4f} loss {float(loss.mean()):.4f} "
              f"maxshare {rec['pred_class_share_max']:.2f} sel {chosen.tolist()} "
              f"{'(explore)' if explored else ''} {rec['sec']}s", flush=True)
        if rec["pred_class_share_max"] > 0.99 and t >= 3:
            summary.setdefault("warnings", []).append(f"round {t}: >99% of test predicted as one class")

    summary["minutes"] = round((time.time() - t0) / 60, 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log_f.close()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv")
    ap.add_argument("--rounds", type=int, help="smoke test: run only N rounds (separate output folder)")
    ap.add_argument("--out")
    ap.add_argument("--device")
    a = ap.parse_args()
    s = run(load_config(a.config), a.seed, a.csv, a.out, a.device, a.rounds)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
