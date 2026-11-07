"""Shared helpers for run_core.py, run_x1_seeds.py and run_x2_random.py: launch runs as parallel
subprocesses (spread over the visible GPUs), skip runs that already finished, and build tables.
[EXTRA] infrastructure only; no modelling choices live here."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from data import CLASS_NAMES, REPO, repo_path   # noqa: E402
from fl import default_out_dir, load_config     # noqa: E402


def run_dir(config_name: str, seed: int) -> Path:
    return default_out_dir(load_config(f"kaur2024/configs/{config_name}.yaml"), seed)


def is_done(config_name, seed) -> bool:
    f = run_dir(config_name, seed) / "summary.json"
    return f.exists() and json.loads(f.read_text()).get("status") == "ok"


def launch(jobs: list[tuple[str, int]], parallel: int, csv: str | None, rounds: int | None = None):
    """jobs: (config_name, seed). Runs at most `parallel` at a time, round-robin over GPUs."""
    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except Exception:
        n_gpu = 0
    todo = [j for j in jobs if rounds or not is_done(*j)]
    print(f"{len(jobs) - len(todo)} already done, {len(todo)} to run, {parallel} in parallel, {n_gpu} GPU(s)")
    logs = repo_path("kaur2024/results/logs"); logs.mkdir(parents=True, exist_ok=True)
    running, k = [], 0
    while todo or running:
        while todo and len(running) < parallel:
            name, seed = todo.pop(0)
            env = dict(os.environ)
            if n_gpu:
                env["CUDA_VISIBLE_DEVICES"] = str(k % n_gpu)
            k += 1
            cmd = [sys.executable, str(SRC / "fl.py"), "--config", f"kaur2024/configs/{name}.yaml",
                   "--seed", str(seed)] + (["--csv", csv] if csv else []) + (["--rounds", str(rounds)] if rounds else [])
            log = open(logs / f"{name}_seed{seed}{'_smoke' if rounds else ''}.log", "w")
            print("start", name, "seed", seed, flush=True)
            running.append((name, seed, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(5)
        for item in list(running):
            name, seed, p, log = item
            if p.poll() is not None:
                log.close(); running.remove(item)
                print("done ", name, "seed", seed, "exit", p.returncode, flush=True)


def load_round(config_name, seed, rnd):
    d = run_dir(config_name, seed)
    f = d / f"metrics_round{rnd}.json"
    if not f.exists():
        return None
    s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    m = json.loads(f.read_text())
    m["status"] = s.get("status")
    return m


def scalar_metrics(m: dict) -> dict:
    g = m["global"]
    out = {"acc": g["accuracy"], "macro_P": g["macro"]["precision"], "macro_R": g["macro"]["recall"],
           "macro_F1": g["macro"]["f1"], "AUC": g["auc"]["macro_ovr"], "best_client": m["best_client_acc"],
           "worst_client": m["worst_client_acc"], "energy_cum_J": m["energy_cum"],
           "unseen_acc": m["global_unseen_subset"].get("accuracy"),
           "unseen_macro_F1": (m["global_unseen_subset"].get("macro") or {}).get("f1")}
    for c in CLASS_NAMES:
        for k in ("precision", "recall", "f1"):
            out[f"{c}_{k[0].upper() if k != 'f1' else 'F1'}"] = g["per_class"][c][k]
    return out


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}"


def mean_std_table(configs, seeds, rounds, keys, labels=None) -> list[str]:
    lines = ["| config | round | n seeds | " + " | ".join(labels or keys) + " |",
             "|---|---|---|" + "---|" * len(keys)]
    for c in configs:
        for r in rounds:
            vals = [scalar_metrics(m) for s in seeds if (m := load_round(c, s, r)) and m["status"] == "ok"]
            if not vals:
                lines.append(f"| {c} | {r} | 0 | " + " | ".join("n/a" for _ in keys) + " |"); continue
            cells = []
            for k in keys:
                v = np.array([x[k] for x in vals if x[k] is not None], float)
                scale = 1 if k == "energy_cum_J" else 100
                cells.append("n/a" if not len(v) else
                             f"{scale * v.mean():.2f} ± {scale * v.std(ddof=1) if len(v) > 1 else 0:.2f}")
            lines.append(f"| {c} | {r} | {len(vals)} | " + " | ".join(cells) + " |")
    return lines
