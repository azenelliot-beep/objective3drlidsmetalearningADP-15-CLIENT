"""Command-line entry point.

    python -m torre2025_fl_cnn_pp.run --dataset ton_iot --data-path /data/Train_Test_Network.csv
    python -m torre2025_fl_cnn_pp.run --dataset synthetic --smoke          # quick end-to-end check

For each dataset two experiments are run (Section 4.4): the FL system without PP and with the three PP
techniques. Results: <results_dir>/<dataset>/<run_tag>/{summary.json, history.jsonl, run.log, checkpoints}.
Finished runs are skipped on re-launch (use --force to recompute).
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

from .config import SMOKE_OVERRIDES, Config
from .data import SPECS, data_fingerprint, prepare
from .paper_results import PAPER_FL, PAPER_OVERALL, PAPER_PER_CLASS


def _setup_logging(path):
    root = logging.getLogger("torre2025")
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        h.setFormatter(fmt)
        root.addHandler(h)
    return root


def _set_global_seeds(cfg):
    import keras
    import tensorflow as tf
    keras.utils.set_random_seed(cfg.seed)
    if cfg.deterministic_ops:
        tf.config.experimental.enable_op_determinism()


def format_tables(dataset, summaries):
    """Markdown: per-class table (Tables 3-9), overall row (Table 10) and FL table (Tables 11/12)."""
    out = [f"## {SPECS[dataset].name}\n"]
    paper_pc = PAPER_PER_CLASS.get(dataset, {})
    for tag, s in summaries.items():
        out.append(f"### {tag}: per-class metrics (%), obtained vs paper (paper Tables 3-9, PP setting not stated by the authors)\n")
        out.append("| Class | support | A | P | R | F1 | paper A | paper P | paper R | paper F1 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in s["per_class"]:
            p = paper_pc.get(r["class"], (None,) * 4)
            ps = [f"{v:.2f}" if v is not None else "-" for v in p]
            out.append(f"| {r['class']} | {r['support']} | {r['A']:.2f} | {r['P']:.2f} | {r['R']:.2f} | "
                       f"{r['F1']:.2f} | " + " | ".join(ps) + " |")
        o, po = s["overall"], PAPER_OVERALL.get(dataset)
        out.append(f"| **Overall (Table 10)** | | **{o['A']:.2f}** | **{o['P']:.2f}** | **{o['R']:.2f}** | "
                   f"**{o['F1']:.2f}** | " + (" | ".join(f"{v:.2f}" for v in po) if po else "- | - | - | -") + " |\n")
        out.append(f"### {tag}: FL performance (Table {'12' if s['use_pp'] else '11'} format)\n")
        out.append("| Client | I# | CCT (s) | TFD (s) | L-A (%) | F-A (%) | paper I# | paper CCT | paper TFD | paper L-A | paper F-A |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        pfl = PAPER_FL.get(dataset, {}).get("pp" if s["use_pp"] else "nopp", {})
        for c in s["clients"]:
            p = pfl.get(c["client"], ("-",) * 5)
            out.append(f"| {c['client']} | {c['I#']}{'' if c['converged'] else ' (not conv.)'} | "
                       f"{c['CCT_mean_s']:.2f} | {c['TFD_mean_s']:.2f} | {c['L-A']:.2f} | {c['F-A']:.2f} | "
                       + " | ".join(str(v) for v in p) + " |")
        out.append("")
    return "\n".join(out)


def run_one(dataset, data_path, cfg, use_pp, tag, force=False):
    from .fl import run_federated
    out_dir = os.path.join(cfg.results_dir, dataset, tag)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path) and not force:
        print(f"[skip] {summary_path} exists (use --force to recompute)")
        with open(summary_path) as fh:
            return json.load(fh)
    log = _setup_logging(os.path.join(out_dir, "run.log"))
    _set_global_seeds(cfg)
    t0 = time.time()
    fd = prepare(dataset, data_path, cfg)
    log.info("data: %s", json.dumps({k: v for k, v in fd.info.items() if k != "preprocessing"}, default=str))
    log.info("features: %d  classes: %s  fingerprint %s", fd.n_features, fd.classes, data_fingerprint(fd))
    res = run_federated(fd, cfg, use_pp, out_dir)
    res.update(dataset=dataset, tag=tag, config=cfg.to_dict(), data_info=fd.info,
               data_fingerprint=data_fingerprint(fd), wall_time_s=time.time() - t0)
    with open(os.path.join(out_dir, "history.jsonl"), "w") as fh:
        for h in res["history"]:
            fh.write(json.dumps(h) + "\n")
    with open(summary_path, "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    log.info("overall %s", res["overall"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(SPECS))
    ap.add_argument("--data-path", default=None, help="CSV file or directory (see README, 'Datasets')")
    ap.add_argument("--pp", choices=["both", "on", "off"], default="both")
    ap.add_argument("--smoke", action="store_true", help=f"smoke-test overrides {SMOKE_OVERRIDES}")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--architecture", choices=["fig4", "algorithm1_literal"], default=None)
    ap.add_argument("--dp-noise-removal", choices=["own_share", "own_full"], default=None)
    ap.add_argument("--dp-scale-mode", choices=["fixed", "sensitivity"], default=None)
    ap.add_argument("--dp-fixed-scale", type=float, default=None)
    ap.add_argument("--federated-arrays", choices=["trainable", "all", "conv"], default=None)
    ap.add_argument("--tolerance", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--independent-init", action="store_true", help="clients initialise their CNNs independently")
    ap.add_argument("--minmax", choices=["on", "off"], default=None, help="override the per-dataset MinMaxScaler choice")
    ap.add_argument("--synthetic-rows", type=int, default=None)
    ap.add_argument("--tag-suffix", default="")
    a = ap.parse_args(argv)
    if a.dataset != "synthetic" and not a.data_path:
        ap.error("--data-path is required for real datasets")
    cfg = Config()
    if a.smoke:
        cfg = cfg.with_overrides(**SMOKE_OVERRIDES)
    cfg = cfg.with_overrides(results_dir=a.results_dir, seed=a.seed, sample_frac=a.sample_frac,
                             max_rows=a.max_rows, architecture=a.architecture,
                             dp_noise_removal=a.dp_noise_removal, dp_scale_mode=a.dp_scale_mode,
                             dp_fixed_scale=a.dp_fixed_scale,
                             federated_arrays=a.federated_arrays, tolerance=a.tolerance,
                             epochs=a.epochs, max_iterations=a.max_iterations,
                             synthetic_rows=a.synthetic_rows, minmax=None if a.minmax is None else a.minmax == "on",
                             common_initialization=False if a.independent_init else None)
    modes = {"both": [False, True], "on": [True], "off": [False]}[a.pp]
    summaries = {}
    for use_pp in modes:
        tag = ("with_pp" if use_pp else "without_pp") + ("_smoke" if a.smoke else "") + a.tag_suffix
        summaries[tag] = run_one(a.dataset, a.data_path, cfg, use_pp, tag, a.force)
    report = format_tables(a.dataset, summaries)
    path = os.path.join(cfg.results_dir, a.dataset, f"report{'_smoke' if a.smoke else ''}{a.tag_suffix}.md")
    with open(path, "w") as fh:
        fh.write(report)
    print(report)
    print(f"report written to {path}")


if __name__ == "__main__":
    main()
