"""Aggregate summary.json files of one dataset into seed-averaged tables and an ablation table.

    python -m torre2025_fl_cnn_pp.summarize <results_dir> <dataset> [--out report.md]
"""
import argparse
import glob
import json
import os

import numpy as np


def _load(results_dir, dataset):
    runs = {}
    for f in sorted(glob.glob(os.path.join(results_dir, dataset, "*", "summary.json"))):
        with open(f) as fh:
            runs[os.path.basename(os.path.dirname(f))] = json.load(fh)
    return runs


def _ms(vals):
    vals = np.asarray(vals, float)
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{np.nanmean(vals):.2f} ± {np.nanstd(vals, ddof=1):.2f}"


def _fl_stats(run):
    h = run["history"]
    fa = [x["fed_acc"] for x in h]
    return {"I#": np.mean([c["I#"] for c in run["clients"]]),
            "converged": sum(c["converged"] for c in run["clients"]),
            "CCT": np.mean([c["CCT_mean_s"] for c in run["clients"]]),
            "TFD": np.mean([c["TFD_mean_s"] for c in run["clients"]]),
            "L-A": np.mean([c["L-A"] for c in run["clients"]]),
            "F-A": np.nanmean([c["F-A"] for c in run["clients"]]),
            "nan_rounds": int(np.sum(~np.isfinite(fa))),
            "min_max_dW": min(x["max_abs_diff_weights"] for x in h)}


def report(results_dir, dataset):
    runs = _load(results_dir, dataset)
    main = {m: [r for t, r in runs.items() if t.startswith(m + "_seed")] for m in ("without_pp", "with_pp")}
    out = [f"# Results: {dataset} ({results_dir})\n"]
    if all(main.values()):
        seeds = sorted({r["config"]["seed"] for rs in main.values() for r in rs})
        out.append(f"## Main configuration, seeds {seeds} (mean ± sample std)\n")
        classes = [r["class"] for r in main["without_pp"][0]["per_class"]]
        out.append("| Class | without PP: A / P / R / F1 | with PP: A / P / R / F1 |\n|---|---|---|")
        for k, c in enumerate(classes + ["Overall (Table 10)"]):
            cells = []
            for m in ("without_pp", "with_pp"):
                rows = [r["per_class"][k] if k < len(classes) else r["overall"] for r in main[m]]
                cells.append(" / ".join(_ms([x[q] for x in rows]) for q in ("A", "P", "R", "F1")))
            out.append(f"| {c} | " + " | ".join(cells) + " |")
        out.append("\n## FL performance (Tables 11/12 format, averaged over clients, then seeds)\n")
        out.append("| Setting | I# | clients converged | CCT (s) | TFD (s) | L-A (%) | F-A (%) | rounds with NaN F-A |")
        out.append("|---|---|---|---|---|---|---|---|")
        st = {m: [_fl_stats(r) for r in rs] for m, rs in main.items()}
        for m in ("without_pp", "with_pp"):
            s = st[m]
            out.append(f"| {m} | {_ms([x['I#'] for x in s])} | {sum(x['converged'] for x in s)}/{4 * len(s)} | "
                       f"{_ms([x['CCT'] for x in s])} | {_ms([x['TFD'] for x in s])} | {_ms([x['L-A'] for x in s])} | "
                       f"{_ms([x['F-A'] for x in s])} | {sum(x['nan_rounds'] for x in s)} |")
        for q in ("CCT", "TFD"):
            a = np.mean([x[q] for x in st["without_pp"]]); b = np.mean([x[q] for x in st["with_pp"]])
            out.append(f"\nPP overhead on mean {q}: {b - a:+.2f} s ({100 * (b - a) / a:+.1f} %)  "
                       f"[paper: about +12 % CCT, +8 % TFD averaged over datasets]")
    abl = {t: r for t, r in runs.items() if "_abl_" in t}
    if abl:
        out.append("\n## Ablations of the ambiguous choices (seed 42; one change each vs the main configuration)\n")
        out.append("| Run | overall A | overall F1 | final F-A (%) | min over rounds of max\\|dW\\| | clients converged | rounds with NaN F-A |")
        out.append("|---|---|---|---|---|---|---|")
        ref = {t: r for t, r in runs.items() if t.endswith("_seed42")}
        for t, r in sorted({**ref, **abl}.items()):
            s = _fl_stats(r)
            out.append(f"| {t} | {r['overall']['A']:.2f} | {r['overall']['F1']:.2f} | {s['F-A']:.2f} | "
                       f"{s['min_max_dW']:.2e} | {s['converged']}/4 | {s['nan_rounds']} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("dataset")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text = report(a.results_dir, a.dataset)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
