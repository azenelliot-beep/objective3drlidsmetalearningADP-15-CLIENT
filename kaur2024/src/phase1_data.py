"""Phase 1 data work for the Kaur (2024) reimplementation on TON_IoT. No training code.

Does, in order:
  1. Inspect the CSV: shape, dtypes, missing values (NaN and the '-' placeholder),
     class counts of `type` vs Table II [PAPER, p.8].
  2. X3 [EXTRA] cardinality report for every string column.
  3. G1/G9 [GAP-FILL] create the train/test split ONCE with partition seed 0,
     using the per-class counts of Table II [PAPER, p.8].
  4. G8/G9 [GAP-FILL] create the IID and the non-IID (Dirichlet alpha=0.5) client
     partitions of the TRAIN indices, N=15 clients, partition seed 0. Plot them.
  5. X3 [EXTRA] duplicate rows shared between train and test.

Indices saved are 0-based row positions in the CSV (header excluded). The CSV's
sha256 is stored so every later run can check it uses the same file.

The test set is never used for any decision here: the split is made from labels
only, partitions use train indices only, and X3 only *reports* overlap.

Usage:
  python kaur2024/src/phase1_data.py --csv <path/to/train_test_network.csv>
Outputs (all under kaur2024/results/, never the repo's top-level results/):
  partitions/  split_train_idx.npy, split_test_idx.npy, iid_clients.npz,
               noniid_dir0.5_clients.npz, client_class_counts_*.csv, manifest.json,
               class_distribution_*.png
  phase1_data_report.md
  extra/x3_leakage.md
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]           # kaur2024/
RESULTS = ROOT / "results"
PART_DIR = RESULTS / "partitions"
EXTRA_DIR = RESULTS / "extra"

TARGET = "type"                                       # [GAP-FILL G2] target column
# [GAP-FILL G2] dropped before modelling (identifiers / timestamp / binary label)
DROP_COLS = ["ts", "src_ip", "dst_ip", "src_port", "dst_port", "label"]
HIGH_CARD = 50                                        # [GAP-FILL G2] >50 unique -> drop

# [PAPER] Table II, p.8 (TON_IoT). Keys are the CSV's `type` values.
TABLE_II = {  # type value: (paper name, train, test)
    "normal": ("Normal", 245000, 55000),
    "scanning": ("Scanning", 20000, 3973),
    "dos": ("DoS", 20000, 4000),
    "ddos": ("DDoS", 20000, 4000),
    "ransomware": ("Ransomware", 16030, 3970),
    "backdoor": ("Backdoor", 20000, 4000),
    "injection": ("Injection Attack", 20000, 4000),
    "xss": ("XSS", 13844, 6116),
    "password": ("Password Violations", 20000, 4000),
    "mitm": ("MITM", 593, 459),
}
CLASS_ORDER = list(TABLE_II)

PARTITION_SEED = 0          # [GAP-FILL G9]
N_CLIENTS = 15              # [PAPER] Table VIII protocol (see SPEC.md: Table VIII is Edge-IIoT)
DIRICHLET_ALPHA = 0.5       # [GAP-FILL G8]
MIN_CLIENT_SIZE = 20        # [GAP-FILL G8b] >= one local batch (batch size 20, Fig. 4)
MAX_DIRICHLET_TRIES = 1000  # [GAP-FILL G8b]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def md_table(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- split (G1, G9)
def make_split(y, rng):
    """[GAP-FILL G1] Per class: draw exactly Table II's test count at random, then up to
    Table II's train count from the remaining rows. If a class has too few rows, the
    test count is kept and train gets the remainder (shortfall reported). Rows beyond
    Table II's totals are left unused (reported)."""
    tr, te, rows = [], [], []
    for c in CLASS_ORDER:
        name, n_tr, n_te = TABLE_II[c]
        idx = np.flatnonzero(y == c)
        idx = idx[rng.permutation(len(idx))]
        k_te = min(n_te, len(idx))
        k_tr = min(n_tr, len(idx) - k_te)
        te.append(idx[:k_te])
        tr.append(idx[k_te:k_te + k_tr])
        rows.append({"class": name, "in_file": len(idx), "paper_train": n_tr, "ours_train": k_tr,
                     "paper_test": n_te, "ours_test": k_te,
                     "unused": len(idx) - k_te - k_tr,
                     "short_train": n_tr - k_tr, "short_test": n_te - k_te})
    unknown = sorted(set(np.unique(y)) - set(CLASS_ORDER))
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te)), pd.DataFrame(rows), unknown


# ------------------------------------------------------------ partitions (G8, G9)
def iid_partition(train_idx, rng):
    perm = train_idx[rng.permutation(len(train_idx))]
    return [np.sort(p) for p in np.array_split(perm, N_CLIENTS)]


def dirichlet_partition(train_idx, y, rng):
    """[GAP-FILL G8] label-skew Dirichlet split (per class, client shares ~ Dir(alpha)).
    [GAP-FILL G8b] redraw until every client has >= MIN_CLIENT_SIZE samples."""
    for attempt in range(1, MAX_DIRICHLET_TRIES + 1):
        parts = [[] for _ in range(N_CLIENTS)]
        for c in CLASS_ORDER:
            idx = train_idx[y[train_idx] == c]
            idx = idx[rng.permutation(len(idx))]
            p = rng.dirichlet(np.full(N_CLIENTS, DIRICHLET_ALPHA))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].append(chunk)
        parts = [np.sort(np.concatenate(p)) for p in parts]
        if min(len(p) for p in parts) >= MIN_CLIENT_SIZE:
            return parts, attempt
    raise RuntimeError("Dirichlet partition: no draw met the minimum client size")


def client_counts(parts, y):
    m = pd.DataFrame([[int((y[p] == c).sum()) for c in CLASS_ORDER] for p in parts],
                     columns=[TABLE_II[c][0] for c in CLASS_ORDER])
    m.index = [f"client_{k:02d}" for k in range(len(parts))]
    m["total"] = m.sum(axis=1)
    return m


def plot_counts(m, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = m.drop(columns="total")
    frac = data.div(data.sum(axis=1), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, d, lab in [(axes[0], data, "samples"), (axes[1], frac, "fraction of client data")]:
        bottom = np.zeros(len(d))
        cmap = plt.get_cmap("tab10")
        for j, col in enumerate(d.columns):
            ax.bar(range(len(d)), d[col].values, bottom=bottom, color=cmap(j), label=col)
            bottom += d[col].values
        ax.set_xticks(range(len(d)), [str(i + 1) for i in range(len(d))])
        ax.set_xlabel("client"); ax.set_ylabel(lab)
    axes[1].legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- X3 leakage
def duplicate_report(df, tr, te, feat_cols):
    """Rows of the test set whose feature vector also occurs in the training set."""
    out = []
    for label, cols in [("all columns except `type`/`label`",
                         [c for c in df.columns if c not in (TARGET, "label")]),
                        ("model features (after G2 drops, before encoding)", feat_cols)]:
        key = pd.util.hash_pandas_object(df[cols].astype(str), index=False).values
        ktr, kte = key[tr], key[te]
        shared = np.isin(kte, ktr)
        # label conflict: same features, a different `type` in train
        tr_map = pd.DataFrame({"k": ktr, "y": df[TARGET].values[tr]}).groupby("k")["y"].agg(set)
        conflict = sum(1 for k, yv in zip(kte[shared], df[TARGET].values[te][shared])
                       if tr_map[k] != {yv})
        per_class = pd.Series(df[TARGET].values[te][shared]).value_counts()
        out.append((label, int(shared.sum()), len(te), conflict, per_class,
                    int(pd.Series(ktr).duplicated().sum()), int(pd.Series(kte).duplicated().sum())))
    return out


def main():
    global RESULTS, PART_DIR, EXTRA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=str(RESULTS), help="results root (default kaur2024/results)")
    args = ap.parse_args()
    csv = Path(args.csv)
    RESULTS = Path(args.out); PART_DIR = RESULTS / "partitions"; EXTRA_DIR = RESULTS / "extra"
    PART_DIR.mkdir(parents=True, exist_ok=True)
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv, low_memory=False)
    y = df[TARGET].astype(str).str.strip().str.lower().values
    rep = [f"# Phase 1 data report: `{csv.name}`", "",
           f"- sha256: `{sha256(csv)}`", f"- shape: {df.shape[0]:,} rows x {df.shape[1]} columns", ""]

    # 1. dtypes and missing values
    obj = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]  # string columns
    miss = pd.DataFrame({
        "column": df.columns, "dtype": [str(t) for t in df.dtypes],
        "n_nan": [int(df[c].isna().sum()) for c in df.columns],
        "n_dash": [int((df[c].astype(str).str.strip() == "-").sum()) if c in obj else 0
                   for c in df.columns]})
    rep += ["## Columns, dtypes, missing values",
            "`n_dash` counts the '-' placeholder TON_IoT uses for 'not applicable'.", "",
            md_table(miss), ""]

    # 2. X3 cardinality
    card = pd.DataFrame({"column": obj,
                         "n_unique": [int(df[c].nunique(dropna=False)) for c in obj],
                         "top_values": [", ".join(map(str, df[c].value_counts().index[:5]))
                                        for c in obj]})
    card["G2 action"] = ["drop (G2 list)" if c in DROP_COLS else "target" if c == TARGET
                         else f"drop (>{HIGH_CARD} unique)" if n > HIGH_CARD else "one-hot"
                         for c, n in zip(card.column, card.n_unique)]
    feat_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET]
                 and not (c in obj and df[c].nunique(dropna=False) > HIGH_CARD)]
    # NOTE: cardinality here is counted on the whole file only to *report* it; the
    # rule is re-applied on training rows only in Phase 2 (src/data.py).

    # 3. split
    rng = np.random.default_rng(PARTITION_SEED)
    tr, te, split_tab, unknown = make_split(y, rng)
    np.save(PART_DIR / "split_train_idx.npy", tr)
    np.save(PART_DIR / "split_test_idx.npy", te)
    rep += ["## Class counts vs Table II [PAPER, p.8] and the split [GAP-FILL G1]", "",
            md_table(split_tab), "",
            f"Totals: train {len(tr):,} (paper 395,467), test {len(te):,} (paper 89,518).",
            f"`type` values not in Table II: {unknown or 'none'}", ""]

    # 4. partitions
    iid = iid_partition(tr, rng)
    non, tries = dirichlet_partition(tr, y, rng)
    np.savez_compressed(PART_DIR / "iid_clients.npz", **{f"client_{k:02d}": p for k, p in enumerate(iid)})
    np.savez_compressed(PART_DIR / f"noniid_dir{DIRICHLET_ALPHA}_clients.npz",
                        **{f"client_{k:02d}": p for k, p in enumerate(non)})
    for tag, parts in [("iid", iid), (f"noniid_dir{DIRICHLET_ALPHA}", non)]:
        m = client_counts(parts, y)
        m.to_csv(PART_DIR / f"client_class_counts_{tag}.csv")
        plot_counts(m, f"TON_IoT train set, {N_CLIENTS} clients, {tag}",
                    PART_DIR / f"class_distribution_{tag}.png")
        rep += [f"## Client partition: {tag}", "", md_table(m.reset_index()), ""]
    rep += [f"Dirichlet draws needed to reach min client size {MIN_CLIENT_SIZE}: {tries}", ""]

    manifest = {"csv_name": csv.name, "csv_sha256": sha256(csv), "n_rows": int(len(df)),
                "partition_seed": PARTITION_SEED, "n_clients": N_CLIENTS,
                "dirichlet_alpha": DIRICHLET_ALPHA, "min_client_size": MIN_CLIENT_SIZE,
                "dirichlet_tries": tries, "index_base": "0-based CSV row position, header excluded",
                "rng": "numpy.default_rng(0); split, then IID, then Dirichlet, in that order"}
    (PART_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 5. X3 duplicates
    x3 = ["# X3 [EXTRA] Leakage checks", "", f"File `{csv.name}`, split from partition seed 0.", "",
          "## Cardinality of every string column (whole file, report only)", "",
          md_table(card), "", "## Duplicate rows shared between train and test", ""]
    for label, n_sh, n_te, conf, per_cls, d_tr, d_te in duplicate_report(df, tr, te, feat_cols):
        x3 += [f"### Key: {label}",
               f"- test rows whose key also occurs in train: {n_sh:,} / {n_te:,} ({100 * n_sh / n_te:.2f}%)",
               f"- of those, with a different `type` in train: {conf:,}",
               f"- duplicates within train: {d_tr:,}; within test: {d_te:,}",
               "- shared test rows by class: " + ", ".join(f"{k} {v:,}" for k, v in per_cls.items()), ""]
    (EXTRA_DIR / "x3_leakage.md").write_text("\n".join(x3) + "\n")
    (RESULTS / "phase1_data_report.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    print("\n".join(x3))


if __name__ == "__main__":
    main()
