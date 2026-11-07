"""Data loading and preprocessing for the Kaur (2024) reimplementation on TON_IoT.

[PAPER] Sec. IV-A (p.5-6): remove unwanted features, handle missing values, one-hot encode
categorical features, min-max normalisation (Eq. 19).
Every encoder/scaler decision is fitted on TRAINING rows only (hard rule). The test set is
only transformed.

The split and client partitions are NOT created here: they are read from the files written
once by src/phase1_data.py (partition seed 0, [GAP-FILL G9]) and checked against the CSV's sha256.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# [PAPER] Table II order; keys are the CSV's `type` values.
CLASS_ORDER = ["normal", "scanning", "dos", "ddos", "ransomware", "backdoor",
               "injection", "xss", "password", "mitm"]
CLASS_NAMES = ["Normal", "Scanning", "DoS", "DDoS", "Ransomware", "Backdoor",
               "Injection", "XSS", "Password", "MITM"]


def repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


@dataclass
class Prepared:
    X_train: np.ndarray            # float32 [n_train, d]
    y_train: np.ndarray            # int64   [n_train]
    X_test: np.ndarray
    y_test: np.ndarray
    clients: list                  # list of arrays, positions into X_train
    feature_names: list
    dropped_high_card: list
    unseen_test_mask: np.ndarray   # [GAP-FILL G17] test rows whose encoded features never occur in train
    fingerprints: dict = field(default_factory=dict)


class Preprocessor:
    """Fitted on training rows only. [PAPER] Sec. IV-A; details [GAP-FILL G2, G2b-d]."""

    def __init__(self, drop_cols, target, high_card, clip_test):
        self.drop_cols, self.target = list(drop_cols), target
        self.high_card, self.clip_test = high_card, clip_test

    def fit(self, df_tr: pd.DataFrame):
        cols = [c for c in df_tr.columns if c not in self.drop_cols + [self.target]]
        self.str_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df_tr[c])]
        self.num_cols = [c for c in cols if c not in self.str_cols]
        # [GAP-FILL G2] cardinality counted on TRAIN rows only
        self.dropped = sorted(c for c in self.str_cols if df_tr[c].nunique(dropna=False) > self.high_card)
        self.cat_cols = [c for c in self.str_cols if c not in self.dropped]
        # [GAP-FILL G2b] '-' is its own category; G2c: unseen test categories -> all-zero
        self.categories = {c: sorted(df_tr[c].astype(str).unique()) for c in self.cat_cols}
        num = df_tr[self.num_cols].to_numpy(np.float64)
        self.vmin, self.vmax = num.min(0), num.max(0)   # [PAPER] Eq. 19 on train
        self.feature_names = self.num_cols + [f"{c}={v}" for c in self.cat_cols for v in self.categories[c]]
        return self

    def transform(self, df: pd.DataFrame, is_test: bool) -> np.ndarray:
        num = df[self.num_cols].to_numpy(np.float64)
        rng = np.where(self.vmax > self.vmin, self.vmax - self.vmin, 1.0)
        num = (num - self.vmin) / rng                    # [PAPER] Eq. 19
        num[:, self.vmax <= self.vmin] = 0.0             # constant-in-train column -> 0
        if is_test and self.clip_test:
            num = np.clip(num, 0.0, 1.0)                 # [GAP-FILL G2d]
        blocks = [num.astype(np.float32)]
        for c in self.cat_cols:                          # [PAPER] one-hot encoding
            cats = pd.Categorical(df[c].astype(str), categories=self.categories[c])
            codes = cats.codes
            oh = np.zeros((len(df), len(self.categories[c])), np.float32)
            ok = codes >= 0
            oh[np.flatnonzero(ok), codes[ok]] = 1.0
            blocks.append(oh)
        return np.concatenate(blocks, axis=1)


def _row_keys(X: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    return v.ravel()


def load_prepared(cfg: dict, csv_override: str | None = None, verify_sha: bool = True) -> Prepared:
    dc = cfg["data"]
    csv = Path(csv_override or dc["csv"])
    pdir = repo_path(dc["partitions_dir"])
    manifest = json.loads((pdir / "manifest.json").read_text())
    if verify_sha:
        sha = sha256_file(csv)
        if sha != dc["csv_sha256"] or sha != manifest["csv_sha256"]:
            raise RuntimeError(f"CSV sha256 {sha} does not match config/manifest; wrong file?")
    df = pd.read_csv(csv, low_memory=False)
    y_all = df[dc["target"]].astype(str).str.strip().str.lower().map({c: i for i, c in enumerate(CLASS_ORDER)})
    if y_all.isna().any():
        raise RuntimeError("unknown class label in CSV")
    y_all = y_all.to_numpy(np.int64)

    tr = np.load(pdir / "split_train_idx.npy")
    te = np.load(pdir / "split_test_idx.npy")
    assert len(np.intersect1d(tr, te)) == 0
    part_file = {"iid": "iid_clients.npz", "noniid_dir0.5": "noniid_dir0.5_clients.npz"}[cfg["partition"]]
    z = np.load(pdir / part_file)
    parts = [z[f"client_{k:02d}"] for k in range(cfg["n_clients"])]
    # positions of each client's CSV rows inside the training arrays
    pos = np.full(len(df), -1, np.int64)
    pos[tr] = np.arange(len(tr))
    clients = [pos[p] for p in parts]
    assert all((c >= 0).all() for c in clients), "client index outside the training split"
    assert sum(len(c) for c in clients) == len(tr)

    pre = Preprocessor(dc["drop_cols"], dc["target"], dc["high_cardinality"], dc["clip_test"])
    pre.fit(df.iloc[tr])
    X_tr = pre.transform(df.iloc[tr], is_test=False)
    X_te = pre.transform(df.iloc[te], is_test=True)

    # [GAP-FILL G17] "unseen" test rows: encoded feature vector absent from train (no labels used)
    unseen = ~np.isin(_row_keys(X_te), _row_keys(X_tr))

    fp = {"csv_sha256": manifest["csv_sha256"], "train_idx": sha256_array(tr), "test_idx": sha256_array(te),
          "partition": sha256_array(np.concatenate(parts)), "d": int(X_tr.shape[1]),
          "n_unseen_test": int(unseen.sum())}
    return Prepared(X_tr, y_all[tr], X_te, y_all[te], clients, pre.feature_names, pre.dropped, unseen, fp)
