"""Model-independent evaluation. [EXTRA] X5.

Takes (y_true, y_pred, y_score) from ANY model and returns the full metric set, so the same code
scores this reimplementation and any other model later.

[PAPER] Eqs. 31-34 (accuracy, recall, precision, F1), Fig. 7 (ROC/AUC), Table V (per-class and
macro P/R/F1). Multiclass versions are [GAP-FILL G13]: accuracy = fraction correct; macro =
unweighted mean over classes; AUC = one-vs-rest on scores (macro, per class, micro).

CLI:
  python kaur2024/src/evaluate.py preds.npz [--mask unseen_test_mask.npy] [--out metrics.json] [--roc roc.png]
  where preds.npz holds y_true [n], y_pred [n], y_score [n, n_classes].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

DEFAULT_CLASSES = ["Normal", "Scanning", "DoS", "DDoS", "Ransomware", "Backdoor",
                   "Injection", "XSS", "Password", "MITM"]


def evaluate(y_true, y_pred, y_score=None, class_names=DEFAULT_CLASSES, mask=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if mask is not None:
        mask = np.asarray(mask, bool)
        y_true, y_pred = y_true[mask], y_pred[mask]
        y_score = None if y_score is None else np.asarray(y_score)[mask]
    if len(y_true) == 0:
        return {"n": 0, "accuracy": None}
    K = len(class_names)
    labels = np.arange(K)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(class_names)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    present = s > 0
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels[present],
                                                        average=avg, zero_division=0)
        out[avg] = {"precision": float(pa), "recall": float(ra), "f1": float(fa)}
    if y_score is not None:
        y_score = np.asarray(y_score, np.float64)
        onehot = np.eye(K)[y_true]
        per = {}
        for i, c in enumerate(class_names):
            if 0 < onehot[:, i].sum() < len(y_true):
                per[c] = float(roc_auc_score(onehot[:, i], y_score[:, i]))
        out["auc"] = {
            "macro_ovr": float(np.mean(list(per.values()))) if per else None,
            "micro": float(roc_auc_score(onehot[:, present].ravel(), y_score[:, present].ravel())),
            "per_class": per,
        }
    return out


def plot_roc(y_true, y_score, path, title="ROC (micro-average, one-vs-rest)", K=10):
    """[PAPER] Fig. 7 style single ROC curve; micro-average OvR [GAP-FILL G13]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    onehot = np.eye(K)[np.asarray(y_true)]
    fpr, tpr, _ = roc_curve(onehot.ravel(), np.asarray(y_score).ravel())
    auc = roc_auc_score(onehot.ravel(), np.asarray(y_score).ravel())
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC curve (area = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--mask", help="boolean .npy, e.g. unseen_test_mask.npy [GAP-FILL G17]")
    ap.add_argument("--out")
    ap.add_argument("--roc")
    a = ap.parse_args()
    z = np.load(a.preds)
    mask = np.load(a.mask) if a.mask else None
    res = evaluate(z["y_true"], z["y_pred"], z["y_score"] if "y_score" in z else None, mask=mask)
    txt = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)
    if a.roc and "y_score" in z:
        plot_roc(z["y_true"], z["y_score"], a.roc)


if __name__ == "__main__":
    main()
