"""Section 4.3 metrics.

Per class (one-vs-rest): accuracy (TP+TN)/(TP+TN+FP+FN), precision TP/(TP+FP), recall TP/(TP+FN),
F1 = 2PR/(P+R) (Tables 3-9). The per-dataset row of Table 10 is the unweighted mean of the per-class rows:
this was checked against the paper (mean of Table 3 columns = Table 10 TON-IoT row: 99.63/96.48/93.18/94.45).
"""
import numpy as np


def per_class_metrics(y_true, y_pred, class_names):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    rows = []
    for k, name in enumerate(class_names):
        t, p = y_true == k, y_pred == k
        tp = int(np.sum(t & p)); fp = int(np.sum(~t & p)); fn = int(np.sum(t & ~p)); tn = n - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": name, "support": int(np.sum(t)), "A": 100 * (tp + tn) / n,
                     "P": 100 * prec, "R": 100 * rec, "F1": 100 * f1})
    return rows


def overall_from_per_class(rows):
    """Table 10 row: mean over classes of A, P, R, F1 (classes with no test support are excluded)."""
    rows = [r for r in rows if r["support"] > 0]
    return {m: float(np.mean([r[m] for r in rows])) for m in ("A", "P", "R", "F1")}
