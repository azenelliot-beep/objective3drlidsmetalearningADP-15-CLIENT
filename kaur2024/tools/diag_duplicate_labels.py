"""DIAGNOSTIC [EXTRA], report only (no training or tuning decision uses it).
For each test row whose encoded feature vector also occurs in train: which labels does train carry
for that exact vector? Gives the accuracy ceiling of a perfect 'lookup' on duplicated rows, and the
per-class picture. Run: python kaur2024/tools/diag_duplicate_labels.py --csv <csv>"""
import argparse, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import CLASS_NAMES, load_prepared, _row_keys      # noqa: E402
from fl import load_config                                   # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True)
d = load_prepared(load_config("kaur2024/configs/e1_iid_dqn.yaml"), ap.parse_args().csv)
ktr, kte = _row_keys(d.X_train), _row_keys(d.X_test)
lab = {}
for k, y in zip(ktr, d.y_train):
    lab.setdefault(k.tobytes(), Counter())[y] += 1
print("| class | test rows | duplicated in train | of those: train majority = same class | train majority = Normal | lookup-ceiling recall |")
print("|---|---|---|---|---|---|")
tot_ok = 0
for c, name in enumerate(CLASS_NAMES):
    idx = np.flatnonzero(d.y_test == c)
    dup = same = norm = 0
    for i in idx:
        cnt = lab.get(kte[i].tobytes())
        if cnt:
            dup += 1
            maj = cnt.most_common(1)[0][0]
            same += maj == c; norm += maj == 0
    tot_ok += same
    print(f"| {name} | {len(idx)} | {dup} | {same} | {norm} | {same/len(idx):.3f} |")
print(f"\nLookup on duplicated rows alone (unseen rows counted wrong): accuracy {tot_ok/len(d.y_test):.4f}")
