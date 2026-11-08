"""DIAGNOSTIC ONLY [EXTRA]; not part of any reported result and not used to tune anything.
Question: is the collapse seen in the round-1/2 smoke test caused by the learning rate?
Setup: one client (IID client_00, 24,766 rows), 1 local epoch, batch 20, the same GRU and training
code as fl.py, global test set. Compares
  - Adam lr 0.2   (paper: Table I(B) Adam + Fig. 4 eta = 0.2; the configured setting)
  - SGD  lr 0.2   (paper: Eqs. 17/24 define eta as a plain gradient-descent step; SPEC C3)
  - Adam lr 1e-3  (framework default, reference only)
Run: python kaur2024/tools/diag_lr_collapse.py --csv kaur2024/data/raw/TON_IoT_Train_Test_Network.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fl                                                       # noqa: E402
from data import load_prepared                                  # noqa: E402
from gru_model import GRUClassifier, Layout, param_names        # noqa: E402


class FlatSGD:
    """Plain gradient descent, Eq. 17 / Eq. 24: w <- w - eta * grad."""
    def __init__(self, flat, lr, *a, **k):
        self.flat, self.lr = flat, lr
        self.t = torch.zeros(flat.shape[0], 1)

    @torch.no_grad()
    def step(self, a):
        self.flat[:a] -= self.lr * self.flat.grad[:a]
        self.t[:a] += 1
        self.flat.grad = None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True)
    cfg = fl.load_config("kaur2024/configs/e1_iid_dqn.yaml")
    data = load_prepared(cfg, ap.parse_args().csv)
    X, y, Xt = map(torch.as_tensor, (data.X_train, data.y_train, data.X_test))
    print(f"d = {X.shape[1]}; dropped (>50 unique in train): {data.dropped_high_card}; "
          f"unseen test rows: {int(data.unseen_test_mask.sum())}")
    print("| optimiser | lr | epoch-1 train loss | test acc | largest predicted-class share |")
    print("|---|---|---|---|---|")
    adam = fl.FlatAdam
    for name, opt, lr in (("Adam", adam, 0.2), ("SGD", FlatSGD, 0.2), ("Adam", adam, 1e-3)):
        torch.manual_seed(0)
        m = GRUClassifier(X.shape[1]); lay = Layout(m.state_dict(), param_names(2))
        cfg["local"].update(lr=lr, epochs=1)
        fl.FlatAdam = opt
        out, loss = fl.local_train(lay.flatten(m.state_dict()), lay, [data.clients[0]], X, y, cfg,
                                   torch.Generator().manual_seed(0))
        fl.FlatAdam = adam
        pred = fl.predict(out, lay, Xt, cfg)[0].argmax(1).numpy()
        share = np.bincount(pred, minlength=10).max() / len(pred)
        print(f"| {name} | {lr:g} | {float(loss[0]):.4f} | {np.mean(pred == data.y_test):.4f} | {share:.3f} |")


if __name__ == "__main__":
    main()
