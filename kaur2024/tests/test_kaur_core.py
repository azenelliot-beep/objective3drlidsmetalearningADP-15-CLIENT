"""Correctness tests for the Kaur (2024) reimplementation (synthetic data, CPU).
Run from the repo root:  python -m pytest kaur2024/tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate import evaluate                                    # noqa: E402
from fl import load_config, local_train                          # noqa: E402
from gru_model import FlatAdam, FlatSGD, GRUClassifier, Layout, functional_forward, param_names  # noqa: E402

D = 7


def _models(n, seed=0):
    torch.manual_seed(seed)
    return [GRUClassifier(D, 16, 2, 0.3, 10) for _ in range(n)]


@pytest.mark.parametrize("T", [1, 3])
def test_functional_forward_matches_nn_gru(T):
    ms = _models(2)
    lay = Layout(ms[0].state_dict(), param_names(2))
    flat = torch.stack([lay.flatten(m.state_dict()) for m in ms])
    x = torch.randn(2, 5, T, D)
    got = functional_forward(lay.views(flat), x, 2, 0.3, training=False)
    for c, m in enumerate(ms):
        m.eval()
        torch.testing.assert_close(got[c], m(x[c]), rtol=1e-5, atol=1e-5)


def test_flat_adam_matches_torch_adam():
    (m,) = _models(1)
    lay = Layout(m.state_dict(), param_names(2))
    flat = lay.flatten(m.state_dict())[None].clone().requires_grad_(True)
    opt_f = FlatAdam(flat, 0.01, (0.9, 0.999), 1e-7)
    opt_t = torch.optim.Adam(m.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-7)
    m.eval()
    for i in range(5):
        x, y = torch.randn(4, 1, D), torch.randint(0, 10, (4,))
        opt_t.zero_grad()
        torch.nn.functional.cross_entropy(m(x), y).backward()
        opt_t.step()
        logits = functional_forward(lay.views(flat), x[None], 2, 0.3, False)[0]
        torch.nn.functional.cross_entropy(logits, y).backward()
        opt_f.step(1)
    torch.testing.assert_close(flat.detach()[0], lay.flatten(m.state_dict()), rtol=1e-5, atol=1e-6)


def _cfg():
    cfg = load_config("kaur2024/configs/e1_iid_dqn.yaml")
    cfg["model"].update(hidden=16, dropout=0.0)
    cfg["local"].update(optimizer="adam", lr=0.01, epochs=2)   # these tests exercise the Adam path
    return cfg


def test_parallel_clients_are_independent(monkeypatch):
    """Client A trained together with client B must equal client A trained alone.
    Shuffling is replaced by the identity so both runs see the same batch order (with a shared
    generator the shuffles legitimately differ between the two runs)."""
    import fl
    monkeypatch.setattr(fl.torch, "randperm", lambda n, generator=None, device=None: torch.arange(n, device=device))
    cfg = _cfg()
    m = GRUClassifier(D, 16, 2, 0.0, 10)
    lay = Layout(m.state_dict(), param_names(2))
    w = lay.flatten(m.state_dict())
    X, y = torch.randn(300, D), torch.randint(0, 10, (300,))
    A, B = np.arange(0, 90), np.arange(90, 300)          # uneven sizes on purpose
    out_pair, _ = local_train(w, lay, [A, B], X, y, cfg, torch.Generator())
    out_a, _ = local_train(w, lay, [A], X, y, cfg, torch.Generator())
    out_b, _ = local_train(w, lay, [B], X, y, cfg, torch.Generator())
    torch.testing.assert_close(out_pair[0], out_a[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_pair[1], out_b[0], rtol=1e-5, atol=1e-6)


def test_uneven_clients_respect_epochs():
    """A small client stops updating when its own batches are exhausted (no extra steps)."""
    cfg = _cfg()
    m = GRUClassifier(D, 16, 2, 0.0, 10)
    lay = Layout(m.state_dict(), param_names(2))
    w = lay.flatten(m.state_dict())
    X, y = torch.randn(300, D), torch.randint(0, 10, (300,))
    small, big = np.arange(0, 30), np.arange(30, 300)
    # small client alone vs. with a big one: the generator order differs (big is first), so
    # compare step counts instead: 30 samples / batch 20 = 2 steps x 2 epochs = 4 Adam steps.
    flat_holder = {}
    orig = FlatAdam.__init__

    def spy(self, flat, *a, **k):
        orig(self, flat, *a, **k)
        flat_holder["opt"] = self
    FlatAdam.__init__ = spy
    try:
        local_train(w, lay, [small, big], X, y, cfg, torch.Generator().manual_seed(0))
    finally:
        FlatAdam.__init__ = orig
    steps = flat_holder["opt"].t.squeeze(1).tolist()   # ordered big first
    assert steps == [2 * 14, 2 * 2]


def test_evaluate_basic():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9])
    s = np.eye(10)[y] * 0.9 + 0.01
    r = evaluate(y, y, s)
    assert r["accuracy"] == 1.0 and r["macro"]["f1"] == 1.0 and r["auc"]["macro_ovr"] == 1.0
    yp = y.copy(); yp[0] = 1
    r = evaluate(y, yp, s)
    assert r["per_class"]["Normal"]["recall"] == 0.5
    assert abs(r["accuracy"] - 12 / 13) < 1e-12


def test_learns_a_learnable_task():
    """Sanity check that the training path learns (so a collapse on real data at lr 0.2 can be
    attributed to the learning rate, not to a bug). Uses lr 0.01 here, NOT the paper's 0.2."""
    torch.manual_seed(0)
    cfg = _cfg()
    cfg["local"].update(lr=0.01, epochs=5)
    X = torch.rand(1000, D)
    y = (X[:, 0] * 10).long().clamp(max=9)          # class determined by feature 0
    m = GRUClassifier(D, 16, 2, 0.0, 10)
    lay = Layout(m.state_dict(), param_names(2))
    out, loss = local_train(lay.flatten(m.state_dict()), lay, [np.arange(1000)], X, y, cfg,
                            torch.Generator().manual_seed(0))
    logits = functional_forward(lay.views(out), X[None, :, None, :], 2, 0.0, False)[0]
    assert (logits.argmax(1) == y).float().mean() > 0.8


def test_flat_sgd_matches_torch_sgd():
    (m,) = _models(1)
    lay = Layout(m.state_dict(), param_names(2))
    flat = lay.flatten(m.state_dict())[None].clone().requires_grad_(True)
    opt_f = FlatSGD(flat, 0.2)
    opt_t = torch.optim.SGD(m.parameters(), lr=0.2)
    m.eval()
    for _ in range(5):
        x, y = torch.randn(4, 1, D), torch.randint(0, 10, (4,))
        opt_t.zero_grad()
        torch.nn.functional.cross_entropy(m(x), y).backward()
        opt_t.step()
        logits = functional_forward(lay.views(flat), x[None], 2, 0.3, False)[0]
        torch.nn.functional.cross_entropy(logits, y).backward()
        opt_f.step(1)
    torch.testing.assert_close(flat.detach()[0], lay.flatten(m.state_dict()), rtol=1e-5, atol=1e-6)


def test_sgd_path_learns():
    """The configured path (SGD, eta = 0.2) learns a learnable synthetic task."""
    torch.manual_seed(0)
    cfg = load_config("kaur2024/configs/e1_iid_dqn.yaml")
    assert cfg["local"]["optimizer"] == "sgd" and cfg["local"]["lr"] == 0.2
    cfg["model"].update(hidden=16, dropout=0.0)
    cfg["local"].update(epochs=5)
    y = torch.randint(0, 10, (1000,))
    centers = torch.randn(10, D) * 2
    X = centers[y] + 0.3 * torch.randn(1000, D)          # well-separated clusters
    m = GRUClassifier(D, 16, 2, 0.0, 10)
    lay = Layout(m.state_dict(), param_names(2))
    out, _ = local_train(lay.flatten(m.state_dict()), lay, [np.arange(1000)], X, y, cfg,
                         torch.Generator().manual_seed(0))
    logits = functional_forward(lay.views(out), X[None, :, None, :], 2, 0.0, False)[0]
    assert (logits.argmax(1) == y).float().mean() > 0.8
