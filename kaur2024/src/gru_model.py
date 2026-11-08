"""GRU classifier. [PAPER] Table I(B) p.7 and Sec. IV-B p.6: input layer, 2 GRU layers with
128 hidden units, dropout 0.3, one fully connected output layer, Adam.

Two implementations of the SAME network, sharing one parameter layout (the names of
torch.nn.GRU / nn.Linear):

* `GRUClassifier`: a plain nn.Module (nn.GRU + nn.Linear). Used as the reference, for saved
  checkpoints and for evaluation.
* `functional_forward`: the same GRU equations written out for a stack of C models at once,
  shape [C, ...]. This lets all selected clients train in parallel on one GPU, with exactly
  the maths of nn.GRU (checked numerically in tests/test_kaur_core.py). It is an
  implementation speed-up only; each client's training is mathematically independent and
  identical to training it alone.

Gate order and equations follow torch.nn.GRU:
  r = sigma(W_ir x + b_ir + W_hr h + b_hr)
  z = sigma(W_iz x + b_iz + W_hz h + b_hz)
  n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
  h' = (1 - z) * n + z * h,   h_0 = 0
Dropout [GAP-FILL G4]: on the output of GRU layer 1 (as nn.GRU's `dropout` argument does) and
on the last hidden state before the FC layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUClassifier(nn.Module):
    def __init__(self, d, hidden=128, layers=2, dropout=0.3, n_classes=10):
        super().__init__()
        self.gru = nn.GRU(d, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):                       # x: [B, T, d]
        out, _ = self.gru(x)
        return self.fc(self.drop(out[:, -1]))   # logits


def param_names(layers=2):
    names = []
    for l in range(layers):
        names += [f"gru.weight_ih_l{l}", f"gru.weight_hh_l{l}", f"gru.bias_ih_l{l}", f"gru.bias_hh_l{l}"]
    return names + ["fc.weight", "fc.bias"]


def functional_forward(P: dict, x: torch.Tensor, layers: int, dropout: float, training: bool):
    """x: [C, B, T, d] -> logits [C, B, n_classes]."""
    C, B, T, _ = x.shape
    inp = x
    for l in range(layers):
        W_ih, W_hh = P[f"gru.weight_ih_l{l}"], P[f"gru.weight_hh_l{l}"]   # [C,3H,in], [C,3H,H]
        b_ih, b_hh = P[f"gru.bias_ih_l{l}"], P[f"gru.bias_hh_l{l}"]       # [C,3H]
        H = W_hh.shape[-1]
        gi_all = torch.einsum("cbti,cgi->cbtg", inp, W_ih) + b_ih[:, None, None, :]
        h = x.new_zeros(C, B, H)
        outs = []
        for t in range(T):
            gi = gi_all[:, :, t]
            gh = torch.einsum("cbh,cgh->cbg", h, W_hh) + b_hh[:, None, :]
            i_r, i_z, i_n = gi.chunk(3, -1)
            h_r, h_z, h_n = gh.chunk(3, -1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1 - z) * n + z * h
            outs.append(h)
        inp = torch.stack(outs, 2)                          # [C, B, T, H]
        if l < layers - 1 and dropout > 0:
            inp = F.dropout(inp, dropout, training)         # as nn.GRU(dropout=...)
    last = F.dropout(inp[:, :, -1], dropout, training)      # [GAP-FILL G4]
    return torch.einsum("cbh,coh->cbo", last, P["fc.weight"]) + P["fc.bias"][:, None, :]


class Layout:
    """Flat parameter layout: all parameters of one model concatenated in `param_names` order,
    so C models are one tensor [C, total]. Used for fast stacked training, aggregation and
    the C_dq norm."""

    def __init__(self, state: dict, names):
        self.names, self.shapes, self.offsets = list(names), [], []
        off = 0
        for n in self.names:
            self.shapes.append(tuple(state[n].shape))
            self.offsets.append(off)
            off += state[n].numel()
        self.total = off

    def flatten(self, state: dict) -> torch.Tensor:
        return torch.cat([state[n].detach().reshape(-1) for n in self.names])

    def views(self, flat: torch.Tensor) -> dict:
        """flat [C, total] -> dict of [C, *shape] views (keeps autograd)."""
        C = flat.shape[0]
        return {n: flat[:, o:o + int(torch.Size(sh).numel())].view(C, *sh)
                for n, sh, o in zip(self.names, self.shapes, self.offsets)}

    def state_dict(self, flat_row: torch.Tensor) -> dict:
        return {n: flat_row[o:o + int(torch.Size(sh).numel())].view(*sh).detach().clone()
                for n, sh, o in zip(self.names, self.shapes, self.offsets)}


class FlatAdam:
    """Adam on a flat stacked parameter tensor [C, total], one step counter per client.
    `step(a)` updates only the first `a` rows (the clients that had a batch this step; the
    trainer orders clients so the active ones are always a prefix). Same update as
    torch.optim.Adam without weight decay: p -= lr * m_hat / (sqrt(v_hat) + eps)."""

    def __init__(self, flat: torch.Tensor, lr, betas=(0.9, 0.999), eps=1e-8):
        self.flat, self.lr, self.b1, self.b2, self.eps = flat, lr, betas[0], betas[1], eps
        self.m = torch.zeros_like(flat)
        self.v = torch.zeros_like(flat)
        self.t = torch.zeros(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)

    @torch.no_grad()
    def step(self, a: int):
        p, g = self.flat[:a], self.flat.grad[:a]
        m, v, t = self.m[:a], self.v[:a], self.t[:a]
        t += 1
        m.mul_(self.b1).add_(g, alpha=1 - self.b1)
        v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        bc1 = 1 - self.b1 ** t
        bc2 = 1 - self.b2 ** t
        denom = (v / bc2).sqrt_().add_(self.eps)
        p.sub_(self.lr * (m / bc1) / denom)
        self.flat.grad = None


class FlatSGD:
    """Plain gradient descent on the flat stacked tensor: w <- w - eta * grad.
    [PAPER] Eqs. 17 and 24 (no momentum, no weight decay). Chosen over Adam by the user on
    2026-11-08 (SPEC section 7, option A) because Adam with eta = 0.2 collapses."""

    def __init__(self, flat: torch.Tensor, lr, *unused, **unused_kw):
        self.flat, self.lr = flat, lr
        self.t = torch.zeros(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)

    @torch.no_grad()
    def step(self, a: int):
        self.flat[:a] -= self.lr * self.flat.grad[:a]
        self.t[:a] += 1
        self.flat.grad = None


def n_params(state: dict) -> int:
    return int(sum(v.numel() for v in state.values()))
