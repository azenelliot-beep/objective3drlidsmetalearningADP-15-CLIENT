"""Client selection. The DQN selects clients; it never classifies traffic.

[PAPER] Sec. III-B (p.4-5): MDP with state Eq. 6, action Eq. 7, reward Eq. 8; DQN with a primary
and a target network and experience replay (Eqs. 12-13); replay buffer 10,000 (Table I(B));
gamma = 0.99 (Fig. 5).
[GAP-FILL G5, G6] network shape, top-k joint action, epsilon schedule, update schedule.

Joint-action Q [GAP-FILL G5]: the network outputs Q[i, p] for every client i and power level p.
The action picks k clients (and one power each); Q(s, a) = mean of the k chosen Q[i, p_i].
Target = r + gamma * mean of the top-k values of max_p Q_target(s', i, p).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, state_dim, n_clients, n_powers, hidden=(64, 64)):
        super().__init__()
        layers, d = [], state_dim
        for h in hidden:                                   # [GAP-FILL G6] 2 x 64 ReLU
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_clients * n_powers))
        self.net = nn.Sequential(*layers)
        self.n, self.p = n_clients, n_powers

    def forward(self, s):                                  # [B, state_dim] -> [B, N, P]
        return self.net(s).view(-1, self.n, self.p)


class Replay:
    def __init__(self, cap, rng):
        self.cap, self.rng, self.buf, self.i = cap, rng, [], 0

    def push(self, tr):
        if len(self.buf) < self.cap:
            self.buf.append(tr)
        else:
            self.buf[self.i] = tr
        self.i = (self.i + 1) % self.cap

    def sample(self, n):                                   # [PAPER] uniform mini-batch, xi ~ U(D)
        idx = self.rng.choice(len(self.buf), n, replace=False)
        return [self.buf[j] for j in idx]


class Selector:
    """method: 'dqn' (paper), 'random' ([EXTRA] X2), 'all' (FedAvg baseline)."""

    def __init__(self, cfg: dict, n_clients: int, rng: np.random.Generator, device="cpu"):
        sc, dq = cfg["selection"], cfg["dqn"]
        self.method, self.k = sc["method"], (n_clients if sc["method"] == "all" else sc["k"])
        self.powers_dbm = np.array(sc["power_levels_dbm"], float)
        self.fedavg_power = sc["fedavg_power_dbm"]
        self.n, self.rng, self.dq, self.device = n_clients, rng, dq, device
        self.state_dim = 5 * n_clients                     # [PAPER] Eq. 6, 5 features per device
        self.updates, self.losses = 0, []
        if self.method == "dqn":
            self.q = QNet(self.state_dim, n_clients, len(self.powers_dbm), dq["hidden"]).to(device)
            self.q_t = QNet(self.state_dim, n_clients, len(self.powers_dbm), dq["hidden"]).to(device)
            self.q_t.load_state_dict(self.q.state_dict())  # [PAPER] target network
            self.opt = torch.optim.Adam(self.q.parameters(), lr=dq["lr"])
            self.replay = Replay(dq["buffer"], rng)

    def epsilon(self, rnd: int) -> float:
        """[GAP-FILL G5] linear 1.0 (round 1) -> 0.05 (round 50), then constant."""
        d = self.dq
        frac = min(1.0, (rnd - 1) / max(1, d["eps_decay_rounds"] - 1))
        return d["eps_start"] + frac * (d["eps_end"] - d["eps_start"])

    def select(self, state: np.ndarray, rnd: int):
        """Returns (selected client ids sorted, power index per selected client, explored flag)."""
        if self.method == "all":                            # [PAPER] FedAvg; power [GAP-FILL G12]
            p = int(np.flatnonzero(self.powers_dbm == self.fedavg_power)[0]) \
                if self.fedavg_power in self.powers_dbm else None
            return np.arange(self.n), np.full(self.n, p if p is not None else -1), False
        explore = self.method == "random" or self.rng.random() < self.epsilon(rnd)
        if explore:
            sel = np.sort(self.rng.choice(self.n, self.k, replace=False))
            return sel, self.rng.integers(0, len(self.powers_dbm), self.k), True
        with torch.no_grad():
            q = self.q(torch.as_tensor(state, dtype=torch.float32, device=self.device)[None])[0]
        best_q, best_p = q.max(1)
        sel = np.sort(torch.topk(best_q, self.k).indices.cpu().numpy())
        return sel, best_p.cpu().numpy()[sel], False

    def power_w(self, pidx: np.ndarray) -> np.ndarray:
        dbm = np.where(pidx >= 0, self.powers_dbm[np.clip(pidx, 0, None)], self.fedavg_power)
        return 10.0 ** ((dbm - 30.0) / 10.0)

    def observe(self, s, sel, pidx, r, s_next, done):
        if self.method != "dqn":
            return
        self.replay.push((s.astype(np.float32), sel.copy(), pidx.copy(), float(r),
                          s_next.astype(np.float32), float(done)))
        for _ in range(self.dq["updates_per_round"]):
            self._update()

    def _update(self):
        d = self.dq
        if len(self.replay.buf) < d["batch"]:
            return
        batch = self.replay.sample(d["batch"])
        dev = self.device
        s = torch.as_tensor(np.stack([b[0] for b in batch]), device=dev)
        s2 = torch.as_tensor(np.stack([b[4] for b in batch]), device=dev)
        r = torch.as_tensor([b[3] for b in batch], device=dev)
        done = torch.as_tensor([b[5] for b in batch], device=dev)
        sel = torch.as_tensor(np.stack([b[1] for b in batch]), device=dev)       # [B, k]
        pid = torch.as_tensor(np.stack([b[2] for b in batch]), device=dev)       # [B, k]
        q = self.q(s)                                                            # [B, N, P]
        q_sa = q[torch.arange(len(batch), device=dev)[:, None], sel, pid].mean(1)
        with torch.no_grad():
            nxt = self.q_t(s2).max(2).values.topk(self.k, dim=1).values.mean(1)
            target = r + d["gamma"] * (1 - done) * nxt                           # [PAPER] Eq. 12
        loss = ((target - q_sa) ** 2).mean()                                     # [PAPER] Eq. 12 (MSE)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()                                                          # [PAPER] Eq. 13
        self.updates += 1
        self.losses.append(float(loss))
        if self.updates % d["target_sync_every"] == 0:                           # [GAP-FILL G6]
            self.q_t.load_state_dict(self.q.state_dict())
