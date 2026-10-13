# BLOCK 16b (v4 - speed): all clients' local DP-SGD steps in parallel on the GPU (same algorithm, far fewer GPU calls)
# For EACH client k, one step is exactly DPClient._dp_step:
#   Poisson sampling with the client's own rate q_k -> epsilon-greedy action -> one-step reward target ->
#   per-sample gradient clipped to C -> + N(0, (sigma_k * C)^2) noise -> / expected batch q_k * n_k ->
#   + FedProx term mu * (w - w_global) -> Adam.
# Adam is element-wise, so ONE Adam over the stacked [K, ...] tensors is K independent per-client Adams.
# Privacy accounting stays in each client's own RDP accountant (DPClient), step by step as before.
# Needs every client to train every round (CLIENT_FRAC = 1.0, the thesis setting); otherwise the per-client loop runs.
# v4: fast.basis = V (public subspace, Block 17d) -> the same projected clip + k-space noise as DPClient; None = unchanged.
from torch.func import stack_module_state


class FastFederation:
    def __init__(self, clients):
        self.clients, self.K = clients, len(clients)
        c0 = clients[0]
        self.dev, self.C = c0.device, c0.C
        assert all(c.C == self.C for c in clients), "all clients must share the clip norm"
        nets = [c.ddqn.online_net for c in clients]
        self.names = [n for n, _ in nets[0].named_parameters()]
        params, buffers = stack_module_state(nets)
        self.params = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
        self.buffers = {k: v.detach().clone() for k, v in buffers.items()}
        self.base = copy.deepcopy(nets[0]).to("meta")
        self.opt = optim.Adam([self.params[n] for n in self.names], lr=c0.ddqn.optimizer.param_groups[0]["lr"])
        self.ag = c0.ddqn                                   # epsilon schedule + reward (same weights for all clients)
        self.steps = c0.ddqn.steps
        self.n = torch.tensor([c.n for c in clients], device=self.dev)
        self.q = torch.tensor([c.q for c in clients], device=self.dev, dtype=torch.float32)
        nmax = int(self.n.max())
        self.X = torch.zeros(self.K, nmax, c0.X.shape[1], device=self.dev)
        self.Y = torch.zeros(self.K, nmax, dtype=torch.long, device=self.dev)
        for k, c in enumerate(clients):
            self.X[k, :c.n], self.Y[k, :c.n] = c.X, c.y
        self.row_ok = torch.arange(nmax, device=self.dev)[None, :] < self.n[:, None]

        base = self.base
        def q_all(p, b, x):                                 # Q-values of one client's batch
            return functional_call(base, (p, b), (x,))
        def loss_one(p, b, xi, ai, ri):                     # one record's loss (one-step target, gamma = 0)
            qv = functional_call(base, (p, b), (xi.unsqueeze(0),)).squeeze(0)
            return F.smooth_l1_loss(qv.gather(0, ai.unsqueeze(0)).squeeze(0), ri)
        self._q = vmap(q_all)
        self._g = vmap(vmap(grad(loss_one), in_dims=(None, None, 0, 0, 0)), in_dims=(0, 0, 0, 0, 0))
        self.last_grad = None
        self.basis = None                                   # v4: public-subspace basis V (d x k) or None = full space

    def _epsilon(self):
        a = self.ag
        return a.eps_start + min(1.0, self.steps / a.eps_decay_steps) * (a.eps_end - a.eps_start)

    def _step(self, sigma, mu, gp):
        K, dev = self.K, self.dev
        take = (torch.rand(self.row_ok.shape, device=dev) < self.q[:, None]) & self.row_ok   # Poisson, per client
        cnt = take.sum(1)
        B = int(cnt.max())
        if B > 0:
            order = torch.argsort((~take).to(torch.int8), dim=1, stable=True)[:, :B]
            valid = torch.arange(B, device=dev)[None, :] < cnt[:, None]
            x = torch.gather(self.X, 1, order[..., None].expand(-1, -1, self.X.shape[2]))
            y = torch.gather(self.Y, 1, order)
            p_det = {k: v.detach() for k, v in self.params.items()}
            with torch.no_grad():
                greedy = self._q(p_det, self.buffers, x).argmax(-1)
                explore = torch.rand(greedy.shape, device=dev) < self._epsilon()
                a = torch.where(explore, torch.randint(0, action_dim, greedy.shape, device=dev), greedy)
                r = self.ag.reward(a, y)
            g = self._g(p_det, self.buffers, x, a, r)                                       # [K, B, ...]
            if self.C is not None:
                norms = torch.sqrt(sum(gi.reshape(K, B, -1).pow(2).sum(-1) for gi in g.values()))
                factor = (self.C / (norms + 1e-12)).clamp(max=1.0)
            else:
                factor = torch.ones(K, B, device=dev)
            factor = factor * valid
            if self.basis is None:
                summed = {k: (gi * factor.view(K, B, *([1] * (gi.dim() - 2)))).sum(1) for k, gi in g.items()}
            else:                                           # v4: clip + noise inside the public subspace
                G = torch.cat([g[k].reshape(K, B, -1) for k in self.names], 2)
                like = {k: self.params[k][0] for k in self.names}
                summed = flat_to_params(projected_private_sum(G, self.basis, self.C, sigma, valid=valid), like)
        else:
            summed = {k: torch.zeros_like(v) for k, v in self.params.items()}
            if self.basis is not None:                      # empty batches: the subspace noise is still added
                like = {k: self.params[k][0] for k in self.names}
                G = torch.zeros(K, 0, self.basis.shape[0], device=dev)
                summed = flat_to_params(projected_private_sum(G, self.basis, self.C, sigma), like)
        ebs = self.q * self.n
        grads = {}
        for k in self.names:
            p = self.params[k]
            shape = (K,) + (1,) * (p.dim() - 1)
            gk = summed[k]
            if bool((sigma > 0).any()) and self.basis is None:
                gk = gk + torch.randn_like(p) * (sigma * self.C).view(shape)
            gk = gk / ebs.view(shape)
            if mu > 0:
                gk = gk + mu * (p.detach() - gp[k])
            p.grad = gk
            grads[k] = gk
        self.last_grad = grads
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.steps += 1

    def fit(self, gp_list, sigmas, steps, mu):
        """gp_list: global parameters (list, in parameter order). sigmas: one noise multiplier per client.
        Returns (params per client, sizes, metrics per client) like DPClient.fit."""
        sig = torch.tensor([float(s) for s in sigmas], device=self.dev)
        assert bool((sig >= 0).all()), "noise multiplier must be >= 0"
        if bool((sig > 0).any()):
            assert self.C is not None, "DP guard: noise without clipping has unbounded sensitivity"
        gp = {k: t.to(self.dev) for k, t in zip(self.names, gp_list)}
        with torch.no_grad():
            for k in self.names:
                self.params[k].copy_(gp[k].unsqueeze(0).expand_as(self.params[k]))
        for c, s in zip(self.clients, sigmas):           # accounting: identical to DPClient._dp_step, every step
            for _ in range(int(steps)):
                if s > 0:
                    c.accountant.step(noise_multiplier=float(s), sample_rate=c.q)
                else:
                    c.nonprivate_steps += 1
        for _ in range(int(steps)):
            self._step(sig, mu, gp)
        out, mets = [], []
        with torch.no_grad():
            gvec = torch.cat([gp[k].reshape(-1) for k in self.names])
            for i in range(self.K):
                p_i = [self.params[k][i].detach().cpu().clone() for k in self.names]
                out.append(p_i)
                d = float(torch.norm(torch.cat([t.reshape(-1) for t in p_i]).to(self.dev) - gvec))
                mets.append({"sigma": float(sigmas[i]), "epsilon_spent": self.clients[i].epsilon_spent(),
                             "dp_enabled": float(sigmas[i]) > 0, "delta_l2": d})
        return out, [c.n for c in self.clients], mets


def fit_clients(clients, active, gp, sigmas, steps, mu, fast=None):
    """Local training of the active clients. sigmas: dict client index -> noise multiplier.
    Uses FastFederation when every client is active; otherwise the per-client loop (identical algorithm)."""
    active = [int(i) for i in active]
    if fast is not None and sorted(active) == list(range(len(clients))):
        return fast.fit(gp, [sigmas[i] for i in range(len(clients))], steps, mu)
    params, sizes, mets = [], [], []
    for i in active:
        p, n, m = clients[i].fit(gp, {"noise_multiplier": sigmas[i], "local_steps": steps, "mu": mu})
        params.append(p); sizes.append(n); mets.append(m)
    return params, sizes, mets
