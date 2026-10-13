# BLOCK 16 (v4 - C0): Record-level DP-SGD client + RDP privacy accounting
# - Each flow (record) belongs to exactly ONE client. Within a client it is Poisson-sampled many times
#   over the run (expected ROUNDS * LOCAL_STEPS * q times); the RDP accountant counts every step.
# - Per-sample gradients clipped to C, Gaussian noise N(0, (sigma*C)^2), Poisson sampling rate q = B / n
# - Privacy measured with the Opacus RDP accountant (no hand formula)
# - DP runs use one-step targets (gamma = 0): each loss term depends on one record only.
# C0 changes (see CHANGELOG.md):
#   - noise_multiplier = 0 -> no noise and NO accountant step; such a client reports epsilon = None (never a number)
#   - clip_norm = None -> no clipping; only allowed together with noise_multiplier = 0 (non-private baselines)
#   - projected_epsilon(sigma, steps) lives here so every runner uses the same pre-round budget check
# v4 (Option 2, public-subspace projection; used only by Block 17d, default OFF):
#   - client.basis = V (d x k, orthonormal columns, built from PUBLIC data only) -> each per-sample gradient is
#     projected to e_i = V^T g_i, clipped to ||e_i|| <= C, summed, N(0, (sigma*C)^2 I_k) noise is added in the
#     k-dim space and the result is mapped back with V. Sensitivity is still C and V does not depend on private
#     data, so the RDP accounting is unchanged; the noise norm drops from ~sigma*C*sqrt(d) to sigma*C*sqrt(k).
#   - client.basis = None (default) -> exactly the full-space DP-SGD step above (code path unchanged).
import copy
import math
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from opacus.accountants import RDPAccountant
from sklearn.utils.class_weight import compute_class_weight

# ---------- DP data: ALL training records, each in exactly one client (no under/oversampling) ----------
X_dp, y_dp = X_train.astype(np.float32), y_train.astype(np.int64)
cw_dp = compute_class_weight('balanced', classes=np.arange(len(class_names)), y=y_dp)
class_weights_dp = (np.sqrt(cw_dp) / np.sqrt(cw_dp).mean()).astype(np.float32)

dp_client_indices = dirichlet_split(y_dp, NUM_CLIENTS, ALPHA, np.random.default_rng(SEED), MIN_PER_CLIENT)
_all = np.concatenate(dp_client_indices)
assert len(_all) == len(y_dp) and len(np.unique(_all)) == len(y_dp), "A record is in more than one client"
print("DP class weights:", dict(zip(class_names, np.round(class_weights_dp, 3))))
print("DP training data per class:", dict(zip(class_names, np.bincount(y_dp, minlength=len(class_names)))))
print("DP client sizes (seed", SEED, "):", [len(ci) for ci in dp_client_indices])


def flat_to_params(vec, like):
    """Split a flat [..., d] tensor into tensors shaped like `like` (dict name -> tensor), in dict order."""
    out, off, lead = {}, 0, vec.shape[:-1]
    for k, v in like.items():
        n = v.numel()
        out[k] = vec[..., off:off + n].reshape(*lead, *v.shape)
        off += n
    assert off == vec.shape[-1], "flat vector length does not match the parameters"
    return out


def projected_private_sum(G, basis, C, sigma, valid=None):
    """v4 projection step. G: [..., B, d] per-sample gradients; basis V: [d, k].
    Returns sum_i clip_C(V^T g_i) + N(0, (sigma*C)^2 I_k), mapped back to [..., d] (noise only if sigma > 0).
    sigma may be a float or a [...] tensor (one value per client)."""
    E = G @ basis                                                           # [..., B, k]
    if C is not None:
        factor = (C / (E.norm(dim=-1) + 1e-12)).clamp(max=1.0)              # per-sample clipping in k-space
    else:
        factor = torch.ones(E.shape[:-1], device=E.device)
    if valid is not None:
        factor = factor * valid
    e = (E * factor.unsqueeze(-1)).sum(-2)                                  # [..., k]
    sig = torch.as_tensor(sigma, dtype=e.dtype, device=e.device)
    if bool((sig > 0).any()):
        assert C is not None, "DP guard: noise without clipping has unbounded sensitivity"
        e = e + torch.randn_like(e) * (sig * C).reshape(*sig.shape, *([1] * (e.dim() - sig.dim())))
    return e @ basis.T                                                      # back to parameter space [..., d]


class DPClient:
    def __init__(self, ddqn, X_local, y_local, device, model_name="Client",
                 batch_size=64, clip_norm=1.0, delta=1e-5):
        self.ddqn, self.device, self.model_name = ddqn, device, model_name
        self.X = torch.as_tensor(X_local, dtype=torch.float32, device=device)
        self.y = torch.as_tensor(y_local, dtype=torch.long, device=device)
        self.n = len(self.y)
        self.batch_size, self.delta = batch_size, float(delta)
        self.C = None if clip_norm is None else float(clip_norm)
        self.q = min(1.0, batch_size / self.n)             # Poisson sampling rate
        self.accountant = RDPAccountant()
        self.nonprivate_steps = 0                          # steps taken with sigma = 0 (no privacy claim possible)
        self.mu, self.global_params = 0.0, None
        self.basis = None                                  # v4: public-subspace basis V (d x k) or None = full space

    def epsilon_spent(self):
        if self.nonprivate_steps:
            return None                                    # any non-private step -> no epsilon may be reported
        return float(self.accountant.get_epsilon(delta=self.delta)) if self.accountant.history else 0.0

    def projected_epsilon(self, sigma, steps):
        """Epsilon this client WOULD have after `steps` more DP steps at `sigma` (nothing is spent)."""
        acc = copy.deepcopy(self.accountant)
        acc.history = list(acc.history) + [(float(sigma), self.q, int(steps))]
        return float(acc.get_epsilon(delta=self.delta))

    def _dp_step(self, sigma):
        sigma = float(sigma)
        assert sigma >= 0.0, "noise multiplier must be >= 0"
        if sigma > 0:
            assert self.C is not None, "DP guard: noise without clipping has unbounded sensitivity"
            self.accountant.step(noise_multiplier=sigma, sample_rate=self.q)   # counted even if batch is empty
        else:
            self.nonprivate_steps += 1

        net = self.ddqn.online_net
        idx = (torch.rand(self.n, device=self.device) < self.q).nonzero().squeeze(1)
        params  = {k: v.detach() for k, v in net.named_parameters()}
        buffers = {k: v.detach() for k, v in net.named_buffers()}

        if len(idx) > 0:
            x, y = self.X[idx], self.y[idx]
            a = self.ddqn.act(x)                          # epsilon-greedy action
            r = self.ddqn.reward(a, y)                    # one-step target (gamma = 0)
            net.train()

            def loss_fn(p, b, xi, ai, ri):
                qv = functional_call(net, (p, b), (xi.unsqueeze(0),)).squeeze(0)
                return F.smooth_l1_loss(qv.gather(0, ai.unsqueeze(0)).squeeze(0), ri)

            g = vmap(grad(loss_fn), in_dims=(None, None, 0, 0, 0), randomness="different")(params, buffers, x, a, r)
            B = len(idx)
            if self.C is not None:
                norms = torch.sqrt(sum(gi.reshape(B, -1).pow(2).sum(1) for gi in g.values()))
                factor = (self.C / (norms + 1e-12)).clamp(max=1.0)        # per-sample clipping
            else:
                factor = torch.ones(B, device=self.device)                 # non-private baseline: no clipping
            if self.basis is None:
                summed = {k: (gi * factor.view(-1, *([1] * (gi.dim() - 1)))).sum(0) for k, gi in g.items()}
            else:                                       # v4: clip + noise inside the public subspace
                G = torch.cat([g[k].reshape(B, -1) for k in params], 1)
                summed = flat_to_params(projected_private_sum(G, self.basis, self.C, sigma), params)
        else:
            summed = {k: torch.zeros_like(v) for k, v in params.items()}
            if self.basis is not None:                  # empty batch: the subspace noise is still added
                G = torch.zeros(0, self.basis.shape[0], device=self.device)
                summed = flat_to_params(projected_private_sum(G, self.basis, self.C, sigma), params)

        expected_bs = self.q * self.n
        for k, p in net.named_parameters():
            p.grad = summed[k] / expected_bs
            if sigma > 0 and self.basis is None:        # full-space noise (with a basis it was added in k-space)
                p.grad = p.grad + torch.normal(0.0, sigma * self.C, size=p.shape, device=self.device) / expected_bs
            if self.global_params is not None and self.mu > 0:          # FedProx: data-independent, no privacy cost
                p.grad = p.grad + self.mu * (p.detach() - self.global_params[k])
        self.ddqn.optimizer.step()
        self.ddqn.optimizer.zero_grad(set_to_none=True)
        self.ddqn.update_target()
        self.ddqn.steps += 1

    @torch.no_grad()
    def local_accuracy(self, max_samples=2048):
        xs, ys = self.X[:max_samples], self.y[:max_samples]
        return 100.0 * float((self.ddqn.predict(xs) == ys).float().mean())

    def fit(self, parameters, config):
        self.ddqn.set_fl_parameters(parameters)
        net = self.ddqn.online_net
        self.global_params = {k: v.detach().clone() for k, v in net.named_parameters()}
        self.mu = float(config.get("mu", 0.0))
        sigma = float(config["noise_multiplier"])
        for _ in range(int(config["local_steps"])):
            self._dp_step(sigma)
        # Aggregation of a DP-trained model is post-processing: no extra noise needed here
        metrics = {"accuracy": self.local_accuracy(), "sigma": sigma,
                   "epsilon_spent": self.epsilon_spent(), "dp_enabled": sigma > 0,
                   "delta_l2": float(torch.norm(self.ddqn.param_vector() -
                                     torch.cat([v.view(-1) for v in self.global_params.values()])))}
        return self.ddqn.get_fl_parameters(), self.n, metrics
