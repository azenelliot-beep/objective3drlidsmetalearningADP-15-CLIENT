"""Execute after Block 18a and BEFORE Block 18b in the supplied notebook.
Conditional record-level DP research implementation; see RUN_WITH_CLAUDE.md.
No performance improvement or end-to-end privacy guarantee is asserted.
"""

@torch.no_grad()
def public_double_target(online, target, anchors, rewards, gamma):
    """Online selects, target evaluates. Anchors MUST be designated public data.
    These are synthetic continuing classification-MDP transitions, not captured
    chronological transitions. No second private record enters a loss term.
    """
    online.eval()
    target.eval()
    action = online(anchors).argmax(1, keepdim=True)
    bootstrap = target(anchors).gather(1, action).squeeze(1)
    online.train()
    return rewards + gamma * bootstrap


def patched_private_step(self, sigma):
    sigma = float(sigma)
    assert sigma >= 0
    if sigma > 0:
        assert self.C is not None and self.C > 0
        self.accountant.step(noise_multiplier=sigma, sample_rate=self.q)
    else:
        self.nonprivate_steps += 1
    assert self.basis is None, 'This patch supports full-space gradients only; v4 needs a separate audit.'
    assert not CFG.get('BINARY') and not CFG.get('DRIFT'), 'Use the fixed 10-class experiment.'
    net = self.ddqn.online_net
    take = torch.rand(self.n, device=self.device) < self.q
    idx = take.nonzero().flatten()
    params = {k: v.detach() for k, v in net.named_parameters()}
    buffers = {k: v.detach() for k, v in net.named_buffers()}
    if len(idx):
        x, y = self.X[idx], self.y[idx]
        actions = self.ddqn.act(x)
        rewards = self.ddqn.reward(actions, y)
        # Fixed validation half A is explicitly treated as PUBLIC proxy data.
        # Sampling is independent of private labels/features and client batch indices.
        anchor_idx = torch.randint(len(PUBLIC_DDQN_ANCHORS), (len(idx),), device=self.device)
        anchors = PUBLIC_DDQN_ANCHORS[anchor_idx]
        target = public_double_target(net, self.ddqn.target_net, anchors, rewards,
                                      CFG['DDQN_GAMMA'])
        def one_loss(p, b, xi, ai, ti):
            qv = functional_call(net, (p, b), (xi.unsqueeze(0),)).squeeze(0)
            return F.smooth_l1_loss(qv[ai], ti)
        grads = vmap(grad(one_loss), in_dims=(None, None, 0, 0, 0),
                     randomness='different')(params, buffers, x, actions, target)
        norms = torch.sqrt(sum(g.reshape(len(idx), -1).square().sum(1) for g in grads.values()))
        factor = torch.ones_like(norms) if self.C is None else (self.C / (norms + 1e-12)).clamp(max=1)
        summed = {k: (g * factor.reshape(-1, *([1] * (g.ndim - 1)))).sum(0)
                  for k, g in grads.items()}
    else:
        summed = {k: torch.zeros_like(v) for k, v in params.items()}
    self.ddqn.optimizer.zero_grad(set_to_none=True)
    for k, p in net.named_parameters():
        value = summed[k]
        if sigma > 0:
            value = value + torch.randn_like(value) * (sigma * self.C)
        value = value / (self.q * self.n)
        if self.global_params is not None and self.mu > 0:
            value = value + self.mu * (p.detach() - self.global_params[k])
        p.grad = value
    self.ddqn.optimizer.step()
    self.ddqn.update_target()
    self.ddqn.steps += 1


def patched_fit(self, parameters, config):
    self.ddqn.set_fl_parameters(parameters)
    self.global_params = {k: v.detach().clone() for k, v in self.ddqn.online_net.named_parameters()}
    self.mu = float(config.get('mu', 0))
    sigma = float(config['noise_multiplier'])
    for _ in range(int(config['local_steps'])):
        self._dp_step(sigma)
    # Exclude raw local training accuracy from transmitted/released telemetry.
    norm = float(torch.norm(self.ddqn.param_vector() -
                           torch.cat([v.flatten() for v in self.global_params.values()])))
    return self.ddqn.get_fl_parameters(), self.n, dict(
        sigma=sigma, epsilon_spent=self.epsilon_spent(), dp_enabled=sigma > 0,
        delta_l2=norm)


def patched_finish_episode(self, rewards):
    assert len(rewards) == len(self._logp) and len(rewards) > 0
    returns = np.cumsum(np.asarray(rewards, dtype=np.float64)[::-1])[::-1]
    # Zero is the first-episode baseline, so reward gradients are active immediately.
    advantages = [float(g) - self.baseline.get(t, 0.0) for t, g in enumerate(returns)]
    loss = -sum(a * lp for a, lp in zip(advantages, self._logp))
    loss = loss - self.entropy_coef * sum(self._ent)
    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    self.last_policy_grad_norm = float(torch.sqrt(sum(
        p.grad.square().sum() for p in self.parameters() if p.grad is not None)))
    torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
    self.opt.step()
    for t, g in enumerate(returns):
        self.baseline[t] = self.beta * self.baseline.get(t, 0.0) + (1 - self.beta) * float(g)
    self._logp, self._ent = [], []
    return float(returns[0])


# A new directory prevents reuse of controllers/results from a different algorithm.
CFG.update(FAST=False, WEIGHTS_FROM='val', DDQN_GAMMA=0.1,
           RESULTS_DIR='results_ddqn_repaired_v1')
_DEFAULT_CFG.update(CFG)
DPClient._dp_step = patched_private_step
DPClient.fit = patched_fit
ADPMetaV2.finish_episode = patched_finish_episode
ia_public, _ = val_halves()
PUBLIC_DDQN_ANCHORS = torch.as_tensor(X_val[ia_public], dtype=torch.float32, device=device)

# Functional DDQN test: online chooses action 1, target evaluates it as 7.
class _ConstantQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer('values', torch.tensor(values, dtype=torch.float32, device=device))
    def forward(self, x):
        return self.values.expand(len(x), -1)
_test = public_double_target(_ConstantQ([9., 10.]), _ConstantQ([100., 7.]),
                             torch.zeros(2, input_dim, device=device),
                             torch.ones(2, device=device), 0.1)
assert torch.allclose(_test, torch.full_like(_test, 1.7)), 'Double-DQN select/evaluate failed'
print('DDQN patch loaded: nonzero public-anchor bootstrapping; FAST disabled; new result directory.')
