"""FastFederation (Block 16b) must compute exactly what the per-client DPClient loop computes."""
import numpy as np
import pytest
import torch


def make_pair(nb, n_per=40, clip=0.5, seed=0):
    """Two identical sets of 3 clients (same init, same data). q = 1 (batch >= n) and greedy actions make
    the sampling and the actions deterministic, so both implementations must give the same parameters."""
    def build():
        torch.manual_seed(seed)
        cs = []
        for k in range(3):
            ag = nb["SmallDDQN"](nb["input_dim"], nb["action_dim"], nb["device"], nb["class_weights_dp"],
                                 gamma=0.0, eps_start=0.0, eps_end=0.0)
            X, y = nb["X_dp"][k * n_per:(k + 1) * n_per + k], nb["y_dp"][k * n_per:(k + 1) * n_per + k]
            cs.append(nb["DPClient"](ag, X, y, nb["device"], batch_size=10_000, clip_norm=clip))
        return cs
    return build(), build()


@pytest.mark.parametrize("clip,mu", [(0.5, 0.0), (0.05, 0.1), (None, 0.01)])
def test_fast_equals_per_client_loop(nb, clip, mu):
    slow, fastc = make_pair(nb, clip=clip)
    gp = slow[0].ddqn.get_fl_parameters()
    fast = nb["FastFederation"](fastc)
    sig = {0: 0.0, 1: 0.0, 2: 0.0}
    for _ in range(2):                                     # two rounds: Adam state must carry over like the loop
        ps, ns, _ = nb["fit_clients"](slow, [0, 1, 2], gp, sig, 5, mu, None)
        pf, nf, mf = nb["fit_clients"](fastc, [0, 1, 2], gp, sig, 5, mu, fast)
        assert ns == nf
        for a, b in zip(ps, pf):
            for ta, tb in zip(a, b):
                assert torch.allclose(ta, tb, atol=1e-5, rtol=1e-4)
        gp = nb["weighted_average"](pf, nf)
    assert all(m["epsilon_spent"] is None for m in mf)    # sigma 0 -> never an epsilon

def test_fast_accounting_matches(nb):
    slow, fastc = make_pair(nb)
    gp = slow[0].ddqn.get_fl_parameters()
    fast = nb["FastFederation"](fastc)
    sig = {0: 1.1, 1: 1.3, 2: 0.9}
    nb["fit_clients"](slow, [0, 1, 2], gp, sig, 7, 0.0, None)
    _, _, mf = nb["fit_clients"](fastc, [0, 1, 2], gp, sig, 7, 0.0, fast)
    for s, f, m in zip(slow, fastc, mf):
        assert s.accountant.history == f.accountant.history
        assert f.epsilon_spent() == pytest.approx(s.epsilon_spent()) == pytest.approx(m["epsilon_spent"])

def test_fast_noise_scale_is_sigma_C_over_expected_batch(nb):
    _, cs = make_pair(nb, clip=0.5)
    fast = nb["FastFederation"](cs)
    gp = {k: v.detach().clone() for k, v in zip(fast.names, cs[0].ddqn.get_fl_parameters())}
    torch.manual_seed(1); fast._step(torch.zeros(3), 0.0, gp); g0 = {k: v.clone() for k, v in fast.last_grad.items()}
    sig = torch.tensor([1.0, 2.0, 4.0])
    for k in fast.names:                                  # reset params so the signal part is identical
        with torch.no_grad():
            fast.params[k].copy_(gp[k].unsqueeze(0).expand_as(fast.params[k]))
    fast.opt = torch.optim.Adam([fast.params[n] for n in fast.names], lr=1e-3)
    fast.steps = 0
    torch.manual_seed(1); fast._step(sig, 0.0, gp)
    noise = torch.cat([(fast.last_grad[k] - g0[k]).reshape(3, -1) for k in fast.names], 1)
    expected = sig * 0.5 / (fast.q * fast.n.float())
    assert torch.allclose(noise.std(1), expected, rtol=0.1)

def test_fast_refuses_noise_without_clipping(nb):
    _, cs = make_pair(nb, clip=None)
    fast = nb["FastFederation"](cs)
    with pytest.raises(AssertionError, match="unbounded"):
        fast.fit(cs[0].ddqn.get_fl_parameters(), [1.0, 1.0, 1.0], 1, 0.0)

def test_runner_uses_fast_path_and_respects_budget(nb, small_cfg, monkeypatch):
    called = []
    orig = nb["FastFederation"].fit
    monkeypatch.setattr(nb["FastFederation"], "fit", lambda self, *a: called.append(1) or orig(self, *a))
    out = nb["run_dp_pipeline"]("dp_fixed", 0, 2.0, C=1.0)
    assert called and all(h["epsilon"] <= 2.0 + 1e-9 for h in out["history"]) and out["final_epsilon"] > 1.8
