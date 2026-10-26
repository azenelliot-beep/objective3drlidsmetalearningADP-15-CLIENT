"""Block 18d: Torre et al. CNN detector inside the Chapter 6 pipeline."""
import numpy as np
import pytest
import torch


def make_pair(nb, n_per=40, clip=0.5, seed=0):
    def build():
        torch.manual_seed(seed)
        cs = []
        for k in range(3):
            ag = nb["CNNAgent"](nb["input_dim"], nb["action_dim"], nb["device"], nb["class_weights_dp"])
            X, y = nb["X_dp"][k * n_per:(k + 1) * n_per + k], nb["y_dp"][k * n_per:(k + 1) * n_per + k]
            cs.append(nb["SupervisedDPClient"](ag, X, y, nb["device"], batch_size=10_000, clip_norm=clip))
        return cs
    return build(), build()


def test_architecture_matches_algorithm1(nb):
    net = nb["TorreCNN"](29, 10)
    kinds = [type(m).__name__ for m in net.children()]
    assert kinds == ["Conv1d", "MaxPool1d", "Linear", "Linear"]
    assert net.conv.out_channels == 32 and net.conv.kernel_size == (3,) and net.pool.kernel_size == 2
    assert net.dense.in_features == 32 * ((29 - 3 + 1) // 2) and net.dense.out_features == 128
    assert not any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.Dropout)) for m in net.modules())
    assert net(torch.zeros(5, 29)).shape == (5, 10)


@pytest.mark.parametrize("clip,mu", [(0.5, 0.0), (0.05, 0.1), (None, 0.01)])
def test_fast_equals_per_client_loop(nb, clip, mu):
    slow, fastc = make_pair(nb, clip=clip)
    gp = slow[0].ddqn.get_fl_parameters()
    fast = nb["SupervisedFastFederation"](fastc)
    sig = {0: 0.0, 1: 0.0, 2: 0.0}
    for _ in range(2):
        ps, ns, _ = nb["fit_clients"](slow, [0, 1, 2], gp, sig, 5, mu, None)
        pf, nf, _ = nb["fit_clients"](fastc, [0, 1, 2], gp, sig, 5, mu, fast)
        for a, b in zip(ps, pf):
            for ta, tb in zip(a, b):
                assert torch.allclose(ta, tb, atol=1e-5, rtol=1e-4)
        gp = nb["weighted_average"](pf, nf)


def test_per_sample_clipping_bounds_each_record(nb):
    """With q = 1 and no noise, the summed update is the sum of per-record gradients each clipped to C."""
    _, cs = make_pair(nb, clip=1e-3)
    c = cs[0]
    net = c.ddqn.online_net
    params = {k: v.detach() for k, v in net.named_parameters()}
    g = torch.func.vmap(torch.func.grad(nb["_ce_one"](net, c.ddqn.ce_w)), in_dims=(None, None, 0, 0))(
        params, {}, c.X, c.y)
    norms = torch.sqrt(sum(gi.reshape(c.n, -1).pow(2).sum(1) for gi in g.values()))
    assert bool((norms > 1e-3).all())                    # every record is clipped at this C
    clipped = torch.sqrt(sum((gi * (1e-3 / norms).view(-1, *[1] * (gi.dim() - 1))).sum(0).pow(2).sum()
                             for gi in g.values()))
    assert float(clipped) <= c.n * 1e-3 + 1e-9


def test_accounting_matches_ddqn_client(nb):
    _, cs = make_pair(nb)
    gp = cs[0].ddqn.get_fl_parameters()
    nb["fit_clients"](cs, [0, 1, 2], gp, {0: 1.1, 1: 1.3, 2: 0.9}, 7, 0.0, None)
    for c, s in zip(cs, (1.1, 1.3, 0.9)):
        assert sum(h[2] for h in c.accountant.history) == 7
        assert {h[0] for h in c.accountant.history} == {s} and {h[1] for h in c.accountant.history} == {c.q}
        assert c.epsilon_spent() > 0


def test_cnn_pipeline_runs_and_respects_budget(nb, small_cfg):
    nb["CFG"]["TAG"] = "torre_cnn_test"
    out = nb["run_cnn_pipeline"]("dp_fixed", 0, 2.0, C=nb["cnn_clip_norm"]())
    assert out["arch"] == "torre_cnn"
    assert all(h["epsilon"] <= 2.0 + 1e-9 for h in out["history"]) and out["final_epsilon"] > 1.8
    np_out = nb["run_cnn_pipeline"]("fedavg_np", 0)
    assert np_out["final_epsilon"] is None


def test_block19_can_rebuild_cnn(nb, small_cfg):
    nb["CFG"]["TAG"] = "torre_cnn_test"
    out = nb["run_cnn_pipeline"]("fedprox_np", 1)
    agent = nb["ARCH"][out["arch"]](nb["input_dim"], nb["action_dim"], nb["device"], nb["class_weights_dp"])
    agent.set_fl_parameters(torch.load(nb["result_path"](out["run"], 1, "_best.pt")))
    m = nb["eval_metrics"](agent, nb["val_dataset"])
    assert m["Macro_F1"] == pytest.approx(out["val_at_best"]["Macro_F1"])


@pytest.mark.parametrize("clip,mu", [(0.5, 0.0), (1e-3, 0.1), (None, 0.01), (50.0, 0.0)])
def test_ghost_closed_form_equals_vmap(nb, clip, mu):
    """GhostCNNFederation (closed-form per-record gradients) must give the same parameters as the vmap version,
    with clipping active (small C), inactive (large C) or absent."""
    a, b = make_pair(nb, clip=clip)
    gp = a[0].ddqn.get_fl_parameters()
    fa, fb = nb["SupervisedFastFederation"](a), nb["GhostCNNFederation"](b)
    sig = {0: 0.0, 1: 0.0, 2: 0.0}
    for _ in range(2):
        pa, na, _ = nb["fit_clients"](a, [0, 1, 2], gp, sig, 4, mu, fa)
        pb, nb_, _ = nb["fit_clients"](b, [0, 1, 2], gp, sig, 4, mu, fb)
        for x, y in zip(pa, pb):
            for ta, tb in zip(x, y):
                assert torch.allclose(ta, tb, atol=1e-5, rtol=1e-4)
        gp = nb["weighted_average"](pb, nb_)


def test_runner_uses_closed_form_path(nb, small_cfg, monkeypatch):
    called = []
    orig = nb["GhostCNNFederation"]._clipped_sum
    monkeypatch.setattr(nb["GhostCNNFederation"], "_clipped_sum", lambda self, *a: called.append(1) or orig(self, *a))
    nb["CFG"]["TAG"] = "torre_cnn_test"
    nb["run_cnn_pipeline"]("dp_fixed", 2, 2.0, C=1.0)
    assert called
