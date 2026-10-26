import itertools

import numpy as np
import pytest
from scipy import stats

from torre2025_fl_cnn_pp import privacy as pp

NAMES = ["C1", "C2", "C3", "C4"]


def _model(rng):
    return [rng.normal(0, 0.2, (3, 1, 32)), rng.normal(0, 0.1, 32), rng.normal(0, 0.1, (64, 128)),
            rng.normal(0, 0.1, 128), np.abs(rng.normal(0.5, 0.2, 128))]


def test_he_round_trip_is_exact_with_large_and_zero_offsets():
    rng = np.random.default_rng(0)
    for o_p in (0, 7919, -612345, 812345):
        L = pp.add_offset(_model(rng), o_p)
        enc, d = pp.he_encrypt(L, o_p, 100, rng)
        assert any(np.max(np.abs(e - x)) > 0.5 for e, x in zip(enc, L))       # values really change
        dec = pp.he_decrypt(enc, d)
        assert max(np.max(np.abs(a - b)) for a, b in zip(dec, L)) < 1e-8
        assert d == {"D_X": {}, "D_Y": {}}                                    # entries removed (Alg.5 l.9)


def test_he_uses_all_four_operations_and_sign_flips():
    rng = np.random.default_rng(1)
    seen, flips = set(), 0
    for _ in range(30):
        L = [rng.normal(0, 3, (40, 8)), rng.normal(0, 3, 16)]
        _, d = pp.he_encrypt(L, 0, 20, rng)
        for entries in list(d["D_X"].values()) + list(d["D_Y"].values()):
            for e in entries:
                seen.add(pp.OPS[e["op"]])
                flips += int(np.sum(e["flip"]))
    assert seen == set(pp.OPS)
    assert flips > 0


def test_he_division_never_chosen_for_lists_with_zero_operand():
    rng = np.random.default_rng(2)
    L = [np.full((50, 4), 0.3)]        # trunc(0.3) - 0 = 0 for every element
    for _ in range(50):
        _, d = pp.he_encrypt(L, 0, 100, rng)
        assert all(pp.OPS[e["op"]] != "divide" for e in d["D_X"][0])


def test_he_integrity_check_detects_tampering():
    rng = np.random.default_rng(3)
    enc, d = pp.he_encrypt(_model(rng), 5, 100, rng)
    enc[0] = enc[0] + 1.0
    with pytest.raises(ValueError, match="integrity"):
        pp.he_decrypt(enc, d)


def test_dh_offsets_cancel_for_every_active_subset_and_stay_in_sync():
    parties = pp.diffie_hellman_setup(NAMES)
    for a, b in itertools.permutations(NAMES, 2):
        assert parties[a].deltas[b] == parties[b].deltas[a]
    for _ in range(5):                                             # several iterations of delta updates
        for r in range(1, 5):
            for active in itertools.combinations(NAMES, r):
                assert sum(parties[c].offset_total(set(active)) for c in active) == 0
        assert any(parties["C1"].offset_total(set(NAMES)) != 0 for _ in range(1))
        for p in parties.values():
            p.update_deltas()
        for a, b in itertools.permutations(NAMES, 2):
            assert parties[a].deltas[b] == parties[b].deltas[a]


def test_dp_shares_sum_to_laplace():
    rng = np.random.default_rng(4)
    total = sum(pp.laplace_gamma_noise([(100000,)], 4, 0.0, 0.1, rng)[0] for _ in range(4))
    assert stats.kstest(total, "laplace", args=(0.0, 0.1)).pvalue > 0.01
    one = pp.laplace_gamma_noise([(100000,)], 4, 0.0, 0.1, rng)[0]
    assert stats.kstest(one, "laplace", args=(0.0, 0.1)).pvalue < 1e-6   # a single share is not Laplace


def test_dp_sensitivity_formula():
    x, y = [np.array([1.0, 2.0])], [np.array([1.5, -1.0])]
    assert pp.max_abs_difference(x, y) == 3.0
    assert pp.dp_scale_from_sensitivity(3.0, 2.0) == 1.5
