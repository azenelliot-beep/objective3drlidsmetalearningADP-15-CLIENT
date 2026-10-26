"""The three privacy-preservation (PP) layers of Torre et al. 2025, Section 3.3-3.7 and 3.10.

    DP  - Algorithm 2 and Section 3.10.1: Laplace noise built from Gamma differences (PrivacyFL, ref [59]).
    DK  - Algorithm 3 and Section 3.10.2: pairwise Diffie-Hellman offsets O_tot added to L2.
    HE  - Algorithm 4 (encryption) and Algorithm 5 (decryption): per-list random add/sub/mul/div applied
          to the integer part of every element, with the key material kept in directories D_X / D_Y.

A model ("L2") is a list of numpy arrays (Keras get_weights order). Arrays with ndim >= 2 are "weights"
(directory D_X), 1-D arrays are "biases" (directory D_Y), following Section 3.1. All PP arithmetic is done
in float64 so that adding large integer offsets does not destroy the fractional weight information.
"""
import random
import sys

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def is_weight_array(a):
    """Weights (D_X) are >= 2-D arrays, biases (D_Y) are 1-D (Section 3.1)."""
    return np.ndim(a) >= 2


# =====================================================================================================
# DP: Algorithm 2 + equation L(mu, lambda) = mu/n + sum_{p=1..n} (gamma_p - gamma'_p)
# =====================================================================================================
def dp_scale_from_sensitivity(delta, epsilon):
    """lambda = Delta / x (Section 3.3); x is read as the privacy budget epsilon."""
    return float(delta) / float(epsilon)


def max_abs_difference(xs, ys):
    """Delta = max_i |x_i - y_i| (Section 3.3), over every element of two models."""
    return max(float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
               for x, y in zip(xs, ys))


def laplace_gamma_noise(shapes, n_clients, mu, scale, rng):
    """One client's share of the distributed Laplace mechanism.

    Each element gets mu/n + (G1 - G2) with G1, G2 ~ Gamma(shape=1/n, scale=lambda) (Algorithm 2, lines 5-7).
    Summed over the n clients, the shares are exactly Laplace(mu, lambda) (divisibility of the Laplace law),
    which is the equation in Sections 3.3 / 3.10.1.
    """
    k = 1.0 / n_clients
    return [mu / n_clients + rng.gamma(k, scale, size=s) - rng.gamma(k, scale, size=s) for s in shapes]


def add_dp_noise(L2, n_clients, mu, scale, rng):
    """Algorithm 2: A = zeros(L.shape); fill A with Gamma-difference noise; L2 = L2 + A."""
    A = laplace_gamma_noise([np.shape(a) for a in L2], n_clients, mu, scale, rng)
    return [np.asarray(a, np.float64) + n for a, n in zip(L2, A)], A


# =====================================================================================================
# DK: Diffie-Hellman key agreement and Algorithm 3 offsets
# =====================================================================================================
_CURVES = {"SECP384R1": ec.SECP384R1, "SECP256R1": ec.SECP256R1}


class DiffieHellmanParty:
    """One client's DH state (PrivacyFL utils/diffie_hellman.py + ClientAgent.update_deltas).

    Phase 1 (initialisation): one private/public key pair per peer.
    Phase 2: public keys are exchanged.
    Phase 3 (key agreement): ECDH shared secret -> HKDF-SHA256 -> 256-bit integer common key.
    Deltas (the integer offsets for a peer) are derived from the common key and re-derived after every use
    from a portion of the previous seed (Section 3.3: "using a portion of the original delta as a seed").
    """

    def __init__(self, name, peer_names, curve="SECP384R1", bits_slice=20):
        self.name = name
        self.order = list(peer_names)          # directory order, identical for every client
        self.bits_slice = bits_slice
        self._priv = {p: ec.generate_private_key(_CURVES[curve]()) for p in self.order if p != name}
        self.public_keys = {p: k.public_key() for p, k in self._priv.items()}
        self.seeds = {p: 0 for p in self.order}
        self.deltas = {p: 0 for p in self.order}   # delta with yourself is 0 (Algorithm 3, line 10)
        self._agreed = False

    def agree(self, peer_public_keys):
        """peer_public_keys[p] = the public key that peer p generated for *this* client."""
        for p in self.order:
            if p == self.name:
                continue
            shared = self._priv[p].exchange(ec.ECDH(), peer_public_keys[p])
            key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"handshake data").derive(shared)
            self.seeds[p] = int.from_bytes(key, "big")
        self._agreed = True
        self.update_deltas()

    def update_deltas(self):
        """PrivacyFL update_deltas: delta = leading bits of the seed, next seed from the remaining bits."""
        for p in self.order:
            if p == self.name:
                continue
            seed_b = bin(self.seeds[p])
            self.deltas[p] = int(seed_b[: self.bits_slice], 2)
            rest = seed_b[self.bits_slice:]
            seed = int(rest, 2) if rest else 0
            self.seeds[p] = random.Random(seed).randint(-sys.maxsize, sys.maxsize)

    def offset_total(self, active):
        """Algorithm 3: deltas of clients before CC are added, deltas of clients after CC are subtracted.

        The sign convention makes sum_i O_tot(i) = 0 over the active clients, so the offsets cancel in the
        server's average while hiding every individual L2.
        """
        assert self._agreed, "key agreement must run before offsets are used"
        o_tot, add = 0, True
        for c in self.order:
            if c not in active:
                continue
            O = self.deltas[c]
            if c == self.name:
                add = False          # line 9: for all C after CC, ADD = false
                o_tot -= O           # line 10 (O is 0 for the client itself)
            elif add:
                o_tot += O           # line 12
            else:
                o_tot -= O           # ADD = false branch
        return o_tot


def diffie_hellman_setup(names, curve="SECP384R1", bits_slice=20):
    """Create every client's DH state and run the three-phase key exchange among all clients."""
    parties = {n: DiffieHellmanParty(n, names, curve, bits_slice) for n in names}
    for n, party in parties.items():
        party.agree({p: parties[p].public_keys[n] for p in names if p != n})
    return parties


def add_offset(L2, o_tot):
    """Algorithm 3, line 15: L2 += O_tot (the summed offset, added once to avoid precision loss)."""
    return [np.asarray(a, np.float64) + float(o_tot) for a in L2]


# =====================================================================================================
# HE: Algorithm 4 (encryption) and Algorithm 5 (decryption)
# =====================================================================================================
OPS = ("add", "subtract", "multiply", "divide")
_OP_CODE = {op: i for i, op in enumerate(OPS)}


def _lists(a):
    """The lists L of one array: rows along axis 0 for weights, the whole vector for biases."""
    a = np.asarray(a, np.float64)
    return a.reshape(a.shape[0], -1) if a.ndim >= 2 else a.reshape(1, -1)


def _split(v):
    """E -> (E_int, E_flt): integer part and signed fractional part (same sign as E)."""
    e_int = np.trunc(v)
    return e_int.astype(np.int64), v - e_int


def _random_factor(operand, b_max, rng):
    """Per element, a pseudo-random B_I in 1..b_max that is a factor of the (non-zero) operand."""
    cand = np.arange(1, b_max + 1, dtype=np.int64)
    ok = (np.abs(operand)[:, None] % cand[None, :]) == 0          # 1 always divides
    pick = np.argmax(rng.random(ok.shape) * ok, axis=1)
    return cand[pick]


def _encrypt_list(L, o_p, b_max, rng):
    e_int, e_flt = _split(L)
    operand = e_int - o_p                       # E_int without the previous layer's offset O_tot
    # Zero-division guard: if any element is 0 without O_tot, division is not allowed for this list.
    allowed = OPS[:3] if np.any(operand == 0) else OPS
    op = allowed[rng.integers(len(allowed))]
    if op == "divide":
        B = _random_factor(operand, b_max, rng)                    # B_I is a factor of E_int (Alg.4 l.21)
    else:
        B = np.int64(rng.integers(1, b_max + 1))                  # one list-wise B_I (Alg.4 l.10)
    if op == "add":
        new_int = e_int + B                                         # Eq. (1)
    elif op == "subtract":
        new_int = e_int - B                                         # Eq. (2)
    elif op == "multiply":
        new_int = operand * B + o_p                                 # Eq. (3)
    else:
        new_int = operand // B + o_p                                # Eq. (4), exact because B | operand
    # If E_int and E_flt do not have the same sign, E_flt = -E_flt (Alg.4 l.13-14, l.23-24).
    flip = (np.sign(new_int) * np.sign(e_flt)) < 0
    e_flt = np.where(flip, -e_flt, e_flt)
    enc = new_int.astype(np.float64) + e_flt                        # E = E_int + E_flt (Alg.4 l.26)
    entry = {"op": _OP_CODE[op], "B": B, "flip": flip, "o_p": int(o_p), "enc": enc.copy()}
    return enc, entry


def he_encrypt(L2, o_p, b_max, rng):
    """Algorithm 4. Returns the encrypted L2 and the directories {"D_X": ..., "D_Y": ...}.

    o_p is the integer offset of the previous encryption layer (O_tot on clients, the fixed offset on the
    server). Each directory maps array index -> list of per-list entries (operation code, B_I, sign-flip
    mask, O_p and the encrypted values used for the integrity check of Algorithm 5).
    """
    directory = {"D_X": {}, "D_Y": {}}
    out = []
    for idx, a in enumerate(L2):
        rows = _lists(a)
        enc_rows, entries = np.empty_like(rows), []
        for r in range(rows.shape[0]):
            enc_rows[r], entry = _encrypt_list(rows[r], o_p, b_max, rng)
            entries.append(entry)
        directory["D_X" if is_weight_array(a) else "D_Y"][idx] = entries
        out.append(enc_rows.reshape(np.shape(a)))
    return out, directory


def _inverse(entry, x):
    x_int, x_flt = _split(x)
    x_flt = np.where(entry["flip"], -x_flt, x_flt)                  # undo the sign inversion
    B, o_p, op = entry["B"], entry["o_p"], OPS[entry["op"]]
    if op == "add":
        e_int = x_int - B
    elif op == "subtract":
        e_int = x_int + B
    elif op == "multiply":
        e_int = (x_int - o_p) // B + o_p                             # exact: (x_int - o_p) = operand * B
    else:
        e_int = (x_int - o_p) * B + o_p
    return e_int.astype(np.float64) + x_flt


def he_decrypt(L2_enc, directory):
    """Algorithm 5: INVERSE_{E_opt} element-wise, then remove the used entries from D_X / D_Y."""
    out = []
    for idx, a in enumerate(L2_enc):
        key = "D_X" if is_weight_array(a) else "D_Y"
        entries = directory[key].pop(idx)                            # line 9: remove E from D_X / D_Y
        rows = _lists(a)
        if len(entries) != rows.shape[0]:
            raise ValueError(f"HE directory does not match array {idx}")
        dec = np.empty_like(rows)
        for r, entry in enumerate(entries):
            # Integrity check: the called list must hold the same values as recorded in the directory.
            if not np.array_equal(rows[r], entry["enc"]):
                raise ValueError(f"HE integrity check failed for array {idx}, list {r}")
            dec[r] = _inverse(entry, rows[r])
        out.append(dec.reshape(np.shape(a)))
    return out
