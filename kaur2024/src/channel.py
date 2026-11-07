"""Channel and energy model. [PAPER] Eqs. 1-4 (p.4) and Sec. V-A (p.9); missing pieces are
[GAP-FILL G10, G11, G18] (see SPEC.md). The user has said the energy constants are not a focus:
the defaults are used as proposed and are not studied further.

Units: joules, watts, hertz, bits.
"""
from __future__ import annotations

import numpy as np


def dbm_to_w(dbm):
    return 10.0 ** ((np.asarray(dbm, dtype=np.float64) - 30.0) / 10.0)


class Channel:
    def __init__(self, ch: dict, n_clients: int, rng: np.random.Generator):
        self.c, self.n, self.rng = ch, n_clients, rng
        self.gain = self.draw()

    def draw(self) -> np.ndarray:
        """[PAPER] Rayleigh fading, average path loss 1e-3: power gain = 1e-3 * Exp(1).
        [GAP-FILL G11] redrawn per device per round from the training seed."""
        self.gain = self.c["path_loss"] * self.rng.exponential(1.0, self.n)
        return self.gain

    def snr(self, power_w, bandwidth_hz, gain=None):
        """[PAPER] zeta = P g / N0, with N0 as a density times bandwidth [GAP-FILL G11]."""
        g = self.gain if gain is None else gain
        noise_w = dbm_to_w(self.c["n0_dbm_per_hz"]) * bandwidth_hz
        return np.asarray(power_w) * g / noise_w

    def state_snr_db(self, k: int) -> np.ndarray:
        """[GAP-FILL G18] SNR seen in the DQN state: max power, bandwidth B_max / k."""
        s = self.snr(dbm_to_w(self.c["state_snr_power_dbm"]), self.c["b_max_hz"] / k)
        return 10.0 * np.log10(s)

    def energy(self, n_samples, d, n_params, power_w, n_selected, gain):
        """Per-device energies for the selected devices (arrays aligned with n_samples).
        Eq. 1: E_data = kappa * D * varsigma * upsilon^2
        Eq. 2: E_comp = kappa * D * varsigma * K_L * m * upsilon^2
        Eq. 3: E_comm = rho * q / (B log2(1 + zeta)), rho = P [GAP-FILL G11], U = 1, t' = 1
        """
        c = self.c
        D_bits = np.asarray(n_samples, np.float64) * d * c["quant_bits"]       # [GAP-FILL G11]
        m_mbit = n_params * c["quant_bits"] / 1e6                               # [GAP-FILL G11]
        q_bits = n_params * c["quant_bits"]                                     # [GAP-FILL G11]
        base = c["kappa_cpu"] * D_bits * c["cycles_per_bit"] * c["cpu_hz"] ** 2
        e_data = base                                                           # [PAPER] Eq. 1
        e_comp = base * c["k_l"] * m_mbit                                       # [PAPER] Eq. 2
        B = c["b_max_hz"] / n_selected                                          # [GAP-FILL G11]
        rate = B * np.log2(1.0 + self.snr(power_w, B, gain))
        e_comm = np.asarray(power_w) * q_bits / rate                            # [PAPER] Eq. 3
        return e_data, e_comp, e_comm
