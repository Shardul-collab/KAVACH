"""
coexistence.py — Quantum-classical coexistence noise: spontaneous Raman
scattering (SpRS) from classical WDM channels sharing fiber with the QKD
channel.

This is specifically Prof. Srivastava's group's stated angle (their 2024
high-dimensional-states coexistence paper, 2025 CV-QKD + classical channel
paper, 2026 SDN/DWDM coexistence paper) — modeling this alongside the
routing/RL layer (Bhatia's group's angle) is what actually bridges both
PIs' work in this consortium project.

PHYSICS:
A strong classical signal co-propagating in the same fiber as the quantum
channel generates broadband noise via spontaneous Raman scattering. A
fraction of that noise falls inside the quantum channel's wavelength band
and receiver filter bandwidth, arriving at the QKD detector as spurious
counts indistinguishable from real signal photons — functionally identical
to an elevated dark count rate.

Noise power scattered into the quantum channel (simplified, standard
engineering approximation — see caveats below):

    P_noise = P_classical * beta * L * Delta_lambda

    where:
      P_classical   = classical channel launch power (W)
      beta          = Raman scattering coefficient (1/(km*nm)) — depends on
                       fiber type and, critically, whether the quantum
                       channel sits at longer wavelength (anti-Stokes side,
                       lower beta, less noise) or shorter wavelength
                       (Stokes side, higher beta, more noise) than the
                       classical channel
      L             = fiber length (km)
      Delta_lambda  = quantum receiver's spectral filter bandwidth (nm)

Noise power is converted to a photon count rate via photon energy at the
quantum channel wavelength, then to an "effective extra dark count
probability per gate" for direct use with qkd_channel.compute_link's
extra_dark_count_prob parameter — this is what lets classical-channel
coexistence degrade QBER/SKR through the SAME formula path as detector
dark counts, rather than requiring a parallel channel model.

CAVEATS (be upfront about these if this comes up in an interview):
  - beta (Raman coefficient) values here are ORDER-OF-MAGNITUDE
    illustrative constants from the general coexistence literature, not
    measured/fitted to a specific fiber spool or the group's specific
    wavelength plan. Real work would use the exact Stokes/anti-Stokes
    coefficient curve for the fiber type and channel spacing in question.
  - This model assumes co-propagation (classical and quantum signals
    traveling the same direction) with continuous-wave classical power —
    counter-propagation and burst-mode classical traffic change the noise
    statistics and are not modeled.
  - Srivastava's more recent group papers (2025-2026) are on CV-QKD
    (continuous-variable), which uses homodyne/heterodyne detection with
    different noise statistics than the discrete-variable BB84 (single-
    photon detection) model this whole codebase is built on. This module
    approximates coexistence noise AS IF it were degrading a DV-QKD link
    (consistent with the rest of this codebase) — it is NOT a CV-QKD noise
    model. Flag this explicitly if asked; it's a real scope gap, not a
    subtle detail.
"""

from __future__ import annotations
from dataclasses import dataclass
import math

from .qkd_channel import QKDLinkParams, compute_link

H_PLANCK = 6.626e-34   # J*s
C_LIGHT = 3.0e8         # m/s


@dataclass
class RamanCoexistenceParams:
    classical_power_w: float = 5.0e-5    # 50 uW per-channel power reaching the shared span
                                          # after WDM mux/demux insertion losses (0dBm total
                                          # DWDM launch split/attenuated across mux stages) —
                                          # NOT raw per-channel launch power at the transmitter
    beta_per_km_nm: float = 1.0e-10      # order-of-magnitude Raman coefficient, anti-Stokes
                                          # side, lower end of the literature range (still
                                          # illustrative, not fit to a specific fiber/spacing)
    filter_bandwidth_nm: float = 0.05    # quantum receiver spectral filter bandwidth — dense
                                          # narrowband filtering (tens of pm), not a wide 200pm
                                          # passband
    quantum_wavelength_nm: float = 1550.0
    gate_width_ns: float = 1.0            # detector gate width around expected photon arrival


def raman_scattered_power_w(distance_km: float, params: RamanCoexistenceParams) -> float:
    """Noise power (W) scattered into the quantum channel band over the link."""
    return params.classical_power_w * params.beta_per_km_nm * distance_km * params.filter_bandwidth_nm


def photon_rate_from_power(power_w: float, wavelength_nm: float) -> float:
    """Convert optical power to photon count rate (photons/sec) at given wavelength."""
    photon_energy_j = H_PLANCK * C_LIGHT / (wavelength_nm * 1e-9)
    return power_w / photon_energy_j


def effective_extra_dark_count(distance_km: float, coexist: RamanCoexistenceParams,
                                qkd_params: QKDLinkParams) -> float:
    """
    Convert Raman-scattered noise power into an effective extra dark-count
    probability per detection gate, for direct injection into
    qkd_channel.compute_link.

    IMPORTANT: gated single-photon detectors (standard for QKD, e.g. InGaAs
    SPADs) only listen during a narrow window (~1 ns, `gate_width_ns`) around
    each expected pulse arrival — not continuously for the full inter-pulse
    period. An earlier version of this function divided the continuous noise
    photon rate by rep_rate_hz directly, which implicitly assumed a gate
    width equal to the ENTIRE inter-pulse period (e.g. 1000 ns at 1 MHz rep
    rate) — 1000x too wide — and produced unphysical QBER ~50% at every
    distance tested, even 10 km. Using the actual gate width fixes this.
    """
    noise_power = raman_scattered_power_w(distance_km, coexist)
    noise_photon_rate = photon_rate_from_power(noise_power, coexist.quantum_wavelength_nm)
    detected_noise_rate = noise_photon_rate * qkd_params.eta_det  # photons/sec
    gate_width_s = coexist.gate_width_ns * 1e-9
    p_extra = detected_noise_rate * gate_width_s   # expected noise photons per gate
    return min(p_extra, 1.0)


def compute_link_with_coexistence(
    distance_km: float,
    qkd_params: QKDLinkParams | None = None,
    coexist_params: RamanCoexistenceParams | None = None,
    classical_active: bool = True,
) -> dict:
    """
    Same return shape as qkd_channel.compute_link, but with Raman noise from
    a co-propagating classical channel folded in when classical_active=True.
    Also returns 'extra_dark_count' and 'classical_active' for diagnostics.
    """
    qp = qkd_params or QKDLinkParams()
    cp = coexist_params or RamanCoexistenceParams()

    extra = effective_extra_dark_count(distance_km, cp, qp) if classical_active else 0.0
    result = compute_link(distance_km, qp, extra_dark_count_prob=extra)
    result["extra_dark_count"] = extra
    result["classical_active"] = classical_active
    return result


if __name__ == "__main__":
    qkd_params = QKDLinkParams()
    coexist_params = RamanCoexistenceParams()

    print(f"{'dist(km)':>10} {'QBER (Q only)':>15} {'QBER (coexist)':>16} "
          f"{'SKR bps (Q only)':>18} {'SKR bps (coexist)':>19} {'coexist feasible':>17}")
    for d in [10, 25, 50, 75, 100, 125, 150]:
        quantum_only = compute_link_with_coexistence(d, qkd_params, coexist_params, classical_active=False)
        coexisting = compute_link_with_coexistence(d, qkd_params, coexist_params, classical_active=True)
        print(f"{d:>10} {quantum_only['qber']:>15.4f} {coexisting['qber']:>16.4f} "
              f"{quantum_only['skr_bps']:>18.2f} {coexisting['skr_bps']:>19.2f} "
              f"{str(coexisting['feasible']):>17}")

    print(f"\nMax reachable distance, quantum-only: "
          f"{max(d for d in range(1, 300) if compute_link_with_coexistence(d, qkd_params, coexist_params, False)['feasible'])} km")
    reachable_coexist = [d for d in range(1, 300)
                          if compute_link_with_coexistence(d, qkd_params, coexist_params, True)['feasible']]
    max_coexist = max(reachable_coexist) if reachable_coexist else 0
    print(f"Max reachable distance, with coexistence: {max_coexist} km")
