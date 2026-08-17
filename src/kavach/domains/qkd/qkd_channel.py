"""
qkd_channel.py — BB84 physical-layer model: channel transmittance, QBER,
and asymptotic secret key rate as a function of fiber distance.

MODEL (simplified, non-decoy-state BB84 — matches the level of detail used
in a first-pass simulation study; see references below for where to tighten
this once we're reproducing a specific paper's exact numbers):

  Channel transmittance:      eta_ch = 10^(-alpha * L / 10)
  Total transmittance:        eta = eta_ch * eta_det
  Gain (detection probability per pulse):
      Q_mu = 1 - (1 - p_dark) * exp(-mu * eta)   [~ p_dark + mu*eta for small mu*eta]
  QBER:
      QBER = (p_dark/2 + e_detector * mu * eta) / Q_mu
  Asymptotic secret key rate (bits/pulse), simplified GLLP-style bound:
      R = 0.5 * Q_mu * (1 - f_EC * h2(QBER) - h2(QBER))     if QBER < QBER_max
      R = 0                                                  otherwise
  Secret key rate in bits/sec:
      SKR = R * R_rep

  This is deliberately the same shape of model used in this lab's own
  "Numerical Analysis of QBER and Secure Key Rate under Realistic
  Communication Parameters" work — swap in their exact parameter set
  once you have the paper in hand for a faithful reproduction.

Caveats to flag explicitly when this gets used for real comparison:
  - No decoy-state analysis (real BB84 deployments use decoy states to
    bound photon-number-splitting attacks; this model does NOT capture that).
  - No finite-key statistical corrections (asymptotic-only).
  - Error-correction inefficiency f_EC is a fixed constant, not protocol-specific.
"""

from __future__ import annotations
from dataclasses import dataclass
import math


def h2(p: float) -> float:
    """Binary Shannon entropy, safe at p=0 and p=1."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


@dataclass
class QKDLinkParams:
    alpha_db_km: float = 0.2       # standard SMF attenuation at 1550nm
    eta_det: float = 0.6           # detector efficiency
    p_dark: float = 1e-6           # dark count probability per gate
    mu: float = 0.5                # mean photon number per pulse (weak coherent pulse)
    e_detector: float = 0.02       # detector intrinsic error rate (misalignment etc.)
    f_ec: float = 1.16             # error-correction inefficiency (typical value)
    rep_rate_hz: float = 1e6       # pulse repetition rate
    qber_max: float = 0.11         # BB84 security threshold (~11%, standard bound)


def channel_transmittance(distance_km: float, params: QKDLinkParams) -> float:
    eta_ch = 10 ** (-params.alpha_db_km * distance_km / 10.0)
    return eta_ch * params.eta_det


def compute_link(distance_km: float, params: QKDLinkParams | None = None,
                  extra_dark_count_prob: float = 0.0) -> dict:
    """
    Returns a dict: {distance_km, eta, qber, gain, skr_bps, feasible}
    feasible=False means QBER exceeds the security threshold -> no usable key.

    `extra_dark_count_prob` adds to the detector's intrinsic dark count
    probability per gate — used by coexistence.py to inject Raman-scattering
    noise from a co-propagating classical channel as an effective elevated
    dark count rate, without duplicating this function's logic.
    """
    p = params or QKDLinkParams()
    eta = channel_transmittance(distance_km, p)
    p_dark_total = min(p.p_dark + extra_dark_count_prob, 1.0)

    gain = 1 - (1 - p_dark_total) * math.exp(-p.mu * eta)
    gain = max(gain, 1e-15)  # avoid div-by-zero at extreme distance

    qber = (p_dark_total / 2 + p.e_detector * p.mu * eta) / gain
    qber = min(qber, 1.0)

    feasible = qber < p.qber_max
    if feasible:
        r_per_pulse = 0.5 * gain * (1 - p.f_ec * h2(qber) - h2(qber))
        r_per_pulse = max(r_per_pulse, 0.0)
    else:
        r_per_pulse = 0.0

    skr_bps = r_per_pulse * p.rep_rate_hz

    return {
        "distance_km": distance_km,
        "eta": eta,
        "gain": gain,
        "qber": qber,
        "skr_bps": skr_bps,
        "feasible": feasible and skr_bps > 0,
    }


def max_reachable_distance(params: QKDLinkParams | None = None,
                            step_km: float = 1.0, max_km: float = 400.0) -> float:
    """Binary-search-free linear scan for the distance at which SKR hits 0. Diagnostic only."""
    p = params or QKDLinkParams()
    last_ok = 0.0
    d = 0.0
    while d <= max_km:
        result = compute_link(d, p)
        if result["feasible"]:
            last_ok = d
        d += step_km
    return last_ok


if __name__ == "__main__":
    params = QKDLinkParams()
    print(f"{'dist(km)':>10} {'eta':>12} {'QBER':>8} {'SKR(bps)':>14} {'feasible':>10}")
    for d in [10, 25, 50, 75, 100, 125, 150, 175, 200]:
        r = compute_link(d, params)
        print(f"{d:>10} {r['eta']:>12.3e} {r['qber']:>8.4f} {r['skr_bps']:>14.2f} {str(r['feasible']):>10}")
    print(f"\nMax reachable distance @ default params: ~{max_reachable_distance(params):.0f} km")
