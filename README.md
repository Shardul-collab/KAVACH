# Kavach

Standalone simulation & resource-allocation research agent for QKD-secured
optical networks — built as prep/reproduction infrastructure for JRF work on
DRL-based routing and resource assignment in QKD-secured optical networks
(NSUT / IIT Indore consortium project on quantum communication for
telecommunication optical network security).

Not part of BRAHM. Structurally mirrors BRAHM's Vishwakarma agent pattern
(topology/input builder → physical-layer model → runner → evaluator), kept
separate because the domain (optical/quantum networking) is unrelated to
BRAHM's materials-science scope.

## Status: Phase 3 complete (quantum-classical coexistence)

- `kavach/topology.py` — **NSFNET and UBN24**, both with correctly modeled
  trusted-relay nodes (QKD can't use EDFA amplification — no-cloning
  theorem — so long backbone spans are split into ~80 km trusted-node hops,
  unlike classical WDM which amplifies end-to-end). `get_topology(name)`
  factory for config-driven selection.
- `kavach/qkd_channel.py` — BB84 physical layer: channel transmittance →
  gain → QBER → asymptotic secret key rate, simplified GLLP-style bound.
  `compute_link()` now accepts `extra_dark_count_prob` so coexistence noise
  (or any other extra-noise source) folds in through the same formula path
  rather than needing a parallel channel model.
- `kavach/baselines.py` — naive shortest-hop and key-rate-aware widest-path
  (maximin) routing, plus `k_shortest_candidates()` for the RL action space.
- `kavach/attack_models.py` — **jamming** (availability) and
  **intercept-resend** (QBER-degradation, eavesdropping), both wired into
  `rl_env.py` so the trained agent actually sees both attack types.
- `kavach/key_assignment.py` — FCFS/SJF/LRPT buffer-assignment ordering,
  verified to produce genuinely different blocking outcomes under real
  same-instant contention.
- `kavach/coexistence.py` — **Raman-scattering noise from co-propagating
  classical WDM channels** — Prof. Srivastava's group's specific angle.
  Converts classical launch power → Raman-scattered noise power → photon
  count rate → an effective extra dark-count probability per detector gate,
  injected through `qkd_channel.compute_link`'s `extra_dark_count_prob`.
  Verified sane: max reachable distance drops monotonically with classical
  power (-10dBm→42km, 0dBm→14km, +10dBm→2km), and degrades gracefully with
  distance rather than collapsing everywhere at once — the corrected version
  after fixing an initial bug (see caveat below).
- `kavach/rl_env.py` — Gymnasium env, SB3-verified, both jamming and
  intercept-resend active across the episode timeline.
- `kavach/trainer.py` — trains **PPO and DQN**, evaluates against random
  policy baseline.
- `demo.py` — Phase 0 static-batch heuristic comparison.

**A real physics bug worth knowing, not just the fix**: the first version of
`coexistence.py` divided the continuous Raman noise photon rate directly by
the pulse repetition rate to get a per-gate noise probability. That silently
assumes the detector integrates noise across the *entire* inter-pulse period
(e.g. 1000 ns at a 1 MHz rep rate) — but real gated single-photon detectors
(InGaAs SPADs, standard for fiber QKD) only listen during a narrow window
(~1 ns) around each expected photon arrival. Missing that gating factor
produced ~50% QBER at every distance including 10 km, which is not physically
plausible for known working coexistence demonstrations in the literature.
Adding an explicit `gate_width_ns` parameter fixed it. Worth understanding
this distinction cold — "why does gating matter" is a very plausible
interview question given this is the exact mechanism that makes coexistence
QKD deployments feasible at all.

Run it:
```bash
pip install -r requirements.txt
python3 demo.py                                                   # Phase 0
python3 -m kavach.attack_models                                   # jamming + intercept-resend
python3 -m kavach.key_assignment                                  # ordering policy comparison
python3 -m kavach.coexistence                                     # Raman coexistence degradation
python3 -m kavach.trainer --topology nsfnet --timesteps 50000      # Phase 1: PPO vs DQN
python3 -m kavach.trainer --topology ubn24  --timesteps 50000
```

## Roadmap

**Coexistence not yet wired into rl_env.py.** Same pattern as before —
the physical model is built and correct standalone, but the RL agent
doesn't yet route around coexistence-degraded links dynamically. Natural
next integration step, same shape as the jamming/intercept-resend wiring
already done.

**CV-QKD scope gap, flagged explicitly.** Everything in this codebase
(BB84 channel model, attacks, coexistence) assumes discrete-variable QKD
with single-photon detection. Srivastava's more recent papers (2025 CV-QKD
+ classical coexistence, 2026 SDN-enabled CV-QKD) are on continuous-variable
QKD — homodyne/heterodyne detection, different noise statistics entirely.
Not built here; know the distinction if it comes up.

**Phase 4 — MCF/SDM extension.** Multicore fiber support for the SDM-EON
environment named explicitly in the JRF ad, adding inter-core crosstalk
alongside Raman noise.

**Reward tuning.** Current RL reward weights are a reasonable guess, not
the paper's exact formula. Revisit once you have the paper's equations
directly.

## Physics model caveats (read before quoting numbers anywhere)

The BB84 secret-key-rate formula in `qkd_channel.py` is a simplified,
non-decoy-state, asymptotic model — good enough for routing/RL research
where relative comparisons between policies matter more than absolute
numbers, but **not** a faithful reproduction of any specific paper's results
yet. Before citing numbers in an interview or write-up, replace the default
`QKDLinkParams` with the exact parameter set from the specific paper you're
reproducing (detector efficiency, dark count rate, mean photon number, error
correction inefficiency all vary significantly across papers).
