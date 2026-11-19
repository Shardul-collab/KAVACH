# Kavach

Simulation and routing research for QKD-secured optical networks.

Kavach asks one question: can a reinforcement-learning agent route
key-generation requests through a QKD network better than a classical
heuristic, when links are attacked and classical traffic adds noise?

So far, the answer is **no**: on NSFNET, PPO and DQN come close to a
key-rate-aware widest-path heuristic but do not beat it (results below).

## What is modelled

| Module | What it does |
|---|---|
| `kavach.core.topology` | NSFNET and UBN24 topologies with trusted-relay nodes. QKD signals cannot be optically amplified, so long spans are split into ~80 km trusted-node hops. |
| `kavach.domains.qkd.qkd_channel` | BB84 physical layer: transmittance → gain → QBER → asymptotic secret key rate (simplified GLLP-style bound, no decoy states). |
| `kavach.domains.qkd.coexistence` | Raman-scattering noise from co-propagating classical WDM channels, converted into an extra dark-count probability per detector gate. |
| `kavach.domains.qkd.attack_models` | Jamming (link availability) and intercept-resend (QBER increase). |
| `kavach.domains.qkd.baselines` | Shortest-hop and key-rate-aware widest-path routing, plus k-shortest candidates for the RL action space. |
| `kavach.domains.qkd.key_assignment` | FCFS / SJF / LRPT ordering of key requests under contention. |
| `kavach.envs` | Gymnasium routing environments (simple, buffered, and a GNN variant). |
| `kavach.training` | PPO and DQN training (Stable-Baselines3) with MLP or GNN policies, evaluated against random and heuristic baselines. |

## Results so far

NSFNET, 50,000 training timesteps, 20 evaluation episodes, with jamming and
intercept-resend active. The heuristic runs on the same demand stream and
attacks as the RL agents. From the logs in `results/logs/`.

| Policy | Blocking probability (no coexistence) | Blocking probability (30% classical traffic) |
|---|---|---|
| Random | 63.2% | 64.7% |
| Widest-path heuristic | **24.6%** | **25.0%** |
| PPO | 25.9% | 27.1% |
| DQN | 25.3% | 26.2% |

Open question: why the agents do not learn what the heuristic already does.

### Coexistence: effect of classical launch power

Maximum reachable single-link distance with the default parameters, from
`kavach.domains.qkd.coexistence`:

| Classical power per channel | Max reach |
|---|---|
| quantum channel alone | 238 km |
| −10 dBm | 116 km |
| 0 dBm | 75 km |
| +10 dBm | 39 km |

The Raman coefficient is an order-of-magnitude literature value, not fitted
to a specific fibre, so treat these as trends rather than predictions.

### A bug worth recording

The first version of the coexistence model divided the Raman noise photon
rate by the pulse repetition rate. That assumes the detector integrates noise
over the whole inter-pulse period, but gated InGaAs single-photon detectors
only listen for ~1 ns around each expected arrival. Without the gating factor
the model gave ~50% QBER even at 10 km, which contradicts working coexistence
experiments. An explicit `gate_width_ns` parameter fixed it.

## Run it

```bash
pip install -e .

# Physics modules (each prints a self-test table)
python -m kavach.core.topology
python -m kavach.domains.qkd.qkd_channel
python -m kavach.domains.qkd.coexistence
python -m kavach.domains.qkd.baselines
python -m kavach.domains.qkd.attack_models
python -m kavach.domains.qkd.key_assignment

# RL training: PPO vs DQN vs baselines
python -m kavach.training.trainer --topology nsfnet --timesteps 50000
python -m kavach.training.trainer --topology nsfnet --timesteps 50000 --classical-fraction 0.3
```

An MCP server (`mcp_server.py`) exposes training runs as background jobs to
Claude Desktop; see `claude_desktop_config.example.json`.

## Limitations

- Discrete-variable BB84 only. Continuous-variable QKD (homodyne or
  heterodyne detection) is not modelled.
- The key-rate formula is simplified, asymptotic and has no decoy states. It
  is suitable for comparing routing policies, not for reproducing a specific
  paper's absolute key rates without first setting that paper's detector
  efficiency, dark-count rate, mean photon number and error-correction
  efficiency.
- RL reward weights are hand-chosen, not taken from a published formulation.
- Multicore-fibre (SDM) crosstalk is planned but not built.
