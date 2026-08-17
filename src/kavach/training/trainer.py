"""
trainer.py — Train PPO and DQN agents on QKDRoutingEnv and compare against
the Phase 0 heuristic baselines (naive shortest-hop, key-rate-aware
widest-path). Matches the reference paper's comparison methodology
(policy-gradient NN vs DQN).

Usage:
    python3 -m kavach.trainer --topology nsfnet --timesteps 50000
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from ..envs.qkd_routing_env import QKDRoutingEnv
from ..envs.qkd_routing_env_buffered import QKDRoutingEnvBuffered


ENV_CLASSES = {"simple": QKDRoutingEnv, "buffered": QKDRoutingEnvBuffered}


def make_env(topology_name: str, seed: int, demands_per_episode: int = 50, env_type: str = "simple",
             classical_traffic_fraction: float = 0.0):
    EnvCls = ENV_CLASSES[env_type]
    return Monitor(EnvCls(topology_name=topology_name, seed=seed,
                           demands_per_episode=demands_per_episode,
                           classical_traffic_fraction=classical_traffic_fraction))


def heuristic_action(env: QKDRoutingEnv) -> int:
    """
    Widest-path-style heuristic evaluated ON THE SAME env/candidate structure
    RL uses: pick the feasible candidate with the highest bottleneck SKR that
    still meets the current demand's required rate; reject if none qualify.
    This runs through IDENTICAL sequential demands + active jamming/
    intercept-resend as PPO/DQN saw — Phase 0's demo.py compared heuristics
    under a static batch with no attacks, which is not the same test RL
    was evaluated under.
    """
    candidates = env._current_candidates
    _, _, required_rate = env._current_demand
    best_idx, best_skr = None, -1.0
    for i, cand in enumerate(candidates[: env.k]):
        if cand["feasible"] and cand["skr_bps"] >= required_rate and cand["skr_bps"] > best_skr:
            best_idx = i
            best_skr = cand["skr_bps"]
    return best_idx if best_idx is not None else env.k  # env.k = reject action


def evaluate_heuristic_policy(env: QKDRoutingEnv, n_episodes: int = 20) -> dict:
    served_totals, blocked_totals, reward_totals = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action = heuristic_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        served_totals.append(info["served"])
        blocked_totals.append(info["blocked"])
        reward_totals.append(ep_reward)

    served = np.array(served_totals)
    blocked = np.array(blocked_totals)
    return {
        "avg_served": served.mean(),
        "avg_blocked": blocked.mean(),
        "blocking_prob": blocked.mean() / (served.mean() + blocked.mean()),
        "avg_reward": np.mean(reward_totals),
    }


def evaluate_agent(model, env, n_episodes: int = 20) -> dict:
    served_totals, blocked_totals, reward_totals = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
        served_totals.append(info["served"])
        blocked_totals.append(info["blocked"])
        reward_totals.append(ep_reward)

    served = np.array(served_totals)
    blocked = np.array(blocked_totals)
    return {
        "avg_served": served.mean(),
        "avg_blocked": blocked.mean(),
        "blocking_prob": blocked.mean() / (served.mean() + blocked.mean()),
        "avg_reward": np.mean(reward_totals),
    }


def evaluate_random_policy(env, n_episodes: int = 20) -> dict:
    served_totals, blocked_totals = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        served_totals.append(info["served"])
        blocked_totals.append(info["blocked"])
    served = np.array(served_totals)
    blocked = np.array(blocked_totals)
    return {
        "avg_served": served.mean(),
        "avg_blocked": blocked.mean(),
        "blocking_prob": blocked.mean() / (served.mean() + blocked.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="nsfnet", choices=["nsfnet", "ubn24"])
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                     help="cpu is strongly recommended here — this MLP policy is tiny, "
                          "so GPU transfer overhead makes training SLOWER on GPU, not faster.")
    ap.add_argument("--env-type", default="simple", choices=["simple", "buffered"],
                     help="simple env or buffered env with resource depletion")
    ap.add_argument("--classical-fraction", type=float, default=0.0,
                     help="fraction of edges carrying co-propagating classical WDM traffic "
                          "(Raman-scattering coexistence noise, Phase 3). 0.0 = disabled, "
                          "matches all prior runs.")
    args = ap.parse_args()

    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "none"
    print(f"CUDA available: {cuda_ok} ({gpu_name})  |  training device: {args.device}")
    print(f"Classical-traffic (coexistence) fraction: {args.classical_fraction:.0%}")
    if args.device == "cpu" and cuda_ok:
        print("  Not using GPU by default - this policy is a tiny MLP (few thousand "
              "params); CPU<->GPU transfer overhead makes GPU SLOWER here, not faster. "
              "Pass --device cuda to force it anyway.\n"
              "  When this actually matters as the project scales: bigger policy_kwargs "
              "net_arch, a CNN/graph encoder over topology structure, or (the bigger lever, "
              "matters before GPU does) vectorizing many parallel envs with SubprocVecEnv "
              "instead of one env per run - that's what actually saturates a GPU for RL at "
              "this problem scale, not a single tiny MLP forward pass.\n")

    print(f"=== Kavach Phase 1: PPO vs DQN on {args.topology} ===\n")

    train_env = make_env(args.topology, seed=0, env_type=args.env_type,
                          classical_traffic_fraction=args.classical_fraction)
    EnvCls = ENV_CLASSES[args.env_type]
    eval_env = EnvCls(topology_name=args.topology, seed=None,
                       classical_traffic_fraction=args.classical_fraction)

    print(f"Random policy baseline ({args.eval_episodes} episodes)...")
    random_stats = evaluate_random_policy(eval_env, args.eval_episodes)
    print(f"  blocking_prob={random_stats['blocking_prob']:.2%}  "
          f"avg_served={random_stats['avg_served']:.1f}\n")

    print(f"Widest-path heuristic baseline ({args.eval_episodes} episodes, "
          f"SAME sequential demands + attacks as RL)...")
    heuristic_stats = evaluate_heuristic_policy(eval_env, args.eval_episodes)
    print(f"  blocking_prob={heuristic_stats['blocking_prob']:.2%}  "
          f"avg_served={heuristic_stats['avg_served']:.1f}  "
          f"avg_reward={heuristic_stats['avg_reward']:.2f}\n")

    print(f"Training PPO for {args.timesteps} timesteps (device={args.device})...")
    ppo_model = PPO("MlpPolicy", train_env, verbose=0, seed=0, device=args.device)
    ppo_model.learn(total_timesteps=args.timesteps)
    ppo_model.save(f"ppo_{args.topology}_{args.env_type}_cf{args.classical_fraction}")
    ppo_stats = evaluate_agent(ppo_model, eval_env, args.eval_episodes)
    print(f"  PPO: blocking_prob={ppo_stats['blocking_prob']:.2%}  "
          f"avg_served={ppo_stats['avg_served']:.1f}  avg_reward={ppo_stats['avg_reward']:.2f}\n")

    print(f"Training DQN for {args.timesteps} timesteps (device={args.device})...")
    train_env_dqn = make_env(args.topology, seed=1, env_type=args.env_type,
                              classical_traffic_fraction=args.classical_fraction)
    dqn_model = DQN("MlpPolicy", train_env_dqn, verbose=0, seed=1, device=args.device)
    dqn_model.learn(total_timesteps=args.timesteps)
    dqn_model.save(f"dqn_{args.topology}_{args.env_type}_cf{args.classical_fraction}")
    dqn_stats = evaluate_agent(dqn_model, eval_env, args.eval_episodes)
    print(f"  DQN: blocking_prob={dqn_stats['blocking_prob']:.2%}  "
          f"avg_served={dqn_stats['avg_served']:.1f}  avg_reward={dqn_stats['avg_reward']:.2f}\n")

    print("=== Summary ===")
    print(f"{'Policy':<24}{'Blocking prob':>16}{'Avg served/episode':>22}")
    print(f"{'Random':<24}{random_stats['blocking_prob']:>15.2%}{random_stats['avg_served']:>22.1f}")
    print(f"{'Widest-path heuristic':<24}{heuristic_stats['blocking_prob']:>15.2%}{heuristic_stats['avg_served']:>22.1f}")
    print(f"{'PPO':<24}{ppo_stats['blocking_prob']:>15.2%}{ppo_stats['avg_served']:>22.1f}")
    print(f"{'DQN':<24}{dqn_stats['blocking_prob']:>15.2%}{dqn_stats['avg_served']:>22.1f}")
    print("\nThis IS now an apples-to-apples comparison — heuristic ran through the")
    print("identical sequential demand stream + jamming/intercept-resend as PPO/DQN,")
    print("not Phase 0's static attack-free batch. If PPO/DQN beat the heuristic here,")
    print("that's the actual claim worth making; if they're close to or worse than it,")
    print("that's equally important to know before writing this up anywhere.")


if __name__ == "__main__":
    main()
