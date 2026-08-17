"""
train_gnn.py — Train PPO/DQN with a graph-neural-network policy on
QKDRoutingEnvGNN, and compare against the same widest-path heuristic used
in kavach/trainer.py (heuristic_action is imported unchanged, not
reimplemented — same convention export_trace.py already follows).

Usage:
    python3 train_gnn.py --topology nsfnet --timesteps 50000
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from ..envs.qkd_routing_env_gnn import QKDRoutingEnvGNN
from .trainer import heuristic_action  # reused unchanged, per repo convention
from ..policies.gnn_extractor import GNNFeaturesExtractor


def make_env(topology_name: str, seed: int, demands_per_episode: int = 50,
             classical_traffic_fraction: float = 0.0):
    return Monitor(QKDRoutingEnvGNN(topology_name=topology_name, seed=seed,
                                     demands_per_episode=demands_per_episode,
                                     classical_traffic_fraction=classical_traffic_fraction))


def evaluate_heuristic_policy(env, n_episodes: int = 20) -> dict:
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
    served, blocked = np.array(served_totals), np.array(blocked_totals)
    return {
        "avg_served": served.mean(), "avg_blocked": blocked.mean(),
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
    served, blocked = np.array(served_totals), np.array(blocked_totals)
    return {
        "avg_served": served.mean(), "avg_blocked": blocked.mean(),
        "blocking_prob": blocked.mean() / (served.mean() + blocked.mean()),
        "avg_reward": np.mean(reward_totals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="nsfnet", choices=["nsfnet", "ubn24"])
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                     help="unlike trainer.py's tiny MLP, this network is ~1.6M params — "
                          "GPU is worth using here. Pass --device cuda or auto.")
    ap.add_argument("--classical-fraction", type=float, default=0.0)
    ap.add_argument("--hidden-dim", type=int, default=384)
    ap.add_argument("--gnn-layers", type=int, default=3)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "both"])
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"=== Kavach GNN policy: {args.algo.upper()} on {args.topology} ===\n")

    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractor,
        features_extractor_kwargs=dict(hidden_dim=args.hidden_dim, n_layers=args.gnn_layers),
    )

    train_env = make_env(args.topology, seed=0, classical_traffic_fraction=args.classical_fraction)
    eval_env = QKDRoutingEnvGNN(topology_name=args.topology, seed=None,
                                 classical_traffic_fraction=args.classical_fraction)

    print(f"Widest-path heuristic baseline ({args.eval_episodes} episodes)...")
    heuristic_stats = evaluate_heuristic_policy(eval_env, args.eval_episodes)
    print(f"  blocking_prob={heuristic_stats['blocking_prob']:.2%}  "
          f"avg_served={heuristic_stats['avg_served']:.1f}  "
          f"avg_reward={heuristic_stats['avg_reward']:.2f}\n")

    results = {"Widest-path heuristic": heuristic_stats}

    if args.algo in ("ppo", "both"):
        print(f"Training PPO (GNN policy) for {args.timesteps} timesteps...")
        model = PPO("MultiInputPolicy", train_env, policy_kwargs=policy_kwargs,
                    verbose=0, seed=0, device=device)
        n_params = sum(p.numel() for p in model.policy.parameters())
        print(f"  total policy parameters: {n_params:,}")
        model.learn(total_timesteps=args.timesteps)
        model.save(f"gnn_ppo_{args.topology}_cf{args.classical_fraction}")
        stats = evaluate_agent(model, eval_env, args.eval_episodes)
        print(f"  PPO(GNN): blocking_prob={stats['blocking_prob']:.2%}  "
              f"avg_served={stats['avg_served']:.1f}  avg_reward={stats['avg_reward']:.2f}\n")
        results["PPO (GNN)"] = stats

    if args.algo in ("dqn", "both"):
        print(f"Training DQN (GNN policy) for {args.timesteps} timesteps...")
        train_env_dqn = make_env(args.topology, seed=1, classical_traffic_fraction=args.classical_fraction)
        model = DQN("MultiInputPolicy", train_env_dqn, policy_kwargs=policy_kwargs,
                    verbose=0, seed=1, device=device)
        n_params = sum(p.numel() for p in model.policy.parameters())
        print(f"  total policy parameters: {n_params:,}")
        model.learn(total_timesteps=args.timesteps)
        model.save(f"gnn_dqn_{args.topology}_cf{args.classical_fraction}")
        stats = evaluate_agent(model, eval_env, args.eval_episodes)
        print(f"  DQN(GNN): blocking_prob={stats['blocking_prob']:.2%}  "
              f"avg_served={stats['avg_served']:.1f}  avg_reward={stats['avg_reward']:.2f}\n")
        results["DQN (GNN)"] = stats

    print("=== Summary ===")
    print(f"{'Policy':<24}{'Blocking prob':>16}{'Avg served/episode':>22}")
    for name, stats in results.items():
        print(f"{name:<24}{stats['blocking_prob']:>15.2%}{stats['avg_served']:>22.1f}")


if __name__ == "__main__":
    main()
