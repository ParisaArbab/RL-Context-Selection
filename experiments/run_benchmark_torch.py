"""
run_benchmark_torch.py
=======================

PyTorch equivalent of `run_benchmark.py`: same environment, same 3
methods (REINFORCE no-shaping / REINFORCE +shaping / PPO-lite +shaping),
same baselines, same 3 seeds, same eval protocol -- but the learned
policies are trained with `train_torch.train_reinforce_torch` /
`ppo_torch.train_ppo_torch` (torch.autograd) instead of the hand-derived
NumPy backward pass.

Saves:
    experiments/results/learning_curves_torch.png
    experiments/results/benchmark_comparison_torch.png
    experiments/results/results_torch.csv

Run with:  python experiments/run_benchmark_torch.py
"""
import os
import sys
import time
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from context_selection.environment import ContextSelectionEnv, EnvConfig
from context_selection.train import TrainConfig
from context_selection.train_torch import train_reinforce_torch, evaluate_torch
from context_selection.ppo import PPOConfig
from context_selection.ppo_torch import train_ppo_torch
from context_selection.baselines import RandomPolicy, GreedyRelevancePolicy, OraclePolicy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SEEDS = 3
N_EVAL_EPISODES = 300


def run_baseline(policy, env, n_episodes, seed):
    env.cfg.seed = seed
    rewards, successes, costs = [], [], []
    for _ in range(n_episodes):
        r, info = policy.run_episode(env)
        rewards.append(r)
        successes.append(info.get("success", 0.0))
        costs.append(info.get("cost_frac", 0.0))
    return float(np.mean(rewards)), float(np.mean(successes)), float(np.mean(costs)), float(np.std(rewards))


def main():
    t0 = time.time()
    env_cfg = EnvConfig(seed=0)

    all_learning_curves = {}
    final_scores = {}

    for seed in range(N_SEEDS):
        print(f"\n=== Seed {seed} ===")
        env = ContextSelectionEnv(EnvConfig(**{**env_cfg.__dict__, "seed": 1000 + seed}))

        # ---- REINFORCE, no shaping ----
        cfg = TrainConfig(num_episodes=3000, use_shaping=False, seed=seed, lr=6e-3, entropy_coef=0.015)
        print("Training REINFORCE-torch (no shaping)...")
        policy_ns, log_ns = train_reinforce_torch(env, cfg)
        all_learning_curves.setdefault("REINFORCE-torch (no shaping)", []).append(
            (log_ns.eval_episode, log_ns.eval_reward)
        )

        # ---- REINFORCE, with shaping ----
        cfg = TrainConfig(num_episodes=3000, use_shaping=True, seed=seed, lr=6e-3, entropy_coef=0.015)
        print("Training REINFORCE-torch (+ potential-based shaping)...")
        policy_s, log_s = train_reinforce_torch(env, cfg)
        all_learning_curves.setdefault("REINFORCE-torch (+ shaping)", []).append(
            (log_s.eval_episode, log_s.eval_reward)
        )

        # ---- PPO-lite, with shaping ----
        ppo_cfg = PPOConfig(num_updates=120, batch_episodes=12, epochs_per_update=4,
                             use_shaping=True, seed=seed, lr=6e-3, entropy_coef=0.015)
        print("Training PPO-lite-torch (+ shaping)...")
        policy_ppo, log_ppo = train_ppo_torch(env, ppo_cfg)
        all_learning_curves.setdefault("PPO-lite-torch (+ shaping)", []).append(
            (log_ppo.eval_episode, log_ppo.eval_reward)
        )

        # ---- Final held-out evaluation, same eval env/seed for all methods ----
        eval_env = ContextSelectionEnv(EnvConfig(**{**env_cfg.__dict__, "seed": 5000 + seed}))

        for name, pol in [
            ("REINFORCE-torch (no shaping)", policy_ns),
            ("REINFORCE-torch (+ shaping)", policy_s),
            ("PPO-lite-torch (+ shaping)", policy_ppo),
        ]:
            avg_r, avg_succ, avg_cost = evaluate_torch(eval_env, pol, N_EVAL_EPISODES)
            final_scores.setdefault(name, []).append((avg_r, avg_succ, avg_cost))

        eval_env2 = ContextSelectionEnv(EnvConfig(**{**env_cfg.__dict__, "seed": 5000 + seed}))
        r, s, c, _ = run_baseline(RandomPolicy(include_prob=0.5, seed=seed), eval_env2, N_EVAL_EPISODES, 5000 + seed)
        final_scores.setdefault("Random (p=0.5)", []).append((r, s, c))

        eval_env3 = ContextSelectionEnv(EnvConfig(**{**env_cfg.__dict__, "seed": 5000 + seed}))
        r, s, c, _ = run_baseline(GreedyRelevancePolicy(threshold=0.35), eval_env3, N_EVAL_EPISODES, 5000 + seed)
        final_scores.setdefault("Greedy-relevance", []).append((r, s, c))

        eval_env4 = ContextSelectionEnv(EnvConfig(**{**env_cfg.__dict__, "seed": 5000 + seed}))
        r, s, c, _ = run_baseline(OraclePolicy(), eval_env4, N_EVAL_EPISODES, 5000 + seed)
        final_scores.setdefault("Oracle (upper bound, uses labels)", []).append((r, s, c))

    # ---------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "results_torch.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_reward", "std_reward", "mean_success", "mean_cost_frac"])
        for name, vals in final_scores.items():
            vals = np.array(vals)
            w.writerow([
                name,
                f"{vals[:, 0].mean():.4f}",
                f"{vals[:, 0].std():.4f}",
                f"{vals[:, 1].mean():.4f}",
                f"{vals[:, 2].mean():.4f}",
            ])
    print(f"\nSaved {csv_path}")

    # ---------------------------------------------------------------
    # Plot 1: learning curves
    # ---------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    colors = {"REINFORCE-torch (no shaping)": "#d62728",
              "REINFORCE-torch (+ shaping)": "#1f77b4",
              "PPO-lite-torch (+ shaping)": "#2ca02c"}
    for name, runs in all_learning_curves.items():
        eps = runs[0][0]
        curves = np.array([r[1] for r in runs])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        plt.plot(eps, mean, label=name, color=colors.get(name))
        plt.fill_between(eps, mean - std, mean + std, alpha=0.15, color=colors.get(name))
    plt.xlabel("Training episodes")
    plt.ylabel("Held-out avg. episode reward (greedy policy)")
    plt.title("Context-Selection Policy Learning Curves (PyTorch)\n(mean +/- std over 3 seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "learning_curves_torch.png"), dpi=150)
    plt.close()

    # ---------------------------------------------------------------
    # Plot 2: final benchmark comparison bar chart
    # ---------------------------------------------------------------
    names = list(final_scores.keys())
    means = [np.array(final_scores[n])[:, 0].mean() for n in names]
    stds = [np.array(final_scores[n])[:, 0].std() for n in names]

    order = np.argsort(means)
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]

    plt.figure(figsize=(9, 5))
    bar_colors = ["#9467bd" if "Oracle" in n else
                  ("#7f7f7f" if n in ("Random (p=0.5)", "Greedy-relevance") else "#1f77b4")
                  for n in names]
    plt.barh(names, means, xerr=stds, color=bar_colors)
    plt.xlabel("Mean held-out episode reward (higher is better)")
    plt.title(f"Final Benchmark (PyTorch): {N_EVAL_EPISODES} held-out episodes x {N_SEEDS} seeds")
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "benchmark_comparison_torch.png"), dpi=150)
    plt.close()

    print(f"\nDone in {time.time() - t0:.1f}s. Plots + CSV written to {RESULTS_DIR}/")

    print("\n=== Final scores (mean over seeds) ===")
    for name in names:
        vals = np.array(final_scores[name])
        print(f"{name:38s}  reward={vals[:,0].mean():+.3f}  success={vals[:,1].mean():.3f}  cost_frac={vals[:,2].mean():.3f}")


if __name__ == "__main__":
    main()
