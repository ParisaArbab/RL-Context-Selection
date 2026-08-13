"""
run_benchmark.py
=================

End-to-end experiment:
  1. Train REINFORCE *without* reward shaping.
  2. Train REINFORCE *with* potential-based reward shaping.
  3. Train the PPO-lite trainer (with shaping).
  4. Evaluate all three learned policies plus Random / Greedy / Oracle
     baselines on a held-out set of fresh episodes.
  5. Save:
       experiments/results/learning_curves.png
       experiments/results/benchmark_comparison.png
       experiments/results/results.csv

Run with:  python experiments/run_benchmark.py
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
from context_selection.train import TrainConfig, train_reinforce, evaluate
from context_selection.ppo import PPOConfig, train_ppo
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

    all_learning_curves = {}   # name -> list of (eval_episode, eval_reward) per seed
    final_scores = {}          # name -> list of (reward, success, cost) per seed

    for seed in range(N_SEEDS):
        print(f"\n=== Seed {seed} ===")
        env = ContextSelectionEnv(EnvConfig(**{**env_cfg.__dict__, "seed": 1000 + seed}))

        # ---- REINFORCE, no shaping ----
        cfg = TrainConfig(num_episodes=3000, use_shaping=False, seed=seed, lr=6e-3, entropy_coef=0.015)
        print("Training REINFORCE (no shaping)...")
        policy_ns, log_ns = train_reinforce(env, cfg)
        all_learning_curves.setdefault("REINFORCE (no shaping)", []).append(
            (log_ns.eval_episode, log_ns.eval_reward)
        )

        # ---- REINFORCE, with shaping ----
        cfg = TrainConfig(num_episodes=3000, use_shaping=True, seed=seed, lr=6e-3, entropy_coef=0.015)
        print("Training REINFORCE (+ potential-based shaping)...")
        policy_s, log_s = train_reinforce(env, cfg)
        all_learning_curves.setdefault("REINFORCE (+ shaping)", []).append(
            (log_s.eval_episode, log_s.eval_reward)
        )

        # ---- PPO-lite, with shaping ----
        ppo_cfg = PPOConfig(num_updates=120, batch_episodes=12, epochs_per_update=4,
                             use_shaping=True, seed=seed, lr=6e-3, entropy_coef=0.015)
        print("Training PPO-lite (+ shaping)...")
        policy_ppo, log_ppo = train_ppo(env, ppo_cfg)
        all_learning_curves.setdefault("PPO-lite (+ shaping)", []).append(
            (log_ppo.eval_episode, log_ppo.eval_reward)
        )

        # ---- Final held-out evaluation, same eval env/seed for all methods ----
        eval_env = ContextSelectionEnv(EnvConfig(**{**env_cfg.__dict__, "seed": 5000 + seed}))
        rng = np.random.default_rng(9999 + seed)

        for name, pol in [
            ("REINFORCE (no shaping)", policy_ns),
            ("REINFORCE (+ shaping)", policy_s),
            ("PPO-lite (+ shaping)", policy_ppo),
        ]:
            avg_r, avg_succ, avg_cost = evaluate(eval_env, pol, N_EVAL_EPISODES, rng)
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
    # Save results.csv
    # ---------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
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
    # Plot 1: learning curves (mean +/- std across seeds)
    # ---------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    colors = {"REINFORCE (no shaping)": "#d62728",
              "REINFORCE (+ shaping)": "#1f77b4",
              "PPO-lite (+ shaping)": "#2ca02c"}
    for name, runs in all_learning_curves.items():
        eps = runs[0][0]
        curves = np.array([r[1] for r in runs])  # seeds x evals
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        plt.plot(eps, mean, label=name, color=colors.get(name))
        plt.fill_between(eps, mean - std, mean + std, alpha=0.15, color=colors.get(name))
    plt.xlabel("Training episodes")
    plt.ylabel("Held-out avg. episode reward (greedy policy)")
    plt.title("Context-Selection Policy Learning Curves\n(mean +/- std over 3 seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"), dpi=150)
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
    plt.title(f"Final Benchmark: {N_EVAL_EPISODES} held-out episodes x {N_SEEDS} seeds")
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "benchmark_comparison.png"), dpi=150)
    plt.close()

    print(f"\nDone in {time.time() - t0:.1f}s. Plots + CSV written to {RESULTS_DIR}/")

    print("\n=== Final scores (mean over seeds) ===")
    for name in names:
        vals = np.array(final_scores[name])
        print(f"{name:38s}  reward={vals[:,0].mean():+.3f}  success={vals[:,1].mean():.3f}  cost_frac={vals[:,2].mean():.3f}")


if __name__ == "__main__":
    main()
