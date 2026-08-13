"""
baselines.py
============

Reference policies to benchmark the learned RL policy against:

  * RandomPolicy          -- includes each item with fixed probability p,
                              subject to budget. Sanity-check floor.
  * GreedyRelevancePolicy -- a strong, common-sense heuristic an engineer
                              would reach for first: include an item iff
                              its cosine similarity to the query embedding
                              clears a threshold and the budget allows it.
                              This is the policy an RL approach actually
                              has to beat to justify the extra complexity.
  * OraclePolicy           -- uses the environment's ground-truth item
                              labels (relevant/redundant/distractor),
                              which a real policy never has access to.
                              This is an *upper bound*, not a baseline to
                              beat -- it tells us how much headroom is
                              left above the greedy heuristic.

All policies share the same `select(env) -> (selected_idx, total_reward, info)`
interface so `experiments/run_benchmark.py` can treat them uniformly.
"""

from __future__ import annotations
import numpy as np
from .environment import ContextSelectionEnv


class RandomPolicy:
    def __init__(self, include_prob: float = 0.5, seed: int = 0):
        self.include_prob = include_prob
        self.rng = np.random.default_rng(seed)

    def run_episode(self, env: ContextSelectionEnv):
        obs = env.reset()
        done = False
        total_reward = 0.0
        info = {}
        while not done:
            action = int(self.rng.random() < self.include_prob)
            step = env.step(action)
            obs, r, done, info = step.observation, step.reward, step.done, step.info
            total_reward += r
        return total_reward, info


class GreedyRelevancePolicy:
    """Include iff cos_sim(item, query) > threshold and budget allows.

    Reads the item embedding and query embedding straight out of the
    observation vector -- it does NOT use the ground-truth relevance
    labels, so it's a fair baseline (same information a real retrieval
    -augmented agent would have: embeddings + a similarity threshold).
    """

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def run_episode(self, env: ContextSelectionEnv):
        obs = env.reset()
        d = env.cfg.embed_dim
        done = False
        total_reward = 0.0
        info = {}
        while not done:
            item = obs[:d]
            query = obs[d + 1: d + 1 + d]
            denom = (np.linalg.norm(item) * np.linalg.norm(query) + 1e-8)
            sim = float(item @ query) / denom
            action = int(sim > self.threshold)
            step = env.step(action)
            obs, r, done, info = step.observation, step.reward, step.done, step.info
            total_reward += r
        return total_reward, info


class OraclePolicy:
    """Upper-bound reference using ground-truth labels (not a fair baseline)."""

    def run_episode(self, env: ContextSelectionEnv):
        env.reset()
        target_idx = set(env.oracle_selection())
        done = False
        total_reward = 0.0
        info = {}
        t = 0
        while not done:
            action = int(t in target_idx)
            step = env.step(action)
            done, r, info = step.done, step.reward, step.info
            total_reward += r
            t += 1
        return total_reward, info
