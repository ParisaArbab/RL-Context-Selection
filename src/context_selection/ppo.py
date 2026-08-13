"""
ppo.py
======

A lightweight PPO-style clipped-surrogate trainer (Schulman et al., 2017)
for `MLPBernoulliPolicy`, using `torch.autograd`, as a second
policy-optimization strategy to compare against plain REINFORCE.

Simplifications relative to "full" PPO (deliberate, and noted so the
scope is honest):
  * No separate value network / GAE -- reuses the same moving-average
    baseline as REINFORCE, so any performance difference we see is
    attributable to the *clipped surrogate + multi-epoch reuse of
    on-policy data*, not to a better advantage estimator. This isolates
    the variable we actually want to study.
  * Batched over `batch_episodes` full episodes rather than a fixed
    number of environment steps.

Clipped surrogate objective, per (state, action) with importance ratio
    r = pi_new(a|s) / pi_old(a|s)
    L = min( r * A,  clip(r, 1-eps, 1+eps) * A )

`torch.clamp` has a kink at the clip boundary; PyTorch's subgradient
there (zero gradient for the clamped branch when it's the active
minimum) is exactly the standard PPO subgradient, so
`torch.min(unclipped, clipped)` reproduces the textbook update rule
directly via autograd, with no manual branch selection needed.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List

from .environment import ContextSelectionEnv
from .policy import MLPBernoulliPolicy, to_float64
from .train import run_episode_collect, evaluate, TrainLog


@dataclass
class PPOConfig:
    num_updates: int = 200
    batch_episodes: int = 10
    epochs_per_update: int = 4
    gamma: float = 0.97
    lr: float = 5e-3
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    baseline_momentum: float = 0.95
    use_shaping: bool = True
    grad_clip: float = 5.0
    eval_every_updates: int = 5
    eval_episodes: int = 50
    seed: int = 0


def _action_prob(prob: float, action: int) -> float:
    return prob if action == 1 else (1 - prob)


def train_ppo(env: ContextSelectionEnv, cfg: PPOConfig):
    torch.manual_seed(cfg.seed)
    policy = to_float64(MLPBernoulliPolicy(obs_dim=env.obs_dim, hidden_dim=32, seed=cfg.seed))
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    log = TrainLog()
    baseline = 0.0
    episodes_done = 0

    for update in range(1, cfg.num_updates + 1):
        # ---- collect a batch of on-policy rollouts with CURRENT params ----
        batch_states, batch_actions, batch_adv, batch_old_probs = [], [], [], []
        for _ in range(cfg.batch_episodes):
            states, actions, returns, raw_total, info = run_episode_collect(
                env, policy, cfg.gamma, cfg.use_shaping
            )
            baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * returns[0]
            adv = returns - baseline
            adv = adv / (adv.std() + 1e-6)

            with torch.no_grad():
                states_t = torch.as_tensor(np.stack(states), dtype=torch.float64)
                old_probs = torch.sigmoid(policy(states_t)).numpy()

            batch_states.append(states)
            batch_actions.append(actions)
            batch_adv.append(adv)
            batch_old_probs.append(old_probs)
            log.train_reward.append(raw_total)
            episodes_done += 1

        all_states = np.concatenate([np.stack(s) for s in batch_states])
        all_actions = np.concatenate(batch_actions)
        all_adv = np.concatenate(batch_adv)
        all_old_probs = np.concatenate(batch_old_probs)

        states_t = torch.as_tensor(all_states, dtype=torch.float64)
        actions_t = torch.as_tensor(all_actions, dtype=torch.float64)
        adv_t = torch.as_tensor(all_adv, dtype=torch.float64)
        # p_old(a taken), fixed constants from the rollout (no grad)
        old_action_probs_t = torch.as_tensor(
            np.where(all_actions == 1, all_old_probs, 1 - all_old_probs), dtype=torch.float64
        )

        # ---- multiple epochs of clipped-surrogate updates on this batch ----
        for _epoch in range(cfg.epochs_per_update):
            dist = policy.dist(states_t)
            new_probs = torch.sigmoid(policy(states_t))
            new_action_probs = torch.where(actions_t == 1, new_probs, 1 - new_probs)

            ratio = new_action_probs / (old_action_probs_t + 1e-8)
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
            surrogate = torch.min(unclipped, clipped)

            entropy = dist.entropy()

            loss = -surrogate.mean() - cfg.entropy_coef * entropy.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip)
            opt.step()

        log.episode.append(episodes_done)

        if update % cfg.eval_every_updates == 0 or update == cfg.num_updates:
            avg_r, avg_succ, avg_cost = evaluate(env, policy, cfg.eval_episodes)
            log.eval_episode.append(episodes_done)
            log.eval_reward.append(avg_r)
            log.eval_success.append(avg_succ)
            log.eval_cost_frac.append(avg_cost)

    return policy, log
