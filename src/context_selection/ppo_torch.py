"""
ppo_torch.py
============

PyTorch reimplementation of `ppo.py`'s clipped-surrogate PPO-lite
trainer, using `torch.autograd` in place of the hand-derived
sub-gradient of `min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`.

Kept identical to the NumPy version: same simplifications (no separate
value network / GAE, reuses the same moving-average scalar baseline as
REINFORCE), same batch/epoch structure, same clip epsilon, same
gradient-norm clipping, same eval protocol.

`torch.clamp` is not differentiable at the clip boundary in the
mathematical sense (it has a kink), but PyTorch's subgradient there
(zero gradient for the clamped branch) is exactly the standard PPO
subgradient the NumPy version derives by hand in `ppo.py`'s comments --
so `torch.min(unclipped, clipped)` with `torch.clamp` reproduces the
same update rule, just via autograd instead of an explicit branch
selection.

Same caveat as train_torch.py: independent RNG stream from the NumPy
trainer, so a shared `seed` does not reproduce identical rollouts
across the two implementations.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List

from .environment import ContextSelectionEnv
from .policy_torch import MLPBernoulliPolicyTorch, to_float64
from .train_torch import run_episode_collect_torch, evaluate_torch
from .train import TrainLog
from .ppo import PPOConfig  # reuse the same config schema


def _action_prob(prob: float, action: int) -> float:
    return prob if action == 1 else (1 - prob)


def train_ppo_torch(env: ContextSelectionEnv, cfg: PPOConfig):
    torch.manual_seed(cfg.seed)
    policy = to_float64(MLPBernoulliPolicyTorch(obs_dim=env.obs_dim, hidden_dim=32, seed=cfg.seed))
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    log = TrainLog()
    baseline = 0.0
    episodes_done = 0

    for update in range(1, cfg.num_updates + 1):
        # ---- collect a batch of on-policy rollouts with CURRENT params ----
        batch_states, batch_actions, batch_adv, batch_old_probs = [], [], [], []
        for _ in range(cfg.batch_episodes):
            states, actions, returns, raw_total, info = run_episode_collect_torch(
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
            avg_r, avg_succ, avg_cost = evaluate_torch(env, policy, cfg.eval_episodes)
            log.eval_episode.append(episodes_done)
            log.eval_reward.append(avg_r)
            log.eval_success.append(avg_succ)
            log.eval_cost_frac.append(avg_cost)

    return policy, log
