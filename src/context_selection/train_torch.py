"""
train_torch.py
===============

PyTorch reimplementation of `train.py`'s REINFORCE trainer, using
`torch.autograd` and `torch.optim.Adam` in place of the hand-derived
NumPy backward pass and manual Adam.

Kept identical to the NumPy version:
  * same environment (`ContextSelectionEnv`), same reward shaping
    (`reward.shaped_rewards` / `discounted_returns`), same moving-average
    scalar baseline and advantage normalization, same gradient-norm
    clipping value, same eval protocol (greedy actions, held-out
    episodes).

One deliberate, documented difference:
  * The NumPy trainer uses a hand-written *surrogate* for the entropy
    bonus gradient (see the comment in `train.py`: `dH/dlogit ~
    (0.5 - p) * p * (1-p) * 4`, an approximation chosen to keep the
    manual backprop simple). This PyTorch version uses
    `torch.distributions.Bernoulli(logits=...).entropy()`, which is the
    *exact* analytic entropy, differentiated exactly by autograd. This
    is one of the concrete benefits of the port -- it removes an
    approximation that was only there because manual backprop made the
    exact entropy gradient more annoying to derive by hand. See
    report.md for a discussion of what this changes in practice.

Because this trainer uses its own independent random draws (torch's
RNG, not NumPy's, and a different code path for action sampling), a
given `seed` does **not** reproduce bit-identical episodes to the NumPy
trainer with the same seed -- both are seeded and internally
reproducible, but they are not cross-reproducible with each other. This
is expected and does not affect the validity of either.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List

from .environment import ContextSelectionEnv
from .policy_torch import MLPBernoulliPolicyTorch, to_float64
from .reward import shaped_rewards, discounted_returns
from .train import TrainConfig, TrainLog  # reuse the same config/log schema


def run_episode_collect_torch(env: ContextSelectionEnv, policy: MLPBernoulliPolicyTorch,
                               gamma: float, use_shaping: bool):
    obs = env.reset()
    states, actions, raw_rewards, potentials = [], [], [], [env.potential(env._selected_idx)]

    done = False
    while not done:
        s_t = torch.as_tensor(obs, dtype=torch.float64)
        action, _ = policy.act(s_t, greedy=False)
        step = env.step(action)
        states.append(obs)
        actions.append(action)
        raw_rewards.append(step.reward)
        potentials.append(env.potential(env._selected_idx))
        obs, done, info = step.observation, step.done, step.info

    if use_shaping:
        r = shaped_rewards(raw_rewards, potentials, gamma=gamma)
    else:
        r = np.asarray(raw_rewards, dtype=np.float64)

    returns = discounted_returns(r, gamma=gamma)
    return states, actions, returns, sum(raw_rewards), info


def evaluate_torch(env: ContextSelectionEnv, policy: MLPBernoulliPolicyTorch,
                    n_episodes: int) -> tuple[float, float, float]:
    total = 0.0
    successes, cost_fracs = [], []
    with torch.no_grad():
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            info = {}
            while not done:
                s_t = torch.as_tensor(obs, dtype=torch.float64)
                action, _ = policy.act(s_t, greedy=True)
                step = env.step(action)
                obs, done, info = step.observation, step.done, step.info
                ep_reward += step.reward
            total += ep_reward
            successes.append(info.get("success", 0.0))
            cost_fracs.append(info.get("cost_frac", 0.0))
    return total / n_episodes, float(np.mean(successes)), float(np.mean(cost_fracs))


def train_reinforce_torch(env: ContextSelectionEnv, cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    policy = to_float64(MLPBernoulliPolicyTorch(obs_dim=env.obs_dim, hidden_dim=32, seed=cfg.seed))
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    log = TrainLog()

    baseline = 0.0
    for ep in range(1, cfg.num_episodes + 1):
        states, actions, returns, raw_total, info = run_episode_collect_torch(
            env, policy, cfg.gamma, cfg.use_shaping
        )

        baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * returns[0]
        advantages = returns - baseline
        adv_std = advantages.std() + 1e-6
        advantages = advantages / adv_std

        states_t = torch.as_tensor(np.stack(states), dtype=torch.float64)
        actions_t = torch.as_tensor(actions, dtype=torch.float64)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float64)

        dist = policy.dist(states_t)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy()

        # ascend E[log pi * A] + entropy_coef * H  =>  minimize the negative
        loss = -(log_probs * advantages_t).sum() - cfg.entropy_coef * entropy.sum()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip)
        opt.step()

        log.episode.append(ep)
        log.train_reward.append(raw_total)

        if ep % cfg.eval_every == 0 or ep == cfg.num_episodes:
            avg_r, avg_succ, avg_cost = evaluate_torch(env, policy, cfg.eval_episodes)
            log.eval_episode.append(ep)
            log.eval_reward.append(avg_r)
            log.eval_success.append(avg_succ)
            log.eval_cost_frac.append(avg_cost)

    return policy, log
