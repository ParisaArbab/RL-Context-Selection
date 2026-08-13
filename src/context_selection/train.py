"""
train.py
========

REINFORCE (Williams, 1992) for the context-selection policy, implemented
with `torch.autograd` and `torch.optim.Adam`, with:
  * a moving-average value baseline (variance reduction, does not bias
    the gradient since d/dtheta E[b * log pi] = b * d/dtheta E[log pi] = 0
    for any baseline b that doesn't depend on the action),
  * an entropy bonus (`torch.distributions.Bernoulli.entropy()`,
    differentiated exactly by autograd) to encourage exploration and
    avoid premature collapse to always-skip, which is a tempting local
    optimum since skipping never over-spends the budget,
  * optional potential-based reward shaping (see reward.py),
  * gradient-norm clipping (episodes have variable length, and the
    summed policy gradient can occasionally spike).
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List

from .environment import ContextSelectionEnv
from .policy import MLPBernoulliPolicy, to_float64
from .reward import shaped_rewards, discounted_returns


@dataclass
class TrainConfig:
    num_episodes: int = 2000
    gamma: float = 0.97
    lr: float = 5e-3
    entropy_coef: float = 0.01
    baseline_momentum: float = 0.95
    use_shaping: bool = True
    grad_clip: float = 5.0
    eval_every: int = 100
    eval_episodes: int = 50
    seed: int = 0


@dataclass
class TrainLog:
    episode: List[int] = field(default_factory=list)
    train_reward: List[float] = field(default_factory=list)
    eval_reward: List[float] = field(default_factory=list)
    eval_episode: List[int] = field(default_factory=list)
    eval_success: List[float] = field(default_factory=list)
    eval_cost_frac: List[float] = field(default_factory=list)


def run_episode_collect(env: ContextSelectionEnv, policy: MLPBernoulliPolicy,
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


def evaluate(env: ContextSelectionEnv, policy: MLPBernoulliPolicy,
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


def train_reinforce(env: ContextSelectionEnv, cfg: TrainConfig) -> tuple[MLPBernoulliPolicy, TrainLog]:
    torch.manual_seed(cfg.seed)
    policy = to_float64(MLPBernoulliPolicy(obs_dim=env.obs_dim, hidden_dim=32, seed=cfg.seed))
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    log = TrainLog()

    baseline = 0.0
    for ep in range(1, cfg.num_episodes + 1):
        states, actions, returns, raw_total, info = run_episode_collect(
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
            avg_r, avg_succ, avg_cost = evaluate(env, policy, cfg.eval_episodes)
            log.eval_episode.append(ep)
            log.eval_reward.append(avg_r)
            log.eval_success.append(avg_succ)
            log.eval_cost_frac.append(avg_cost)

    return policy, log
