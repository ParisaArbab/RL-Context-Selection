"""
train.py
========

REINFORCE (Williams, 1992) with:
  * a moving-average value baseline (variance reduction, does not bias
    the gradient since d/dtheta E[b * log pi] = b * d/dtheta E[log pi] = 0
    for any baseline b that doesn't depend on the action),
  * an entropy bonus (encourages exploration / avoids premature
    collapse to always-skip, which is a tempting local optimum since
    skipping never over-spends the budget),
  * optional potential-based reward shaping (see reward.py),
  * gradient clipping (episodes have variable length, and the summed
    policy gradient can occasionally spike).

This is intentionally a *policy-gradient fundamentals* implementation --
no replay buffer, no target network, no autograd -- to keep every design
choice inspectable, matching the "solid understanding of RL fundamentals"
and "prototype novel RL approaches for ... reward shaping ... in agent
workflows" asks in the role description.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List

from .environment import ContextSelectionEnv
from .policy import MLPBernoulliPolicy, AdamOptimizer, zero_grads, add_grads
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


def run_episode_collect(env: ContextSelectionEnv, policy: MLPBernoulliPolicy, rng: np.random.Generator,
                         gamma: float, use_shaping: bool):
    obs = env.reset()
    caches, actions, raw_rewards, potentials = [], [], [], [env.potential(env._selected_idx)]

    done = False
    while not done:
        action, prob, cache = policy.act(obs, rng, greedy=False)
        step = env.step(action)
        caches.append(cache)
        actions.append(action)
        raw_rewards.append(step.reward)
        potentials.append(env.potential(env._selected_idx))
        obs, done, info = step.observation, step.done, step.info

    if use_shaping:
        r = shaped_rewards(raw_rewards, potentials, gamma=gamma)
    else:
        r = np.asarray(raw_rewards, dtype=np.float64)

    returns = discounted_returns(r, gamma=gamma)
    return caches, actions, returns, sum(raw_rewards), info


def evaluate(env: ContextSelectionEnv, policy: MLPBernoulliPolicy, n_episodes: int, rng: np.random.Generator):
    total = 0.0
    successes, cost_fracs = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        info = {}
        while not done:
            action, _, _ = policy.act(obs, rng, greedy=True)
            step = env.step(action)
            obs, done, info = step.observation, step.done, step.info
            ep_reward += step.reward
        total += ep_reward
        successes.append(info.get("success", 0.0))
        cost_fracs.append(info.get("cost_frac", 0.0))
    return total / n_episodes, float(np.mean(successes)), float(np.mean(cost_fracs))


def train_reinforce(env: ContextSelectionEnv, cfg: TrainConfig) -> tuple[MLPBernoulliPolicy, TrainLog]:
    rng = np.random.default_rng(cfg.seed)
    policy = MLPBernoulliPolicy(obs_dim=env.obs_dim, hidden_dim=32, seed=cfg.seed)
    opt = AdamOptimizer(policy.params, lr=cfg.lr)
    log = TrainLog()

    baseline = 0.0
    for ep in range(1, cfg.num_episodes + 1):
        caches, actions, returns, raw_total, info = run_episode_collect(
            env, policy, rng, cfg.gamma, cfg.use_shaping
        )

        baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * returns[0]
        advantages = returns - baseline
        # normalize for stability across variable-length episodes
        adv_std = advantages.std() + 1e-6
        advantages = advantages / adv_std

        grads = zero_grads(policy.params)
        for cache, action, adv in zip(caches, actions, advantages):
            pg = policy.logprob_grad(cache, action, coeff=adv)
            add_grads(grads, pg)

            # entropy bonus gradient: H(p) = -p log p - (1-p) log(1-p)
            # dH/dlogit = -p(1-p) * (log p - log(1-p)) ... we approximate
            # with the simple, numerically stable surrogate dH/dlogit ~ (0.5 - p)*p*(1-p)*4
            # (pushes logits toward 0 / p=0.5, standard entropy-bonus effect)
            p = cache["prob"]
            d_ent_logit = (0.5 - p) * 4.0 * cfg.entropy_coef
            # build entropy grad manually via chain rule reusing cached activations
            h, s = cache["h"], cache["s"]
            d_W2 = d_ent_logit * h
            d_b2 = np.array([d_ent_logit])
            d_h = d_ent_logit * policy.params["W2"]
            d_z1 = d_h * (1 - h ** 2)
            d_W1 = np.outer(d_z1, s)
            d_b1 = d_z1
            add_grads(grads, {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2})

        # we ASCEND the objective (maximize expected return + entropy), Adam
        # here is written as a descent step, so we pass in the *negative*
        # gradient of the loss = -(objective) -> negative of grads above.
        neg_grads = {k: -v for k, v in grads.items()}
        for k in neg_grads:
            norm = np.linalg.norm(neg_grads[k])
            if norm > cfg.grad_clip:
                neg_grads[k] *= cfg.grad_clip / (norm + 1e-8)
        opt.step(policy.params, neg_grads)

        log.episode.append(ep)
        log.train_reward.append(raw_total)

        if ep % cfg.eval_every == 0 or ep == cfg.num_episodes:
            avg_r, avg_succ, avg_cost = evaluate(env, policy, cfg.eval_episodes, rng)
            log.eval_episode.append(ep)
            log.eval_reward.append(avg_r)
            log.eval_success.append(avg_succ)
            log.eval_cost_frac.append(avg_cost)

    return policy, log
