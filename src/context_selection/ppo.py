"""
ppo.py
======

A lightweight, from-scratch PPO-style trainer (Schulman et al., 2017)
for the same `MLPBernoulliPolicy`, used as a second policy-optimization
strategy to compare against plain REINFORCE.

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

Manual gradient: because the policy is a single Bernoulli logit, r is a
simple function of the new probability p (old probability is fixed from
the rollout), so we differentiate the min(...) by selecting whichever
branch is smaller at the current p (the standard sub-gradient PPO
implementations use under the hood, made explicit here instead of
hidden inside autograd).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List

from .environment import ContextSelectionEnv
from .policy import MLPBernoulliPolicy, AdamOptimizer, zero_grads, add_grads
from .reward import shaped_rewards, discounted_returns
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
    rng = np.random.default_rng(cfg.seed)
    policy = MLPBernoulliPolicy(obs_dim=env.obs_dim, hidden_dim=32, seed=cfg.seed)
    opt = AdamOptimizer(policy.params, lr=cfg.lr)
    log = TrainLog()
    baseline = 0.0
    episodes_done = 0

    for update in range(1, cfg.num_updates + 1):
        # ---- collect a batch of on-policy rollouts with CURRENT params ----
        batch = []
        for _ in range(cfg.batch_episodes):
            caches, actions, returns, raw_total, info = run_episode_collect(
                env, policy, rng, cfg.gamma, cfg.use_shaping
            )
            baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * returns[0]
            adv = returns - baseline
            adv = adv / (adv.std() + 1e-6)
            old_probs = [c["prob"] for c in caches]
            batch.append((caches, actions, adv, old_probs))
            log.train_reward.append(raw_total)
            episodes_done += 1

        # ---- multiple epochs of clipped-surrogate updates on this batch ----
        for _epoch in range(cfg.epochs_per_update):
            grads = zero_grads(policy.params)
            n_terms = 0
            for caches, actions, adv, old_probs in batch:
                for cache_old, action, A, p_old_full in zip(caches, actions, adv, old_probs):
                    # re-run forward pass with CURRENT params at the same state
                    s = cache_old["s"]
                    prob_new, cache_new = policy.forward(s)

                    p_old_a = _action_prob(p_old_full, action)
                    p_new_a = _action_prob(prob_new, action)
                    ratio = p_new_a / (p_old_a + 1e-8)

                    unclipped = ratio * A
                    clipped = np.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * A

                    # sub-gradient of min(): use branch achieving the min;
                    # if clipped branch is active AND clipping is "binding"
                    # in the direction that would increase the objective
                    # further, its gradient w.r.t. ratio is zero (this is
                    # exactly what makes PPO's update conservative).
                    use_unclipped = unclipped <= clipped
                    is_clipped_saturated = not (1 - cfg.clip_eps < ratio < 1 + cfg.clip_eps)

                    if use_unclipped or not is_clipped_saturated:
                        dL_dratio = A
                    else:
                        dL_dratio = 0.0

                    # dratio/dp_new_a = 1 / p_old_a ; dp_new_a/dlogit = +-p(1-p)
                    dratio_dp_new_a = 1.0 / (p_old_a + 1e-8)
                    sign = 1.0 if action == 1 else -1.0
                    dp_new_a_dlogit = sign * prob_new * (1 - prob_new)
                    d_logit = dL_dratio * dratio_dp_new_a * dp_new_a_dlogit

                    h, s_ = cache_new["h"], cache_new["s"]
                    d_W2 = d_logit * h
                    d_b2 = np.array([d_logit])
                    d_h = d_logit * policy.params["W2"]
                    d_z1 = d_h * (1 - h ** 2)
                    d_W1 = np.outer(d_z1, s_)
                    d_b1 = d_z1

                    # entropy bonus (same surrogate as REINFORCE trainer)
                    d_ent_logit = (0.5 - prob_new) * 4.0 * cfg.entropy_coef
                    d_W2 += d_ent_logit * h
                    d_b2 += np.array([d_ent_logit])
                    d_h_e = d_ent_logit * policy.params["W2"]
                    d_z1_e = d_h_e * (1 - h ** 2)
                    d_W1 += np.outer(d_z1_e, s_)
                    d_b1 += d_z1_e

                    add_grads(grads, {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2})
                    n_terms += 1

            # average over collected terms, ascend, clip
            neg_grads = {k: -v / max(1, n_terms) for k, v in grads.items()}
            for k in neg_grads:
                norm = np.linalg.norm(neg_grads[k])
                if norm > cfg.grad_clip:
                    neg_grads[k] *= cfg.grad_clip / (norm + 1e-8)
            opt.step(policy.params, neg_grads)

        log.episode.append(episodes_done)

        if update % cfg.eval_every_updates == 0 or update == cfg.num_updates:
            avg_r, avg_succ, avg_cost = evaluate(env, policy, cfg.eval_episodes, rng)
            log.eval_episode.append(episodes_done)
            log.eval_reward.append(avg_r)
            log.eval_success.append(avg_succ)
            log.eval_cost_frac.append(avg_cost)

    return policy, log
