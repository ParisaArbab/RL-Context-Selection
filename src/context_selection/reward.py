"""
reward.py
=========

Potential-based reward shaping (Ng, Harada & Russell, 1999).

The environment only gives a *terminal* reward (task success minus cost
and redundancy penalties). That's a hard credit-assignment problem: early
"include a clearly relevant item" decisions and late "stop padding the
context, you already have enough" decisions get the same delayed signal.

Potential-based shaping adds a term

    F(s, s') = gamma * Phi(s') - Phi(s)

to every step's reward, where Phi(s) is *any* function of state (here:
current coverage of the query -- see `ContextSelectionEnv.potential`).
The key theoretical property (Ng et al., 1999, Theorem 1) is that this
transformation is **policy invariant**: the optimal policy under the
shaped reward is identical to the optimal policy under the original
reward, for any Phi and any gamma in [0, 1]. Shaping can only change the
*speed* of learning, not the *target* of learning -- which is exactly
why it's safe to use here rather than hand-crafting an ad hoc dense
reward that might silently change what the agent is optimizing for.

This module exposes:
  * `shaped_rewards(raw_rewards, potentials, gamma)` -- turns a sparse
    terminal-only reward trajectory into a shaped, per-step trajectory.
  * `discounted_returns(rewards, gamma)` -- standard return computation,
    used identically for shaped and unshaped experiments so comparisons
    are apples-to-apples.
"""

from __future__ import annotations
from typing import List
import numpy as np


def shaped_rewards(raw_rewards: List[float], potentials: List[float], gamma: float = 0.99) -> np.ndarray:
    """
    raw_rewards[t]  : reward received on transition t -> t+1 (only the
                       last entry is non-zero in this environment)
    potentials[t]   : Phi(s_t) for t = 0 .. T   (length = len(raw_rewards) + 1)

    Returns an array of shaped rewards of the same length as raw_rewards:
        r'_t = r_t + gamma * Phi(s_{t+1}) - Phi(s_t)
    """
    raw_rewards = np.asarray(raw_rewards, dtype=np.float64)
    potentials = np.asarray(potentials, dtype=np.float64)
    assert len(potentials) == len(raw_rewards) + 1, "need Phi for every state incl. terminal"
    shaping = gamma * potentials[1:] - potentials[:-1]
    return raw_rewards + shaping


def discounted_returns(rewards: List[float], gamma: float = 0.99) -> np.ndarray:
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns
