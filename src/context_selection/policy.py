"""
policy.py
=========

A tiny 2-layer MLP Bernoulli policy, implemented **from scratch in NumPy**
(forward pass, manual backprop, Adam optimizer) rather than delegated to
an autograd framework.

Why implement it by hand instead of just using PyTorch?
---------------------------------------------------------
Two reasons, both intentional:

1. **This sandbox has no GPU / no PyTorch installed**, and I did not want
   to hand in a project that only "works in theory." Everything in this
   repo actually runs and produces the plots in `experiments/results/`.
2. Writing the REINFORCE and PPO gradients by hand forces you to get the
   score-function estimator, the baseline subtraction, and the clipped
   surrogate exactly right -- it's the same derivation you'd sanity-check
   against `torch.autograd` in a real project, just made explicit here.
   `tests/test_policy.py` includes a finite-difference gradient check
   against this manual backward pass.

A parallel, idiomatic **PyTorch** reimplementation of the same policy is
included in `torch_reference/policy_torch.py` for when a GPU/PyTorch
environment is available (e.g. on Pokee's infra) -- same architecture,
same interface, drop-in replacement.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class AdamOptimizer:
    """Minimal Adam optimizer over a dict of NumPy parameter arrays."""

    def __init__(self, params: dict, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class MLPBernoulliPolicy:
    """pi(a=1 | s) = sigmoid( W2 . tanh(W1 s + b1) + b2 )"""

    def __init__(self, obs_dim: int, hidden_dim: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.params = {
            "W1": rng.normal(scale=scale1, size=(hidden_dim, obs_dim)),
            "b1": np.zeros(hidden_dim),
            "W2": rng.normal(scale=scale2, size=hidden_dim),
            "b2": np.zeros(1),
        }
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

    # ------------------------------------------------------------------
    def forward(self, s: np.ndarray, params: dict | None = None):
        p = params if params is not None else self.params
        z1 = p["W1"] @ s + p["b1"]
        h = np.tanh(z1)
        logit = float(p["W2"] @ h + p["b2"][0])
        prob = _sigmoid(logit)
        cache = dict(s=s, z1=z1, h=h, logit=logit, prob=prob)
        return prob, cache

    def act(self, s: np.ndarray, rng: np.random.Generator, greedy: bool = False):
        prob, cache = self.forward(s)
        if greedy:
            action = int(prob >= 0.5)
        else:
            action = int(rng.random() < prob)
        return action, prob, cache

    # ------------------------------------------------------------------
    # Manual backprop: d/dtheta [ scalar_coeff * log pi(a|s) ]
    # ------------------------------------------------------------------
    def logprob_grad(self, cache: dict, action: int, coeff: float, params: dict | None = None):
        """Returns grads (same shape as params) for `coeff * log pi(a|s)`.

        Bernoulli log-prob:      log pi = a*log(p) + (1-a)*log(1-p)
        d logpi / d logit     =  a - p                     (standard result)
        then backprop through  logit = W2.h + b2,  h = tanh(z1),  z1 = W1 s + b1
        """
        p = params if params is not None else self.params
        prob = cache["prob"]
        h = cache["h"]
        s = cache["s"]
        z1 = cache["z1"]

        d_logit = (action - prob) * coeff           # dL/dlogit

        d_W2 = d_logit * h
        d_b2 = np.array([d_logit])

        d_h = d_logit * p["W2"]
        d_z1 = d_h * (1 - h ** 2)                    # tanh'
        d_W1 = np.outer(d_z1, s)
        d_b1 = d_z1

        return {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}

    def logprob(self, cache: dict, action: int) -> float:
        p = np.clip(cache["prob"], 1e-8, 1 - 1e-8)
        return float(action * np.log(p) + (1 - action) * np.log(1 - p))

    def clone_params(self):
        return {k: v.copy() for k, v in self.params.items()}


def zero_grads(params: dict) -> dict:
    return {k: np.zeros_like(v) for k, v in params.items()}


def add_grads(acc: dict, new: dict):
    for k in acc:
        acc[k] += new[k]
    return acc
