"""
policy_torch.py
================

PyTorch reimplementation of `context_selection.policy.MLPBernoulliPolicy`,
using `torch.autograd` instead of the hand-derived NumPy backward pass.

Same architecture as the NumPy version:

    pi(a=1 | s) = sigmoid( W2 . tanh(W1 s + b1) + b2 )

implemented here as `nn.Linear(obs_dim, hidden_dim) -> tanh -> nn.Linear(hidden_dim, 1)`,
with the Bernoulli distribution and its log-prob / entropy delegated to
`torch.distributions.Bernoulli` rather than derived by hand.

This module is the canonical policy used by `train_torch.py` and
`ppo_torch.py`. It is validated against the from-scratch NumPy
implementation in two ways:

  1. `load_numpy_weights` lets a torch policy be initialized with the
     *exact* weights of a `MLPBernoulliPolicy`, so the two forward passes
     are computing the same function on the same input.
  2. `tests/test_policy_torch.py` uses that to check that
     `torch.autograd`'s gradient of log pi(a|s) w.r.t. every parameter
     matches `MLPBernoulliPolicy.logprob_grad`'s hand-derived gradient
     *exactly* (to float64 numerical precision) -- not just similar in
     sign/magnitude, and not a finite-difference approximation on either
     side. `tests/test_policy.py`'s finite-difference check validates the
     NumPy gradient is correct; this test validates the two
     implementations agree with each other.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPBernoulliPolicyTorch(nn.Module):
    """Same architecture/interface as MLPBernoulliPolicy, autograd-backed."""

    def __init__(self, obs_dim: int, hidden_dim: int = 32, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Returns the Bernoulli logit for INCLUDE (a=1)."""
        h = torch.tanh(self.fc1(s))
        logit = self.fc2(h).squeeze(-1)
        return logit

    def dist(self, s: torch.Tensor) -> torch.distributions.Bernoulli:
        return torch.distributions.Bernoulli(logits=self.forward(s))

    @torch.no_grad()
    def act(self, s: torch.Tensor, greedy: bool = False):
        """Returns (action:int, prob:float) -- mirrors MLPBernoulliPolicy.act's
        return signature (minus the NumPy 'cache' dict, which autograd makes
        unnecessary: the graph itself is the cache)."""
        logit = self.forward(s)
        prob = torch.sigmoid(logit)
        if greedy:
            action = int((prob >= 0.5).item())
        else:
            action = int(torch.bernoulli(prob).item())
        return action, float(prob.item())


def load_numpy_weights(torch_policy: MLPBernoulliPolicyTorch, numpy_params: dict) -> None:
    """Copies weights from a NumPy `MLPBernoulliPolicy.params` dict into a
    `MLPBernoulliPolicyTorch`, so both compute the identical function.
    Used only for the gradient cross-check test -- training does not use
    this (each implementation initializes and trains its own weights)."""
    with torch.no_grad():
        torch_policy.fc1.weight.copy_(torch.as_tensor(numpy_params["W1"], dtype=torch.float64))
        torch_policy.fc1.bias.copy_(torch.as_tensor(numpy_params["b1"], dtype=torch.float64))
        torch_policy.fc2.weight.copy_(torch.as_tensor(numpy_params["W2"], dtype=torch.float64).reshape(1, -1))
        torch_policy.fc2.bias.copy_(torch.as_tensor(numpy_params["b2"], dtype=torch.float64))


def to_float64(module: nn.Module) -> nn.Module:
    """The NumPy implementation is float64 throughout; casting the torch
    module to float64 removes float32-vs-float64 as a confound when
    cross-checking gradients."""
    return module.double()


if __name__ == "__main__":
    # Quick smoke test / demo (see tests/test_policy_torch.py for the real
    # cross-check assertions run under the test suite).
    policy = to_float64(MLPBernoulliPolicyTorch(obs_dim=6, hidden_dim=5, seed=0))
    s = torch.randn(6, dtype=torch.float64)
    action, prob = policy.act(s)
    print(f"sample action={action}, prob(include)={prob:.4f}")
