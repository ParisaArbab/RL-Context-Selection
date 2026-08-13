"""
policy.py
=========

A 2-layer MLP Bernoulli policy for the context-selection MDP, implemented
in PyTorch:

    pi(a=1 | s) = sigmoid( W2 . tanh(W1 s + b1) + b2 )

as `nn.Linear(obs_dim, hidden_dim) -> tanh -> nn.Linear(hidden_dim, 1)`,
with the Bernoulli distribution, its log-prob, and its entropy all
delegated to `torch.distributions.Bernoulli` and differentiated by
`torch.autograd` -- no hand-derived backward pass.

Used by both trainers in this repo (`train.py`'s REINFORCE and `ppo.py`'s
clipped-surrogate PPO-lite). `tests/test_policy.py` validates the
resulting gradients against numerical finite differences via
`torch.autograd.gradcheck`, the standard way to check that an autograd
graph's analytic gradient is correct.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBernoulliPolicy(nn.Module):
    """pi(a=1 | s) = sigmoid( W2 . tanh(W1 s + b1) + b2 )"""

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
        """Returns (action: int, prob: float) for a single state."""
        logit = self.forward(s)
        prob = torch.sigmoid(logit)
        if greedy:
            action = int((prob >= 0.5).item())
        else:
            action = int(torch.bernoulli(prob).item())
        return action, float(prob.item())


def to_float64(module: nn.Module) -> nn.Module:
    """The environment/reward code is float64 NumPy throughout; casting the
    policy to float64 keeps observations and network math in the same
    precision end to end."""
    return module.double()


if __name__ == "__main__":
    # Quick smoke test / demo.
    policy = to_float64(MLPBernoulliPolicy(obs_dim=6, hidden_dim=5, seed=0))
    s = torch.randn(6, dtype=torch.float64)
    action, prob = policy.act(s)
    print(f"sample action={action}, prob(include)={prob:.4f}")
