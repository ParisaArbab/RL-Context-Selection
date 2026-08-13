"""
policy_torch.py
================

A PyTorch reimplementation of `context_selection.policy.MLPBernoulliPolicy`,
same architecture and interface, for use once GPU/PyTorch infra is
available (this sandbox had neither -- see the top-level README for why
the NumPy version is the one all experiments in this repo actually run
on). Included so the from-scratch derivation in `policy.py` can be
cross-checked against `torch.autograd`, and as a starting point for
scaling this up (batched rollouts, GPU, larger networks) on real infra.

NOT executed as part of this repo's test suite / experiments (no torch
in the sandbox); install `torch` and run the `__main__` block below to
sanity check it produces gradients consistent with the NumPy version on
a fixed state/action/seed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBernoulliPolicyTorch(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(s))
        logit = self.fc2(h).squeeze(-1)
        return logit  # return logits; use torch.distributions.Bernoulli(logits=logit)

    def act(self, s: torch.Tensor):
        logit = self.forward(s)
        dist = torch.distributions.Bernoulli(logits=logit)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


def reinforce_update(policy: MLPBernoulliPolicyTorch, optimizer: torch.optim.Optimizer,
                      log_probs: list[torch.Tensor], advantages: torch.Tensor,
                      entropies: list[torch.Tensor], entropy_coef: float = 0.01):
    """Standard PyTorch REINFORCE update -- the natural analogue of the
    manual `logprob_grad` + Adam loop in `context_selection/train.py`,
    provided for parity/scale-up once torch is available."""
    log_probs_t = torch.stack(log_probs)
    entropies_t = torch.stack(entropies)
    loss = -(log_probs_t * advantages).sum() - entropy_coef * entropies_t.sum()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    # Minimal cross-check against the NumPy implementation's math: same
    # architecture, same random weights, same state/action -> same
    # d(logpi)/d(logit) sign and rough magnitude.
    torch.manual_seed(0)
    policy = MLPBernoulliPolicyTorch(obs_dim=6, hidden_dim=5)
    s = torch.randn(6)
    logit = policy(s)
    dist = torch.distributions.Bernoulli(logits=logit)
    action = torch.tensor(1.0)
    logp = dist.log_prob(action)
    logp.backward()
    print("torch grad on fc2.weight:", policy.fc2.weight.grad)
    print("(compare against context_selection.policy.logprob_grad on an")
    print(" equivalent NumPy-initialized network -- see tests/test_policy.py")
    print(" for the from-scratch version's own finite-difference check.)")
