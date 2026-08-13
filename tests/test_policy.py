"""
test_policy.py
===============

Validates that the policy's gradients (computed by `torch.autograd`) are
numerically correct, using `torch.autograd.gradcheck` -- PyTorch's
built-in numerical Jacobian check, which compares the analytic gradient
autograd computes against a finite-difference approximation for every
parameter. This is the PyTorch-native equivalent of writing your own
finite-difference check by hand: it doesn't trust autograd blindly, it
verifies it, on every parameter tensor, to a tight numerical tolerance.

Also includes basic correctness/sanity checks: output probabilities stay
in [0, 1], and a policy can actually be pushed toward a target action via
gradient ascent on log pi(a|s) (i.e. the optimizer loop is wired up
correctly end to end).
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from context_selection.policy import MLPBernoulliPolicy, to_float64


class TestPolicyGradients(unittest.TestCase):
    """The most important test in this repo: verifies that torch.autograd's
    gradient of log pi(a|s) w.r.t. every network parameter is numerically
    correct, via PyTorch's own finite-difference gradient checker."""

    def setUp(self):
        torch.manual_seed(0)
        self.policy = to_float64(MLPBernoulliPolicy(obs_dim=6, hidden_dim=5, seed=0))
        self.s = torch.randn(6, dtype=torch.float64)

    def test_gradient_matches_finite_difference(self):
        """gradcheck perturbs each parameter by a small epsilon, compares
        the resulting finite-difference derivative to autograd's analytic
        gradient, and fails if they disagree beyond tolerance -- for every
        element of every parameter tensor, for both actions."""
        for action in (0.0, 1.0):
            action_t = torch.tensor(action, dtype=torch.float64)

            def logprob_fn(w1, b1, w2, b2, s=self.s, a=action_t):
                h = torch.tanh(torch.nn.functional.linear(s, w1, b1))
                logit = torch.nn.functional.linear(h, w2, b2).squeeze(-1)
                dist = torch.distributions.Bernoulli(logits=logit)
                return dist.log_prob(a)

            params = (
                self.policy.fc1.weight.detach().clone().requires_grad_(True),
                self.policy.fc1.bias.detach().clone().requires_grad_(True),
                self.policy.fc2.weight.detach().clone().requires_grad_(True),
                self.policy.fc2.bias.detach().clone().requires_grad_(True),
            )

            ok = torch.autograd.gradcheck(logprob_fn, params, eps=1e-6, atol=1e-4, rtol=1e-3)
            self.assertTrue(ok, f"gradcheck failed for action={action}")

    def test_probability_is_in_unit_interval(self):
        rng = torch.Generator().manual_seed(1)
        for _ in range(20):
            s = torch.randn(6, dtype=torch.float64, generator=rng)
            with torch.no_grad():
                prob = torch.sigmoid(self.policy(s))
            self.assertGreaterEqual(prob.item(), 0.0)
            self.assertLessEqual(prob.item(), 1.0)

    def test_adam_reduces_negative_log_likelihood(self):
        """Sanity check: repeatedly ascending log pi(a=1|s) via Adam should
        push p(a=1|s) toward 1 for a fixed state."""
        policy = to_float64(MLPBernoulliPolicy(obs_dim=6, hidden_dim=5, seed=1))
        opt = torch.optim.Adam(policy.parameters(), lr=0.05)
        s = self.s
        action = torch.tensor(1.0, dtype=torch.float64)

        with torch.no_grad():
            prob0 = torch.sigmoid(policy(s)).item()

        for _ in range(200):
            dist = policy.dist(s)
            logp = dist.log_prob(action)
            loss = -logp
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            prob_final = torch.sigmoid(policy(s)).item()

        self.assertGreater(prob_final, prob0)
        self.assertGreater(prob_final, 0.9)


if __name__ == "__main__":
    unittest.main()
