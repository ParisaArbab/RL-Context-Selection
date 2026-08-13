"""
test_policy_torch.py
=====================

Cross-checks the PyTorch policy (`policy_torch.MLPBernoulliPolicyTorch`,
gradients via `torch.autograd`) against the from-scratch NumPy policy
(`policy.MLPBernoulliPolicy`, gradients via hand-derived backprop in
`logprob_grad`).

This is NOT a finite-difference check (that already exists in
`test_policy.py` and validates the NumPy gradient in isolation). This
test loads *identical* weights into both implementations, runs the same
state/action through both, and asserts the two gradients agree to within
float64 numerical tolerance -- i.e. it checks that autograd and the
hand-derived backward pass are computing the same function, not just
gradients that are "roughly consistent."

Skips cleanly (rather than failing) if torch is not installed, since the
rest of the test suite (`test_policy.py`, `test_environment.py`,
`test_reward.py`) has no torch dependency by design.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from context_selection.policy import MLPBernoulliPolicy

try:
    import torch
    from context_selection.policy_torch import (
        MLPBernoulliPolicyTorch,
        load_numpy_weights,
        to_float64,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class TestPolicyTorchGradientCrossCheck(unittest.TestCase):
    """The PyTorch analogue of test_policy.py's gradient check: instead of
    comparing the NumPy analytic gradient to a finite-difference estimate,
    this compares it directly to torch.autograd's gradient on the same
    weights/input, for both actions and several random states."""

    def setUp(self):
        self.obs_dim = 6
        self.hidden_dim = 5
        self.np_policy = MLPBernoulliPolicy(obs_dim=self.obs_dim, hidden_dim=self.hidden_dim, seed=0)

        self.torch_policy = to_float64(
            MLPBernoulliPolicyTorch(obs_dim=self.obs_dim, hidden_dim=self.hidden_dim, seed=0)
        )
        load_numpy_weights(self.torch_policy, self.np_policy.params)

        self.rng = np.random.default_rng(42)

    def _torch_grad(self, s_np: np.ndarray, action: int):
        s = torch.as_tensor(s_np, dtype=torch.float64)
        logit = self.torch_policy(s)
        dist = torch.distributions.Bernoulli(logits=logit)
        logp = dist.log_prob(torch.as_tensor(float(action), dtype=torch.float64))

        self.torch_policy.zero_grad()
        logp.backward()

        return {
            "W1": self.torch_policy.fc1.weight.grad.numpy().copy(),
            "b1": self.torch_policy.fc1.bias.grad.numpy().copy(),
            "W2": self.torch_policy.fc2.weight.grad.numpy().reshape(-1).copy(),
            "b2": self.torch_policy.fc2.bias.grad.numpy().copy(),
        }, float(logp.item())

    def test_forward_probabilities_match(self):
        """Sanity precondition: if the forward passes don't agree, a
        gradient match downstream would be meaningless."""
        for _ in range(10):
            s = self.rng.normal(size=self.obs_dim)
            prob_np, _ = self.np_policy.forward(s)

            s_t = torch.as_tensor(s, dtype=torch.float64)
            prob_torch = torch.sigmoid(self.torch_policy(s_t)).item()

            self.assertAlmostEqual(prob_np, prob_torch, places=10)

    def test_gradient_matches_torch_autograd(self):
        """The core cross-check: hand-derived NumPy grad vs. torch.autograd
        grad, same weights, same state, same action -- should match near
        exactly (float64 precision), not just approximately."""
        for trial in range(10):
            s = self.rng.normal(size=self.obs_dim)
            for action in (0, 1):
                _, cache = self.np_policy.forward(s)
                np_grad = self.np_policy.logprob_grad(cache, action, coeff=1.0)
                torch_grad, _ = self._torch_grad(s, action)

                for key in ("W1", "b1", "W2", "b2"):
                    np.testing.assert_allclose(
                        np_grad[key], torch_grad[key], atol=1e-8, rtol=1e-6,
                        err_msg=(
                            f"trial={trial} action={action} param='{key}': "
                            f"NumPy hand-derived grad and torch.autograd grad disagree"
                        ),
                    )

    def test_logprob_values_match(self):
        for _ in range(10):
            s = self.rng.normal(size=self.obs_dim)
            for action in (0, 1):
                _, cache = self.np_policy.forward(s)
                np_logp = self.np_policy.logprob(cache, action)
                _, torch_logp = self._torch_grad(s, action)
                self.assertAlmostEqual(np_logp, torch_logp, places=10)


if __name__ == "__main__":
    unittest.main()
