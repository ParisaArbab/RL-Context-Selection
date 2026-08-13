import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from context_selection.policy import MLPBernoulliPolicy, AdamOptimizer


class TestPolicyGradients(unittest.TestCase):
    """The most important test in this repo: verifies the hand-derived
    backprop in `policy.logprob_grad` against numerical finite-difference
    gradients. If this test passes, the REINFORCE and PPO updates built
    on top of `logprob_grad` are gradient-correct."""

    def setUp(self):
        self.policy = MLPBernoulliPolicy(obs_dim=6, hidden_dim=5, seed=0)
        self.rng = np.random.default_rng(0)
        self.s = self.rng.normal(size=6)

    def _logprob_at(self, params, action):
        prob, cache = self.policy.forward(self.s, params=params)
        return self.policy.logprob(cache, action)

    def test_gradient_matches_finite_difference(self):
        eps = 1e-5
        for action in (0, 1):
            _, cache = self.policy.forward(self.s)
            analytic = self.policy.logprob_grad(cache, action, coeff=1.0)

            for key in self.policy.params:
                param = self.policy.params[key]
                numeric = np.zeros_like(param)
                it = np.nditer(param, flags=["multi_index"])
                for _ in it:
                    idx = it.multi_index
                    params_plus = self.policy.clone_params()
                    params_minus = self.policy.clone_params()
                    params_plus[key][idx] += eps
                    params_minus[key][idx] -= eps
                    f_plus = self._logprob_at(params_plus, action)
                    f_minus = self._logprob_at(params_minus, action)
                    numeric[idx] = (f_plus - f_minus) / (2 * eps)

                np.testing.assert_allclose(
                    analytic[key], numeric, atol=1e-4, rtol=1e-3,
                    err_msg=f"gradient mismatch for param '{key}', action={action}"
                )

    def test_probability_is_in_unit_interval(self):
        for _ in range(20):
            s = self.rng.normal(size=6)
            prob, _ = self.policy.forward(s)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_adam_reduces_negative_log_likelihood(self):
        """Sanity check: repeatedly ascending log pi(a=1|s) via Adam should
        push p(a=1|s) toward 1 for a fixed state."""
        policy = MLPBernoulliPolicy(obs_dim=6, hidden_dim=5, seed=1)
        opt = AdamOptimizer(policy.params, lr=0.05)
        s = self.rng.normal(size=6)
        prob0, _ = policy.forward(s)

        for _ in range(200):
            _, cache = policy.forward(s)
            grads = policy.logprob_grad(cache, action=1, coeff=1.0)
            neg_grads = {k: -v for k, v in grads.items()}
            opt.step(policy.params, neg_grads)

        prob_final, _ = policy.forward(s)
        self.assertGreater(prob_final, prob0)
        self.assertGreater(prob_final, 0.9)


if __name__ == "__main__":
    unittest.main()
