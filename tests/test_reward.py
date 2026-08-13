import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from context_selection.reward import shaped_rewards, discounted_returns


class TestRewardShaping(unittest.TestCase):
    def test_shaped_rewards_length(self):
        raw = [0.0, 0.0, 1.0]
        potentials = [0.0, 0.3, 0.6, 1.0]  # len = len(raw) + 1
        shaped = shaped_rewards(raw, potentials, gamma=0.9)
        self.assertEqual(len(shaped), len(raw))

    def test_shaping_telescopes_to_same_discounted_return_when_gamma_matches(self):
        """Core theoretical property (Ng et al., 1999): the discounted
        SUM of shaped rewards over a finite episode equals the discounted
        sum of raw rewards plus a boundary term gamma^T*Phi(s_T) - Phi(s_0).
        We check this identity directly rather than trusting it blindly.
        """
        gamma = 0.9
        raw = [0.1, -0.05, 0.0, 2.0]
        potentials = [0.2, 0.5, 0.4, 0.9, 0.0]  # Phi(s_0..s_4), terminal potential = 0
        shaped = shaped_rewards(raw, potentials, gamma=gamma)

        T = len(raw)
        lhs = sum(gamma ** t * shaped[t] for t in range(T))
        rhs = sum(gamma ** t * raw[t] for t in range(T)) + (gamma ** T) * potentials[-1] - potentials[0]
        self.assertAlmostEqual(lhs, rhs, places=8)

    def test_discounted_returns_basic(self):
        rewards = [1.0, 1.0, 1.0]
        returns = discounted_returns(rewards, gamma=0.5)
        # R_2 = 1
        # R_1 = 1 + 0.5*1 = 1.5
        # R_0 = 1 + 0.5*1.5 = 1.75
        np.testing.assert_allclose(returns, [1.75, 1.5, 1.0])

    def test_discounted_returns_zero_gamma_equals_immediate_reward(self):
        rewards = [3.0, -1.0, 2.0]
        returns = discounted_returns(rewards, gamma=0.0)
        np.testing.assert_allclose(returns, rewards)


if __name__ == "__main__":
    unittest.main()
