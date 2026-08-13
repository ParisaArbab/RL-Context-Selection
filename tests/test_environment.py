import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from context_selection.environment import ContextSelectionEnv, EnvConfig


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig(seed=42)
        self.env = ContextSelectionEnv(self.cfg)

    def test_reset_shapes(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape[0], self.env.obs_dim)

    def test_episode_terminates(self):
        self.env.reset()
        done = False
        steps = 0
        while not done:
            step = self.env.step(0)  # always skip
            done = step.done
            steps += 1
            self.assertLess(steps, 10_000, "environment failed to terminate")
        self.assertEqual(steps, self.env.cfg.num_candidates)

    def test_skip_all_yields_zero_coverage_and_low_reward(self):
        self.env.reset()
        done = False
        info = {}
        while not done:
            step = self.env.step(0)
            done, info = step.done, step.info
        self.assertAlmostEqual(info["coverage"], 0.0, places=6)
        self.assertAlmostEqual(info["cost_frac"], 0.0, places=6)

    def test_budget_is_respected(self):
        self.env.reset()
        done = False
        while not done:
            step = self.env.step(1)  # always try to include
            done = step.done
        self.assertLessEqual(self.env._spent, self.env.cfg.cost_budget + 1e-9)

    def test_oracle_selection_within_budget(self):
        self.env.reset()
        sel = self.env.oracle_selection()
        total_cost = sum(self.env._costs[i] for i in sel)
        self.assertLessEqual(total_cost, self.env.cfg.cost_budget + 1e-9)
        # oracle should only ever pick items labeled 'relevant'
        self.assertTrue(all(self.env._labels[i] == "relevant" for i in sel))

    def test_coverage_monotonic_in_relevant_items(self):
        """Adding a relevant item should never decrease coverage."""
        self.env.reset()
        relevant_idx = [i for i in range(self.env._n) if self.env._labels[i] == "relevant"]
        if len(relevant_idx) >= 2:
            cov1 = self.env._coverage(relevant_idx[:1])
            cov2 = self.env._coverage(relevant_idx[:2])
            self.assertGreaterEqual(cov2, cov1 - 1e-9)

    def test_reproducible_with_seed(self):
        env_a = ContextSelectionEnv(EnvConfig(seed=7))
        env_b = ContextSelectionEnv(EnvConfig(seed=7))
        obs_a = env_a.reset()
        obs_b = env_b.reset()
        np.testing.assert_allclose(obs_a, obs_b)


if __name__ == "__main__":
    unittest.main()
