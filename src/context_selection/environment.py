"""
environment.py
===============

A synthetic MDP that models the problem an LLM agent faces when it must
decide, item-by-item, which pieces of retrieved context to keep in its
working context window.

Why synthetic?
--------------
Real agent traces (retrieval logs + downstream task success) are exactly
the kind of proprietary, high-value data a company like Pokee would have
and I would not want to fabricate results against a fake "real" dataset.
Instead, this environment is built so that:

  1. It has a *known, checkable* optimal policy (so we can sanity-check
     learning against an oracle rather than just "trust the reward went up").
  2. It reproduces the qualitative structure of real context-selection
     problems: relevant items, near-duplicate/redundant items, irrelevant
     distractors, and a hard cost budget.
  3. Difficulty, redundancy, and noise are all config knobs, so the same
     code can be used to run scaling / ablation studies.

Episode structure
------------------
At the start of an episode we sample:
  * a query/target embedding  t âˆˆ R^d
  * N candidate context items, each with:
      - an embedding x_i âˆˆ R^d
      - a token cost c_i (proxy for context-window budget consumed)
    where some items are "relevant" (x_i close to t), some are
    "redundant" (near-duplicates of an already-relevant item), and some
    are pure distractors (random noise).

The agent is shown candidates **one at a time**, in a fixed (e.g.
retrieval-ranked) order, and for each one chooses INCLUDE (1) or SKIP (0).
This mirrors an agent scanning a ranked retrieval list under a token
budget. The episode ends when all candidates have been scanned or the
cost budget is exhausted.

Reward
------
Terminal reward = task_success(selected_set, target) - cost_penalty.
task_success uses a saturating coverage function so that (a) including
more *relevant, non-redundant* items helps up to a point, and (b) once
the query is "answered", extra items only add cost without benefit --
exactly the trade-off a real context-selection policy has to learn.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EnvConfig:
    embed_dim: int = 8
    num_candidates: int = 14
    cost_budget: float = 5.0
    frac_relevant: float = 0.3
    frac_redundant: float = 0.25
    # frac_distractor is implied = 1 - relevant - redundant
    relevance_noise: float = 0.4        # noise added to relevant items
    redundancy_noise: float = 0.08      # near-duplicate items are very close
    cost_low: float = 0.5
    cost_high: float = 2.0
    coverage_saturation: float = 0.82   # coverage level beyond which extra items don't help
    cost_penalty_weight: float = 0.35
    redundancy_penalty_weight: float = 0.25
    success_scale: float = 1.0
    seed: Optional[int] = None


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class ContextSelectionEnv:
    """A single-episode-at-a-time context-selection MDP.

    Observation for the item currently being considered is the
    concatenation of:
      [ item_embedding (d),
        item_cost (1),
        query_embedding (d),
        running_selected_embedding_mean (d),
        remaining_budget_frac (1),
        frac_items_remaining (1) ]
    """

    def __init__(self, config: EnvConfig = EnvConfig()):
        self.cfg = config
        self._rng = np.random.default_rng(config.seed)
        self.reset()

    @property
    def obs_dim(self) -> int:
        d = self.cfg.embed_dim
        return d + 1 + d + d + 1 + 1

    # ------------------------------------------------------------------
    # Episode construction
    # ------------------------------------------------------------------
    def _sample_episode(self):
        cfg = self.cfg
        d = cfg.embed_dim
        rng = self._rng

        target = rng.normal(size=d)
        target /= np.linalg.norm(target) + 1e-8

        n_relevant = max(1, int(round(cfg.frac_relevant * cfg.num_candidates)))
        n_redundant = max(0, int(round(cfg.frac_redundant * cfg.num_candidates)))
        n_distractor = max(0, cfg.num_candidates - n_relevant - n_redundant)

        items = []
        labels = []  # 'relevant' | 'redundant' | 'distractor', for analysis/oracle only

        for _ in range(n_relevant):
            noise = rng.normal(scale=cfg.relevance_noise, size=d)
            vec = target + noise
            items.append(vec)
            labels.append("relevant")

        # redundant items duplicate a *random already-sampled relevant item*
        relevant_so_far = [items[i] for i in range(n_relevant)] if n_relevant > 0 else [target]
        for _ in range(n_redundant):
            base = relevant_so_far[rng.integers(0, len(relevant_so_far))]
            vec = base + rng.normal(scale=cfg.redundancy_noise, size=d)
            items.append(vec)
            labels.append("redundant")

        for _ in range(n_distractor):
            vec = rng.normal(size=d)
            items.append(vec)
            labels.append("distractor")

        order = rng.permutation(len(items))
        items = [items[i] for i in order]
        labels = [labels[i] for i in order]
        costs = rng.uniform(cfg.cost_low, cfg.cost_high, size=len(items))

        self._target = target
        self._items = np.stack(items)
        self._labels = labels
        self._costs = costs
        self._n = len(items)

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._sample_episode()
        self._t = 0
        self._selected_idx: List[int] = []
        self._spent = 0.0
        return self._observe()

    def _observe(self) -> np.ndarray:
        cfg = self.cfg
        if self._t >= self._n:
            item = np.zeros(cfg.embed_dim)
            cost = 0.0
        else:
            item = self._items[self._t]
            cost = self._costs[self._t]

        if self._selected_idx:
            selected_mean = self._items[self._selected_idx].mean(axis=0)
        else:
            selected_mean = np.zeros(cfg.embed_dim)

        remaining_budget_frac = max(0.0, (cfg.cost_budget - self._spent) / cfg.cost_budget)
        frac_remaining = (self._n - self._t) / max(1, self._n)

        return np.concatenate([
            item,
            [cost],
            self._target,
            selected_mean,
            [remaining_budget_frac],
            [frac_remaining],
        ]).astype(np.float64)

    def _coverage(self, selected_idx: List[int]) -> float:
        """Saturating coverage of the target by the selected set.

        Uses max cosine similarity to any *distinct-direction* selected
        item, softly saturating so redundant items add little once the
        target is already well covered -- this is what makes redundancy
        a genuine (not just cost-based) inefficiency for the agent to learn.
        """
        if not selected_idx:
            return 0.0
        sims = self._items[selected_idx] @ self._target / (
            np.linalg.norm(self._items[selected_idx], axis=1) * np.linalg.norm(self._target) + 1e-8
        )
        sims = np.clip(sims, -1.0, 1.0)
        # diminishing-returns aggregation instead of a hard max: rewards
        # covering the target from more than one item a little, but with
        # strongly decreasing marginal value (models "the answer is
        # confirmed, more evidence barely helps").
        sorted_sims = np.sort(np.clip(sims, 0, None))[::-1]
        cov = 0.0
        remaining = 1.0
        for s in sorted_sims:
            gain = remaining * s
            cov += gain
            remaining *= (1 - s * 0.6)
            if remaining <= 1e-4:
                break
        return float(np.clip(cov, 0.0, 1.0))

    def potential(self, selected_idx: List[int]) -> float:
        """Potential function Phi(s) used for reward shaping (see reward.py).

        Phi is simply the current coverage estimate -- a cheap, always-
        available proxy for "how close is the agent to a satisfying
        context set". Using coverage (rather than the *true* final
        reward, which is unobserved mid-episode) as the potential is
        exactly what makes potential-based shaping useful in practice.
        """
        return self._coverage(selected_idx)

    def step(self, action: int) -> StepResult:
        assert action in (0, 1)
        cfg = self.cfg
        info = {}

        if self._t < self._n and action == 1:
            cost = self._costs[self._t]
            if self._spent + cost <= cfg.cost_budget:
                self._selected_idx.append(self._t)
                self._spent += cost
            else:
                # over-budget include is treated as a no-op skip; the
                # agent must learn to respect the budget itself.
                info["budget_rejected"] = True

        self._t += 1
        done = self._t >= self._n

        reward = 0.0
        if done:
            coverage = self._coverage(self._selected_idx)
            # success ramps linearly with coverage up to the saturation
            # point, then flatlines at 1.0 -- extra coverage beyond that
            # point is real (redundant evidence) but doesn't raise success.
            success = np.clip(coverage / cfg.coverage_saturation, 0.0, 1.0)

            cost_frac = self._spent / cfg.cost_budget
            cost_penalty = cfg.cost_penalty_weight * cost_frac

            n_redundant_selected = sum(
                1 for i in self._selected_idx if self._labels[i] == "redundant"
            )
            redundancy_penalty = cfg.redundancy_penalty_weight * (
                n_redundant_selected / max(1, self._n)
            )

            reward = cfg.success_scale * success - cost_penalty - redundancy_penalty
            info.update(dict(
                coverage=coverage,
                success=success,
                cost_frac=cost_frac,
                n_selected=len(self._selected_idx),
                n_redundant_selected=n_redundant_selected,
                labels_selected=[self._labels[i] for i in self._selected_idx],
            ))

        return StepResult(self._observe(), float(reward), done, info)

    # ------------------------------------------------------------------
    # Oracle helpers (analysis / sanity checks only -- not used by learners)
    # ------------------------------------------------------------------
    def oracle_selection(self) -> List[int]:
        """Greedy-optimal-ish selection using ground-truth labels: take
        relevant items first (cheapest first) until budget or coverage
        saturates, skip redundant/distractor items. Used as an upper
        bound reference point, not as a training signal."""
        idx_by_type = [i for i in range(self._n) if self._labels[i] == "relevant"]
        idx_by_type.sort(key=lambda i: self._costs[i])
        selected, spent = [], 0.0
        for i in idx_by_type:
            if spent + self._costs[i] > self.cfg.cost_budget:
                continue
            selected.append(i)
            spent += self._costs[i]
            if self._coverage(selected) >= self.cfg.coverage_saturation:
                break
        return selected
