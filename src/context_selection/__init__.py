"""
context_selection
==================

A small, self-contained RL research codebase for studying **context
selection** in AI agent workflows: given a query and a ranked list of
candidate context items (retrieved documents, tool outputs, memory
snippets, etc.), learn a policy that decides which items to include
in the agent's context window so as to maximize task success while
respecting a token-cost budget.

This package implements:
  * `environment.py`   -- a synthetic but non-trivial MDP for context
                           selection, with configurable relevance,
                           redundancy, distractor, and cost structure.
  * `policy.py`         -- a small MLP Bernoulli policy (PyTorch),
                           trained with REINFORCE + baseline + entropy
                           bonus via torch.autograd.
  * `reward.py`         -- reward functions, including a potential-based
                           reward-shaping term (Ng, Harada & Russell,
                           1999) that provably preserves the optimal
                           policy while densifying the learning signal.
  * `ppo.py`             -- a lightweight PPO-style clipped-surrogate
                           update on top of the same policy class, for
                           comparing on-policy variance-reduction and
                           policy-optimization strategies.
  * `baselines.py`      -- random, greedy-relevance, and oracle
                           (upper-bound) selection policies.
"""
