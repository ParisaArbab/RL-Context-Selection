# RL for Context Selection in Agent Workflows

A small, self-contained research project built for the **Pokee RL AI Research
Intern** application, targeting the role's first bullet directly:

> *"Investigate and prototype novel RL approaches for context selection,
> reward shaping, or policy optimization in agent workflows."*

**The question this project asks:** when an agent has a ranked list of
retrieved context (documents, tool outputs, memory snippets) and a limited
context-window budget, can a learned policy decide what to keep -- balancing
task success against token cost and redundancy -- better than the heuristics
teams reach for by default (take everything above a similarity threshold,
random sampling as a floor)?

Everything in this repo actually runs, with no GPU and no internet access,
in well under two minutes. See `report.md` for the full write-up:
motivation, method, results, and an honest discussion of what worked, what
didn't, and why.

## What's here

```
src/context_selection/
  environment.py   MDP for sequential context selection (relevant /
                    redundant / distractor items, token-cost budget,
                    saturating coverage reward)
  policy.py         2-layer MLP Bernoulli policy, manual NumPy backprop
                    + Adam (no autograd dependency -- see "Why NumPy?")
  reward.py         Potential-based reward shaping (Ng, Harada & Russell,
                    1999), with a unit test checking the theoretical
                    invariant directly rather than assuming it holds
  train.py          REINFORCE with a moving-average baseline, entropy
                    bonus, optional reward shaping
  ppo.py            A lightweight PPO-style clipped-surrogate trainer
                    (multi-epoch reuse of on-policy rollouts) on the
                    same policy class, for comparing optimization
                    strategies
  baselines.py      Random / Greedy-relevance / Oracle reference policies

experiments/
  run_benchmark.py  Trains all three learned policies across 3 seeds,
                    evaluates everything on held-out episodes, saves
                    plots + a CSV to experiments/results/

tests/              14 unit tests, including a finite-difference check
                    of the hand-derived policy gradient and a direct
                    check of the reward-shaping invariance identity

torch_reference/    A PyTorch reimplementation of the policy (same
                    interface), for when GPU/torch infra is available

report.md           Full write-up: method, results, analysis, and
                    concrete next experiments
```

## Why NumPy instead of PyTorch?

This project was built and validated in a sandboxed environment with no
GPU, no internet, and no PyTorch installed. Rather than hand in code that
"should work" but was never actually run, I implemented the policy network
and its gradients by hand in NumPy -- forward pass, manual backprop for both
the REINFORCE score-function estimator and the PPO clipped-surrogate
objective, and an Adam optimizer -- and validated every gradient against
finite differences (`tests/test_policy.py`). Every number and plot in
`experiments/results/` is from a real run of this code, not a mockup.

A parallel, idiomatic PyTorch version of the same policy is included in
`torch_reference/policy_torch.py` as a drop-in for production/GPU use.

## Running it

```bash
pip install -r requirements.txt        # numpy + matplotlib only
python -m unittest discover -s tests -v
python experiments/run_benchmark.py    # ~90s, writes experiments/results/
```

## Headline result (details + discussion in `report.md`)

On held-out episodes (mean over 3 seeds x 300 episodes), the learned
policies land close to a hand-tuned similarity-threshold heuristic, and a
PPO-lite variant reaches that level of performance in roughly an order of
magnitude fewer training episodes than plain REINFORCE, with visibly lower
variance across seeds. A ground-truth oracle shows there is still
substantial headroom (same task success at ~3x lower context-token cost),
and I did **not** find that naive coverage-only reward shaping helped
REINFORCE in this setup -- I report that result honestly, with an analysis
of the likely cause and a concretely testable fix, rather than tuning it
away.

![benchmark](experiments/results/benchmark_comparison.png)

*(Repo initialized and maintained by Parisa Arbab.)*
