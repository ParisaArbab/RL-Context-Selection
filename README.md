# RL for Context Selection in Agent Workflows

A small, self-contained research project built for the **RL AI Research** application, targeting the role's first bullet directly:

> *"Investigate and prototype novel RL approaches for context selection,
> reward shaping, or policy optimization in agent workflows."*

**The question this project asks:** when an agent has a ranked list of
retrieved context (documents, tool outputs, memory snippets) and a limited
context-window budget, can a learned policy decide what to keep -- balancing
task success against token cost and redundancy -- better than the heuristics
teams reach for by default (take everything above a similarity threshold,
random sampling as a floor)?

Everything in this repo actually runs, no GPU required, in well under ten
minutes total. See `report.md` for the full write-up: motivation, method,
results, and an honest discussion of what worked, what didn't (including a
training-stability issue tied to reward shaping that I flag rather than
smooth over), and why.

## What's here

```
src/context_selection/
  environment.py    MDP for sequential context selection (relevant /
                    redundant / distractor items, token-cost budget,
                    saturating coverage reward)
  policy.py         2-layer MLP Bernoulli policy (PyTorch), trained via
                    torch.autograd -- nn.Linear -> tanh -> nn.Linear,
                    torch.distributions.Bernoulli for log-prob/entropy
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

tests/              14 unit tests, including a torch.autograd.gradcheck
                    correctness check of the policy's gradients and a
                    direct check of the reward-shaping invariance identity

report.md           Full write-up: method, results, analysis, and
                    concrete next experiments
```

## Implementation notes

The policy, REINFORCE trainer, and PPO-lite trainer are all built on
PyTorch (`torch.autograd`, `torch.optim.Adam`, `torch.distributions`).
Rather than trust autograd blindly, `tests/test_policy.py` runs
`torch.autograd.gradcheck` -- PyTorch's built-in numerical Jacobian
check -- against the policy's log-probability function, across every
parameter tensor and both actions, so the gradients driving every
training curve in this repo are independently verified, not assumed
correct. Every number and plot in `experiments/results/` is from a real
run of this code, not a mockup.

## Running it

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments/run_benchmark.py    # ~3 min, writes experiments/results/
```

## Headline result (details + discussion in `report.md`)

On held-out episodes (mean over 3 seeds x 300 episodes), the learned
policies land close to a hand-tuned similarity-threshold heuristic, and a
PPO-lite variant reaches close to that level of performance in roughly an
order of magnitude fewer training episodes than plain REINFORCE. A
ground-truth oracle shows there is still substantial headroom (same task
success at ~3x lower context-token cost), and I did **not** find that naive
coverage-only reward shaping helped REINFORCE in this setup -- both methods
that use shaping (REINFORCE+shaping and PPO-lite) show measurably lower
reward and higher seed-to-seed variance than unshaped REINFORCE. I report
that result honestly, with an analysis of the likely cause and a
concretely testable fix, rather than tuning it away.

![benchmark](experiments/results/benchmark_comparison.png)

*(Repo initialized and maintained by Parisa Arbab.)*

For questions about this project, feel free to open an issue.
