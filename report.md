# RL for Context Selection in Agent Workflows
### Project report -- Pokee RL AI Research Intern application

## 1. Motivation

Agent workflows built on retrieval or tool use routinely face a decision the
underlying LLM never directly optimizes for: **given a ranked list of
candidate context (retrieved chunks, tool outputs, memory), which items
actually go into the context window?** Every extra item costs tokens,
latency, and $ -- and past a point, adds *noise* rather than signal
(irrelevant or redundant context can measurably hurt downstream task
performance, not just waste budget). Teams typically handle this with a
fixed top-k or a similarity threshold. That's a reasonable default, but it's
not obviously optimal, and it's exactly the kind of sequential,
budget-constrained decision problem RL is suited to: the policy should
weigh "does this item add coverage I don't already have?" against
"how much of my remaining budget does it cost?", which is a *contextual*,
state-dependent decision that a fixed threshold structurally cannot make.

This project scopes that problem down to something small enough to build,
run, and rigorously evaluate end-to-end: a synthetic but non-trivial MDP
with a checkable optimal policy, three RL training methods, three reference
baselines, and a PyTorch implementation validated by unit tests (including
a `torch.autograd.gradcheck` gradient check) rather than just "it looked
like it worked."

## 2. Problem formulation

**Environment** (`src/context_selection/environment.py`): each episode
samples a query embedding and 14 candidate context items -- a mix of
*relevant* items (noisy copies of the query embedding), *redundant* items
(near-duplicates of a relevant item), and *distractors* (pure noise) -- each
with a random token cost. The agent scans the (fixed-order) candidate list
once, deciding INCLUDE/SKIP for each item under a hard cost budget.

**Reward** is terminal: a saturating coverage function of the selected
set's similarity to the query (so a second relevant item still helps, but
with rapidly diminishing returns -- "the answer is confirmed, more evidence
barely helps"), minus a cost penalty and a redundancy penalty. This is a
genuinely hard credit-assignment problem: the *useful* signal only arrives
at the end of the episode, and both over-including (wastes budget on
redundant/irrelevant items) and under-including (misses coverage) are
punished.

**Why synthetic data rather than a "real" benchmark?** I did not want to
either (a) fabricate a realistic-looking dataset and imply it's real agent
traces, or (b) claim results on a public benchmark I hadn't actually run
against. The environment is designed so difficulty, redundancy, and noise
are config knobs (`EnvConfig`), an **oracle** policy with access to
ground-truth labels gives a checkable upper bound, and every claim below is
from an actual run of the code in this repo.

## 3. Methods compared

| Method | What it is |
|---|---|
| Random (p=0.5) | Floor baseline |
| Greedy-relevance | Include iff cosine-similarity(item, query) > threshold and budget allows -- the realistic "what a team ships by default" baseline |
| **REINFORCE** (no shaping) | Policy gradient with a moving-average baseline + entropy bonus, on a 2-layer MLP Bernoulli policy |
| **REINFORCE** (+ shaping) | Same, with potential-based reward shaping using Phi(s) = current coverage estimate |
| **PPO-lite** (+ shaping) | Clipped-surrogate objective (Schulman et al., 2017) with 4 epochs of reuse per batch of 12 on-policy episodes, same policy class and shaping |
| Oracle | Uses ground-truth item labels (never available to a real policy) -- upper-bound reference, not a fair comparison |

All three RL variants use the identical policy architecture, optimizer, and
evaluation protocol, so differences are attributable to the *training
method*, not incidental implementation differences.

**Implementation:** the policy (`src/context_selection/policy.py`) is a
2-layer MLP Bernoulli policy in PyTorch
(`nn.Linear -> tanh -> nn.Linear`, with `torch.distributions.Bernoulli`
supplying the log-prob and entropy), trained with `torch.autograd` and
`torch.optim.Adam` throughout -- both the REINFORCE trainer
(`train.py`) and the PPO-lite clipped-surrogate trainer (`ppo.py`).
`tests/test_policy.py` runs `torch.autograd.gradcheck` against the
policy's log-probability function -- PyTorch's own finite-difference
Jacobian checker -- across every parameter tensor and both actions, so the
gradients driving every training curve below are verified numerically, not
assumed correct because "autograd is autograd."

## 4. Reward shaping: the theory, and what I actually found

Potential-based shaping (Ng, Harada & Russell, 1999) adds
`F(s, s') = gamma * Phi(s') - Phi(s)` to the reward, where Phi is any
function of state. Its key property is **policy invariance**: for any Phi
and any gamma, the *optimal* policy under the shaped reward is identical to
the optimal policy under the original reward -- shaping can only change how
fast you learn it, not what you converge to (in the limit, with an
unbiased, well-tuned optimizer). I used current coverage of the query as
Phi, which gives the agent a dense signal after *every* include/skip
decision instead of only at episode end.

I did not just cite the theorem -- `tests/test_reward.py` checks the
underlying telescoping identity directly (the discounted sum of shaped
rewards equals the discounted sum of raw rewards plus a boundary term)
before trusting it in the training loop.

**Empirically, in this environment, shaping did not help REINFORCE, and
measurably hurt it:**

| Method | Reward | Success | Cost frac. |
|---|---:|---:|---:|
| REINFORCE, no shaping | **0.624** | 0.974 | 0.943 |
| REINFORCE, + shaping | 0.441 | 0.697 | 0.688 |
| PPO-lite, + shaping | 0.553 | 0.862 | 0.833 |

(mean over 3 seeds x 300 held-out episodes; see `experiments/results/results.csv`)

Looking at the learning curves (`experiments/results/learning_curves.png`),
shaped REINFORCE is also visibly *less stable* across training -- seed std
on the final shaped-REINFORCE result is 0.176 versus 0.002 for unshaped
REINFORCE, and individual seeds show sharp dips mid-training that the
unshaped runs don't.

**My best explanation, stated as a hypothesis rather than a conclusion:**
a coverage-only potential rewards the agent for *any* movement toward
higher coverage, on every single step, regardless of cost. That's a much
denser and more immediately rewarding signal than the sparse terminal
reward, which also has to account for cost and redundancy. A per-step
optimizer can end up chasing the dense shaping signal (include items that
raise coverage right now) in a way that's harder to walk back later, even
though the terminal objective still eventually dominates in principle.
Policy invariance is a statement about the *optimal* policy an ideal
learner converges to -- it says nothing about the optimization path a
noisy, finite-sample policy-gradient method actually takes, and I think
that's what's biting us here.

**A concrete next experiment** (started but not conclusive in the time
available, so reported here rather than in the headline results): a
cost-aware potential, `Phi(s) = coverage(s) - lambda * cost_spent_frac(s)`,
which should shrink the "grab coverage now regardless of cost" incentive
while keeping the same policy-invariance guarantee (Phi is still a pure
function of state). Early runs were inconclusive; I'd want to sweep
`lambda` and gamma properly rather than report a single noisy run.

## 5. REINFORCE vs. PPO-lite: sample efficiency and stability

PPO-lite reaches close to REINFORCE's final performance level in roughly
an order of magnitude fewer episodes (~500-1000 vs. ~2500-3000; see
`experiments/results/learning_curves.png`, where the green PPO-lite curve
plateaus much earlier than the red unshaped-REINFORCE curve). This is
consistent with what PPO is designed to do: reusing each batch of
on-policy rollouts for several gradient epochs (instead of one), under a
clipped surrogate objective that caps how far a single update can move the
policy, extracts more learning signal per environment interaction while
limiting destructive updates from any one noisy episode. Both trainers
share the exact same policy architecture, optimizer, and (crude,
single-scalar) baseline, so this comparison isolates the effect of the
clipped multi-epoch update itself, not a confound from a better value
function or bigger network.

I want to be precise about scope here: this is a **lightweight** PPO
(no separate value network, no GAE -- see the docstring in `ppo.py` for the
full list of simplifications), so "PPO-lite converges faster" should be
read as "the clipped multi-epoch mechanism itself helps here," not as a
claim that this matches a production PPO implementation's absolute sample
efficiency.

I also want to flag, rather than smooth over, that PPO-lite's final result
here (0.553 ± 0.098) has both a lower mean and a noticeably higher
seed-to-seed variance than I'd like -- one seed's learning curve dips
sharply late in training before partially recovering. Both PPO-lite and
shaped REINFORCE (the two methods using potential-based shaping) show this
instability; unshaped REINFORCE does not. That pattern points at reward
shaping, not the PPO mechanism itself, as the likely source -- see Sec. 4's
discussion and the cost-aware potential proposed there as the next thing
to try.

## 6. How much headroom is left?

The oracle (uses ground-truth relevance labels, which no real policy has)
reaches the same task success as Greedy-relevance at roughly **3x lower
context-token cost** (0.356 vs. 0.927 mean cost fraction). None of the
learned or heuristic policies close more than a small fraction of that
gap. That's the most useful single number in this report: it says the
ceiling on "select less, more precisely" is large, and that the current
policies are mostly learning *whether* an item looks relevant (which
Greedy-relevance already does reasonably well from raw embeddings) rather
than the harder, more valuable skill of *stopping once coverage is already
sufficient*, which is where the real cost savings live.

## 7. What I'd do next with more time

1. **Cost-aware potential shaping** (Sec. 4) -- finish the lambda/gamma
   sweep instead of reporting one run, and see whether it also resolves
   the variance issue in Sec. 5.
2. **A learned value baseline** instead of a single moving-average scalar --
   the current baseline can't tell an easy episode (few distractors, low
   costs) from a hard one, which inflates gradient variance exactly on the
   episodes where the "stop early" skill matters most.
3. **A stopping-focused auxiliary signal** -- since the oracle's advantage
   is almost entirely about *not over-including* rather than *finding* the
   right items, a policy-gradient variant with an explicit bonus for
   correctly predicting "no further item will raise coverage" seems like a
   more targeted fix than more training of the current objective.
4. **Scale up**: bigger candidate lists, richer (non-linear) relevance
   structure, a real embedding model in place of synthetic vectors, batched
   rollouts, and a GPU device -- the current setup is deliberately small
   and CPU-only, one episode at a time, so every result here could be run
   and checked end to end in well under ten minutes; the natural next step
   is running the same trainers against real retrieval traces on GPU
   infra, which mainly means batching `run_episode_collect`/`evaluate`
   over multiple environments in parallel.

## 8. Reproducing everything in this report

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v      # 14 tests, ~12s (gradcheck is the slow one)
python experiments/run_benchmark.py           # ~3 min total
```
`experiments/results/` will contain `results.csv` (the exact numbers
quoted above), `learning_curves.png`, and `benchmark_comparison.png`.
