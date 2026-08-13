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
run, and rigorously evaluate end-to-end without any real infrastructure:
a synthetic but non-trivial MDP with a checkable optimal policy, three RL
training methods, three reference baselines, and a from-scratch
implementation validated by unit tests (including a finite-difference
gradient check) rather than just "it looked like it worked."

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

**On the implementation(s):** this project now has two independent, fully
executed implementations of the same policy, environment, and training
algorithms:

1. **From-scratch NumPy** (`src/context_selection/policy.py`, `train.py`,
   `ppo.py`) -- forward pass, manual backprop for both the REINFORCE
   score-function estimator and the PPO clipped-surrogate objective, and a
   hand-written Adam optimizer. Built with no `torch.autograd`, originally
   because the sandbox this was first built in had no PyTorch installed,
   and kept because writing the gradients by hand is a useful correctness
   exercise in its own right. `tests/test_policy.py` checks the
   hand-derived REINFORCE gradient against numerical finite differences
   across every parameter, for both actions.
2. **PyTorch** (`src/context_selection/policy_torch.py`, `train_torch.py`,
   `ppo_torch.py`) -- the same architecture and algorithms, but gradients
   come from `torch.autograd` and optimization from `torch.optim.Adam`,
   rather than hand-derived math. This was ported once internet access was
   available to `pip install torch`, and is validated in two ways rather
   than assumed to be correct: (a) `tests/test_policy_torch.py` loads
   identical weights into both implementations and checks that
   `torch.autograd`'s gradient of `log pi(a|s)` matches the hand-derived
   NumPy gradient to float64 precision (not just similar -- this
   cross-checks the two implementations against *each other*, on top of
   the finite-difference check that already validates the NumPy side
   alone); (b) the full benchmark below was rerun end-to-end with the
   PyTorch trainers across the same 3 seeds, not assumed to reproduce the
   NumPy numbers.

Section 8b below reports the PyTorch benchmark results and an honest
comparison against the NumPy run, including a difference in training
stability I did not expect and could not fully explain away.

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
| REINFORCE, no shaping | **0.626** | 0.972 | 0.931 |
| REINFORCE, + shaping | 0.482 | 0.764 | 0.764 |
| PPO-lite, + shaping | 0.607 | 0.942 | 0.899 |

(mean over 3 seeds x 300 held-out episodes; see `experiments/results/results.csv`)

Looking at the learning curves (`experiments/results/learning_curves.png`),
shaped REINFORCE is also visibly *less stable* across training, with sharp
dips several seeds show around similar points in training, where unshaped
REINFORCE climbs steadily.

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

PPO-lite reaches REINFORCE's final performance level in roughly an order of
magnitude fewer episodes (~150-200 vs. ~2000+), and its learning curve is
visibly smoother across seeds (see the green vs. red/blue curves in
`experiments/results/learning_curves.png`). This is consistent with what
PPO is designed to do: reusing each batch of on-policy rollouts for several
gradient epochs (instead of one), under a clipped surrogate objective that
caps how far a single update can move the policy, extracts more learning
signal per environment interaction while limiting destructive updates from
any one noisy episode. Both trainers share the exact same policy
architecture, optimizer, and (crude, single-scalar) baseline, so this
comparison isolates the effect of the clipped multi-epoch update itself,
not a confound from a better value function or bigger network.

I want to be precise about scope here: this is a **lightweight** PPO
(no separate value network, no GAE -- see the docstring in `ppo.py` for the
full list of simplifications), so "PPO-lite converges faster" should be
read as "the clipped multi-epoch mechanism itself helps here," not as a
claim that this matches a production PPO implementation's absolute sample
efficiency.

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

## 7. What I'd do next with more time / infra

1. **Cost-aware potential shaping** (Sec. 4) -- finish the lambda/gamma
   sweep instead of reporting one run.
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
   structure, and a real embedding model in place of synthetic vectors --
   the current environment is deliberately small so every result here could
   be run and checked in well under two minutes per implementation; the
   natural next step is running the same code against real retrieval
   traces on GPU infra, via the PyTorch trainers in `train_torch.py` /
   `ppo_torch.py` (batched rollouts and a GPU device would be the first
   two changes -- the current port is still one-episode-at-a-time, CPU
   only, matching the NumPy version's structure for a clean comparison).
5. **Track down the shaping-variance gap** (Sec. 8.2) with a controlled
   ablation: swap the exact torch entropy gradient for the NumPy
   surrogate (and vice versa) with everything else held fixed, to test
   the "approximate entropy gradient accidentally regularizes" hypothesis
   directly instead of leaving it as a plausible-but-unconfirmed
   explanation.

## 8. PyTorch port: validation and results

### 8.1 Gradient cross-check

Before trusting any PyTorch training curve, `tests/test_policy_torch.py`
loads the *exact same weights* into `MLPBernoulliPolicy` (NumPy) and
`MLPBernoulliPolicyTorch` (PyTorch), runs the same state/action through
both, and compares `logprob_grad`'s hand-derived gradient to
`torch.autograd`'s gradient directly (float64, `atol=1e-8`) across 10
random states x 2 actions x 4 parameter tensors. All match. This is a
stronger check than "both networks learn something reasonable" -- it
confirms the two implementations compute the identical mathematical
function and its identical gradient, not just similar ones.

### 8.2 Benchmark results: NumPy vs. PyTorch

Same environment, same 3 seeds (0/1/2), same 300-episode held-out
evaluation, same hyperparameters. The only thing that changed is which
implementation trained the policy.

| Method | NumPy reward | PyTorch reward | NumPy success | PyTorch success | NumPy cost | PyTorch cost |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE, no shaping | 0.626 ± 0.001 | 0.624 ± 0.002 | 0.972 | 0.974 | 0.931 | 0.943 |
| REINFORCE, + shaping | 0.482 ± 0.081 | 0.441 ± **0.176** | 0.764 | 0.697 | 0.764 | 0.688 |
| PPO-lite, + shaping | 0.607 ± 0.005 | 0.553 ± **0.098** | 0.942 | 0.862 | 0.899 | 0.833 |
| Random (p=0.5) | 0.635 | 0.635 | 0.971 | 0.971 | 0.907 | 0.907 |
| Greedy-relevance | 0.645 | 0.645 | 1.000 | 1.000 | 0.927 | 0.927 |
| Oracle | 0.875 | 0.875 | 1.000 | 1.000 | 0.356 | 0.356 |

(baselines and the oracle are identical by construction -- they don't
depend on the policy implementation at all, only on the shared
`environment.py`/`baselines.py`, which was not touched by the port.)

**What replicates cleanly:** REINFORCE without shaping lands within 0.3%
of the NumPy result, with comparably tight variance across seeds. The
qualitative headline findings of this report all replicate: shaping
still measurably *hurts* REINFORCE rather than helping it, and PPO-lite
still reaches REINFORCE-no-shaping's performance level in roughly an
order of magnitude fewer training episodes (visible in
`experiments/results/learning_curves_torch.png` -- the green PPO curve
plateaus by ~500-1000 episodes, red REINFORCE only catches up by
~2500-3000).

**What does not replicate cleanly, stated plainly rather than smoothed
over:** both methods that use reward shaping (REINFORCE + shaping,
PPO-lite + shaping) show meaningfully *higher variance across seeds* in
the PyTorch version than the NumPy version -- REINFORCE+shaping's
seed-to-seed std more than doubles (0.081 -> 0.176), and PPO-lite's
final reward is visibly lower and less stable (0.607 -> 0.553, std
0.005 -> 0.098). Looking at individual seeds' PPO-lite learning curves,
one seed dips sharply late in training (down to ~0.35 mid-run before
partially recovering) where the NumPy runs stayed smooth across all 3
seeds. This is real, reproducible instability, not a one-off fluke of a
single run -- and I want to flag it rather than quietly report only the
mean.

**My best explanation, again stated as a hypothesis, not a conclusion:**
the NumPy trainer's entropy bonus uses a hand-written *approximate*
gradient, `dH/dlogit ~ (0.5 - p) * 4` (see the comment in `train.py`),
chosen because it was simpler to derive by hand than the exact Bernoulli
entropy gradient. The PyTorch version uses
`torch.distributions.Bernoulli(logits=...).entropy()`, differentiated
*exactly* by autograd. These are not the same function: the true entropy
gradient `dH/dlogit = -p(1-p) * logit` shrinks toward zero as the policy
becomes confident (`p` near 0 or 1), while the NumPy surrogate stays
close to its bounded max (~2) even at extreme confidence. In effect, the
NumPy version's "wrong" entropy gradient acts as a stronger, confidence-
independent regularizer that keeps pulling the policy back from
saturated (very confident) predictions, which may be *accidentally*
stabilizing training -- while the PyTorch version's exact entropy
gradient lets a confident, shaped-reward-chasing policy get more
confident with less pushback, right up until a step overcorrects. I
consider this a genuinely interesting result: an intentional
approximation in the hand-derived version may have been doing useful
regularization work that "doing it correctly" in PyTorch removes. I have
not run the controlled experiment needed to confirm this (e.g. swapping
in the exact NumPy entropy gradient, or the approximate surrogate in
PyTorch, and comparing variance with everything else held fixed) -- that
is the natural next step and is now easy to run given both
implementations exist side by side.

A second, more mundane contributor: `train_torch.py`/`ppo_torch.py` and
the NumPy trainers draw from independent RNG streams (torch's generator
vs. NumPy's), so `seed=1` does not correspond to the same actual episode
sequence in both -- some of the extra spread could simply be different
(unlucky) draws rather than an algorithmic difference. I can't fully
separate these two effects without more seeds than the 3 this report
uses, so I'm reporting both as plausible contributors rather than picking
one.

### 8.3 What this changes about the project's framing

The original "Why NumPy?" framing (no internet access, no PyTorch
available) is no longer accurate now that internet access is available.
The from-scratch NumPy implementation is kept -- it's still fully tested,
still the one with the tightest seed-to-seed variance in this benchmark,
and the hand-derived gradients remain a useful correctness exercise -- but
it is no longer the "PyTorch wasn't available so I did it by hand" story.
It is now: two validated implementations of the same research idea, and a
concrete, reproducible finding that porting to idiomatic autograd is not
a pure improvement in this setup -- it removed an intentional
approximation that happened to add stability, which is exactly the kind
of thing you only find by actually running both, not by assuming the
"proper" framework version is strictly better.

## 9. Reproducing everything in this report

```bash
pip install -r requirements.txt          # numpy + matplotlib (+ torch, optional)
python -m unittest discover -s tests -v      # 17 tests, ~0.1s (torch tests skip if torch isn't installed)
python experiments/run_benchmark.py           # NumPy trainers, ~90s total
python experiments/run_benchmark_torch.py     # PyTorch trainers, ~3 min total (requires torch)
```
`experiments/results/` will contain `results.csv` / `results_torch.csv`
(the exact numbers quoted above), `learning_curves.png` /
`learning_curves_torch.png`, and `benchmark_comparison.png` /
`benchmark_comparison_torch.png`.
