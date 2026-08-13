"""
policy_torch.py  (superseded -- kept for history, see below)
================

STATUS UPDATE: this file was originally written as an untested stub, back
when this sandbox had no internet access and no PyTorch installed -- it
was never run. Internet access is now available, `torch` is installed,
and this module has been superseded by the actively-used, actually-run,
and gradient-cross-checked implementation at
`src/context_selection/policy_torch.py` (used by `train_torch.py` and
`ppo_torch.py`, validated by `tests/test_policy_torch.py`, and exercised
end-to-end by `experiments/run_benchmark_torch.py`). See the top-level
README and `report.md` Sec. 8 for what actually ran and what it found.

This file is kept, unmodified in spirit, as a record of the original
plan and to show the delta between "a stub that should work" and "code
that was actually run and validated." The `__main__` block below now
does a real cross-check (loading identical weights and comparing to the
NumPy gradient exactly) instead of only printing a gradient tensor with
nothing to compare it against, since torch is now available to actually
run it.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from context_selection.policy import MLPBernoulliPolicy  # noqa: E402
from context_selection.policy_torch import (  # noqa: E402
    MLPBernoulliPolicyTorch,
    load_numpy_weights,
    to_float64,
)


if __name__ == "__main__":
    # Real cross-check (not just a printed tensor): identical weights,
    # identical state/action, NumPy hand-derived grad vs. torch.autograd
    # grad, compared directly.
    np_policy = MLPBernoulliPolicy(obs_dim=6, hidden_dim=5, seed=0)
    torch_policy = to_float64(MLPBernoulliPolicyTorch(obs_dim=6, hidden_dim=5, seed=0))
    load_numpy_weights(torch_policy, np_policy.params)

    rng = np.random.default_rng(0)
    s = rng.normal(size=6)
    action = 1

    _, cache = np_policy.forward(s)
    np_grad = np_policy.logprob_grad(cache, action, coeff=1.0)

    s_t = torch.as_tensor(s, dtype=torch.float64)
    logit = torch_policy(s_t)
    dist = torch.distributions.Bernoulli(logits=logit)
    logp = dist.log_prob(torch.as_tensor(float(action), dtype=torch.float64))
    logp.backward()

    print("NumPy   dW2:", np_grad["W2"])
    print("Torch   dW2:", torch_policy.fc2.weight.grad.numpy().reshape(-1))
    print("max abs diff (W2):", np.max(np.abs(np_grad["W2"] - torch_policy.fc2.weight.grad.numpy().reshape(-1))))
    print("(see tests/test_policy_torch.py for the full assertion-based")
    print(" version of this check, across more states/actions/params.)")
