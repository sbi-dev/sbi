from copy import deepcopy
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from torch import Tensor


def prior_init(prior: Any) -> Tensor:
    """Return a sample from the prior."""
    return prior.sample((1,)).detach()


def sir(
    prior: Any,
    potential_fn: Callable,
    init_strategy_num_candidates: int,
    batch_size: int = 1000,
) -> Tensor:
    r"""
    Return a sample obtained by sequential importance reweighing.

    This function can also do `SIR` on the conditional posterior
    $p(\theta_i|\theta_j, x)$ when a `condition` and `dims_to_sample` are passed.

    Args:
        prior: Prior distribution, candidate samples are drawn from it.
        potential_fn: Potential function that the candidate samples are weighted with.
            Note that the function needs to return log probabilities.
        init_strategy_num_candidates: Number of candidate samples drawn.
        batch_size: Batch size used for evaluating candidates.

    Returns:
        A single sample.
    """
    with torch.set_grad_enabled(False):
        num_batches = int(init_strategy_num_candidates / batch_size)

        log_weights = []
        init_param_candidates = []
        for i in range(num_batches):
            batch_draws = prior.sample((batch_size,)).detach()
            init_param_candidates.append(batch_draws)
            log_weights.append(potential_fn(batch_draws.numpy()).detach())
        log_weights = torch.cat(log_weights)
        init_param_candidates = torch.cat(init_param_candidates)

        probs = np.exp(log_weights.view(-1).numpy().astype(np.float64))
        probs[np.isnan(probs)] = 0.0
        probs[np.isinf(probs)] = 0.0
        probs /= probs.sum()

        idxs = np.random.choice(
            a=np.arange(init_strategy_num_candidates), size=1, replace=False, p=probs,
        )

        return init_param_candidates[torch.from_numpy(idxs.astype(int)), :]
