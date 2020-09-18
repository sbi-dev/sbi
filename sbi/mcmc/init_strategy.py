from copy import deepcopy
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from torch import Tensor


def prior_init(prior: Any) -> Tensor:
    """Return a sample from the prior."""
    return prior.sample((1,)).detach()


def sir(
    prior: Any, potential_fn: Callable, init_strategy_num_candidates: int,
) -> Tensor:
    r"""
    Return a sample obtained by sequential importance reweighing.

    This function can also do `SIR` on the conditional posterior
    $p(\theta_i|\theta_j, x)$ when a `condition` and `dims_to_sample` are passed.

    Args:
        prior: Prior distribution, candidate samples are drawn from it.
        potential_fn: Potential function that the candidate samples are weighted with.
        init_strategy_num_candidates: Number of candidate samples drawn.

    Returns:
        A single sample.
    """

    init_param_candidates = prior.sample((init_strategy_num_candidates,)).detach()

    log_weights = torch.cat(
        [
            potential_fn(init_param_candidates[i, :]).detach()
            for i in range(init_strategy_num_candidates)
        ]
    )
    probs = np.exp(log_weights.view(-1).numpy().astype(np.float64))
    probs[np.isnan(probs)] = 0.0
    probs[np.isinf(probs)] = 0.0
    probs /= probs.sum()
    idxs = np.random.choice(
        a=np.arange(init_strategy_num_candidates), size=1, replace=False, p=probs,
    )
    return init_param_candidates[torch.from_numpy(idxs.astype(int)), :]
