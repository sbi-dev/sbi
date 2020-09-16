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
    net: Any,
    x: Tensor,
    potential_fn_provider: Callable,
    init_strategy_num_candidates: int,
    condition: Optional[Tensor] = None,
    dims_to_sample: Optional[List] = None,
) -> Tensor:
    r"""
    Return a sample obtained by sequential importance reweighing.

    This function can also do `SIR` on the conditional posterior
    $p(\theta_i|\theta_j, x)$ when a `condition` and `dims_to_sample` are passed.

    Args:
        prior: Prior distribution, candidate samples are drawn from it.
        net: Neural network with `.log_prob()` method. Used to obtain weights for the
            candidate samples.
        x: Context at which to evaluate the `net`.
        potential_fn_provider: Returns the potential function that the candidate
            samples are weighted with.
        init_strategy_num_candidates: Number of candidate samples drawn.
        condition: If provided, we evaluate the candidate samples on the conditional
            log-probability. Should contain dim_theta elements, i.e. it could e.g. be
            a sample from the posterior distribution. The entries at all
            `dims_to_sample` will be ignored.
        dims_to_sample: Which dimensions to sample from. If `None`, sample from the
            full posterior distribution. If not `None`, we sample only from the
            dimensions in the list while the others are fixed to values specified in
            `condition`.

    Returns:
        A single sample.
    """

    net.eval()
    potential_fn = potential_fn_provider(prior, net, x, "slice_np")
    init_param_candidates = prior.sample((init_strategy_num_candidates,)).detach()

    if condition is None:
        theta_dim = len(prior.mean)
        dims_to_sample = list(range(theta_dim))
        condition = torch.empty(1, theta_dim)

    conditional_candidates = deepcopy(condition).repeat(init_strategy_num_candidates, 1)
    conditional_candidates[:, dims_to_sample] = init_param_candidates

    log_weights = torch.cat(
        [
            potential_fn(conditional_candidates[i, :]).detach()
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
    net.train(True)
    return conditional_candidates[torch.from_numpy(idxs.astype(int)), :]
