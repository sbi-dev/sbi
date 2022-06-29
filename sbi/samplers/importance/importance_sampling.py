from math import sqrt
from typing import Tuple

import torch
from torch import Tensor


def importance_sample(
    potential_fn,
    proposal,
    num_samples: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Returns samples from proposal and log(importance weights).

    Args:
        potential_fn: Unnormalized potential function.
        proposal: Proposal distribution with `.sample()` and `.log_prob()` methods.
        num_samples: Number of samples to draw.

    Returns:
        Samples and logarithm of importance weights.
    """
    samples = proposal.sample((num_samples,))

    potential_logprobs = potential_fn(samples)
    proposal_logprobs = proposal.log_prob(samples)
    log_importance_weights = potential_logprobs - proposal_logprobs

    return samples, log_importance_weights


def exponentiate_weights(log_weights: Tensor) -> Tensor:
    """Subtracts the maximum of the `log_weights` and then exponentiates them.

    It also filters out infinite `log_weights`, thus the input and output shape can
    differ.

    Args:
        log_weights: Logarithm of the importance weights.

    Returns:
        Tensor: Importance weights.
    """
    log_weights = log_weights[torch.isfinite(log_weights)]
    logweights_max = log_weights.max()
    weights = torch.exp(log_weights - logweights_max)
    return weights


def largest_weight_indices(weights: Tensor) -> Tensor:
    """Returns the indizes of the largest weights.

    Args:
        weights: Weights of which to return the largest indices. Usually importance
            weights.

    Returns:
        Tensor: The indices of the largest importance weights.
    """
    # Compute number of weights that are used for estimating the Pareto distribution.
    # Vehtari, Gelman, Gabry, 2017.
    # Yao, Vehtari, Simpson, Gelman, 2018
    number_of_weights = int(min(len(weights) / 5, 3 * sqrt(len(weights))))
    _, inds = weights.sort()
    return inds[-number_of_weights:]


def gpdfit(
    x: Tensor, sorted: bool = True, eps: float = 1e-8, return_quadrature: bool = False
) -> Tuple:
    """Maximum a posteriori estimate of a Generalized Paretto distribution.

    Pytorch version of gpdfit according to
    https://github.com/avehtari/PSIS/blob/master/py/psis.py. This function will compute
    a MAP (more stable than the MLE estimator).

    Args:
        x: Tensor of floats, the data which is used to fit the GPD.
        sorted: If x is already sorted
        eps: Numerical stability jitter
        return_quadrature: Weather to return individual results.
    Returns:
        Tuple: Parameters of the Generalized Paretto Distribution.

    """
    if not sorted:
        x, _ = x.sort()
    N = len(x)
    PRIOR = 3
    M = 30 + int(sqrt(N))

    bs = torch.arange(1, M + 1, device=x.device)
    bs = 1 - torch.sqrt(M / (bs - 0.5))
    bs /= PRIOR * x[int(N / 4 + 0.5) - 1]
    bs += 1 / x[-1]

    ks = -bs
    temp = ks[:, None] * x
    ks = torch.log1p(temp).mean(dim=1)
    L = N * (torch.log(-bs / ks) - ks - 1)

    temp = torch.exp(L - L[:, None])
    w = 1 / torch.sum(temp, dim=1)

    dii = w >= 10 * eps
    if not torch.all(dii):
        w = w[dii]
        bs = bs[dii]
    w /= w.sum()

    # posterior mean for b
    b = torch.sum(bs * w)
    # Estimate for k
    temp = (-b) * x
    temp = torch.log1p(temp)
    k = torch.mean(temp)
    if return_quadrature:
        temp = -x
        temp = bs[:, None] * temp
        temp = torch.log1p(temp)
        ks = torch.mean(temp, dim=1)

    # estimate for sigma
    sigma = -k / b * N / (N - 0)
    # weakly informative prior for k
    a = 10
    k = k * N / (N + a) + a * 0.5 / (N + a)
    if return_quadrature:
        ks *= N / (N + a)
        ks += a * 0.5 / (N + a)

    if return_quadrature:
        return k, sigma, ks, w
    else:
        return k, sigma
