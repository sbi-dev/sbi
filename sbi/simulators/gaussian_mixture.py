import logging

import pyro
import torch
from pyro import distributions as pdist
from torch import Tensor

SIM_PARAMS = {
    "mixture_locs_factor": [1.0, 1.0],
    "mixture_scales": [1.0, 0.1],
    "mixture_weights": [0.5, 0.5],
}

PRIOR_PARAMS = {
    "bound": 10.0,
}


def uniform_prior_gaussian_mixture(dim: int, bound: float = 10.0) -> pdist.Uniform:
    """
    Prior distribution for Gaussian Mixture, as implemented in [1].

    Args:
        dim: Dimensionality of parameters and data.
        bound: Prior is uniform in [-bound, +bound], defaults to 10.0.

    Returns: Prior distribution.
    """
    return pdist.Uniform(
        low=-bound * torch.ones((dim,)),
        high=+bound * torch.ones((dim,)),
    ).to_event(1)


def gaussian_mixture(
    theta: Tensor,
    mixture_locs_factor: list = SIM_PARAMS["mixture_locs_factor"],
    mixture_scales: list = SIM_PARAMS["mixture_scales"],
    mixture_weights: list = SIM_PARAMS["mixture_weights"],
) -> Tensor:
    """
    Simulator for Gaussian Mixture, as implemented in [1].

    The mixture components are Gaussians with scaled theta as mean and fixed scale:
    `num_components = dim_theta`, default is 2.

    The dimensionality of the data can be changed, but the mixture parameters
    (locs, scales, weights) need to be adjusted accordingly.

    References:
    [1]: https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/gaussian_mixture/task.py

    Args:
        theta: Parameter sets to be simulated, of shape (num_samples, dim_theta).
        mixture_locs_factor: Factor for the mean of the Gaussian mixture components,
            of length (dim_theta).
        mixture_scales: Scales of the Gaussian mixture components,
            of length (dim_theta).
        mixture_weights: Weights of the Gaussian mixture components,
            of length (dim_theta).

    Returns: Simulated data, of shape (num_samples, dim_theta).
    """

    # Check dimensions
    assert (
        theta.shape[-1]
        == len(mixture_locs_factor)
        == len(mixture_scales)
        == len(mixture_weights)
    ), "Mismatch in dimensions."

    # Sample mixture index for each parameter in batch
    idx = pyro.sample(
        "mixture_idx",
        pdist.Categorical(probs=torch.tensor(mixture_weights)).expand_by([
            theta.shape[0]
        ]),
    ).unsqueeze(1)

    # Select loc and scales according to mixture index
    loc = torch.tensor(mixture_locs_factor)[idx] * theta
    scale = torch.tensor(mixture_scales)[idx]

    return pyro.sample("data", pdist.Normal(loc=loc, scale=scale).to_event(1))


def samples_true_posterior_gaussian_mixture_uniform_prior(
    x_o: Tensor,
    mixture_locs_factor: list = SIM_PARAMS["mixture_locs_factor"],
    mixture_scales: list = SIM_PARAMS["mixture_scales"],
    mixture_weights: list = SIM_PARAMS["mixture_weights"],
    prior_bound: float = 10.0,
    num_samples: int = 1,
) -> torch.Tensor:
    """Samples the true posterior for a given observation x_o when
    the likelihood is a Gaussian Mixture and the prior is uniform.

    The dimensionality of the data is 2 by default, but can be changed if
    the mixture parameters (locs, scales, weights) are adjusted accordingly.

    Uses closed form solution with rejection sampling, as implemented in [1].

    References:
    [1]: https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/gaussian_mixture/task.py

    Args:
        x_o: The observation, of shape (,dim_x).
        mixture_locs_factor: Factor for the mean of the Gaussian mixture components,
            of length (dim_x).
        mixture_scales: Scales of the Gaussian mixture components,
            of length (dim_x).
        mixture_weights: Weights of the Gaussian mixture components,
            of length (dim_x).
        prior_bound: Prior is uniform in [-prior_bound, +prior_bound],
            defaults to 10.0, as in [1].
        num_samples: Desired number of samples, defaults to 1.

    Returns:
        Samples from the true posterior, of shape (num_samples, dim_x).
    """

    log = logging.getLogger(__name__)

    dim = x_o.shape[-1]

    # Check dimensions
    assert (
        dim == len(mixture_locs_factor) == len(mixture_scales) == len(mixture_weights)
    ), "Mismatch in dimensions."

    # Define prior distribution
    prior_dist = uniform_prior_gaussian_mixture(dim, prior_bound)

    posterior_samples = []

    # Reject samples outside of prior bounds
    counter = 0
    while len(posterior_samples) < num_samples:
        counter += 1
        idx = pyro.sample(
            "mixture_idx",
            pdist.Categorical(torch.tensor(mixture_weights)),
        )
        sample = pyro.sample(
            "posterior_sample",
            pdist.Normal(
                loc=torch.tensor(mixture_locs_factor)[idx] * x_o,
                scale=torch.tensor(mixture_scales)[idx],
            ),
        )
        is_outside_prior = torch.isinf(prior_dist.log_prob(sample).sum())

        if len(posterior_samples) > 0:
            is_duplicate = sample in torch.cat(posterior_samples)
        else:
            is_duplicate = False

        if not is_outside_prior and not is_duplicate:
            posterior_samples.append(sample)

    posterior_samples = torch.cat(posterior_samples)
    acceptance_rate = float(num_samples / counter)

    log.info(f"Acceptance rate: {acceptance_rate}")

    return posterior_samples
