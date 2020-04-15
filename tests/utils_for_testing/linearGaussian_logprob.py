from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution
import torch.distributions

from sbi.inference.posteriors.sbi_posterior import Posterior
from sbi.simulators.linear_gaussian import (
    get_true_posterior_log_prob_linear_gaussian_mvn_prior,
    get_true_posterior_log_prob_linear_gaussian_n_prior,
)
from tests.utils_for_testing.dkl import dkl_via_monte_carlo


def get_dkl_gaussian_prior(
    posterior: Posterior, true_observation: Tensor, num_dim: int
) -> Tensor:
    """
    Return the Kullback-Leibler divergence between estimated posterior (with Gaussian
    prior) and ground-truth target posterior.

    Args:
        posterior: estimated posterior
        true_observation: observation where we evaluate the posterior
        num_dim: dimensionality of the problem
    """

    if num_dim == 1:
        target_dist = get_true_posterior_log_prob_linear_gaussian_n_prior(
            true_observation,
        )
    else:
        target_dist = get_true_posterior_log_prob_linear_gaussian_mvn_prior(
            true_observation,
        )

    return dkl_via_monte_carlo(target_dist, posterior, num_samples=200)


def get_prob_outside_uniform_prior(posterior: Posterior, num_dim: int) -> Tensor:
    """
    Return posterior probability for a parameter set outside of the prior support.

    Args:
        posterior: estimated posterior
        num_dim: dimensionality of the problem
    """
    # Test whether likelihood outside prior support is zero. Prior bounds are [-1, 1] in
    # each dimension, so tensor of 2s will be out of bounds.
    sample_outside_support = 2 * torch.ones(num_dim)

    return torch.exp(posterior.log_prob(sample_outside_support))


def get_normalization_uniform_prior(
    posterior: Posterior, prior: Distribution, true_observation: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Return the unnormalized posterior likelihood, the normalized posterior likelihood,
    and the estimated acceptance probability.

    Args:
        posterior: estimated posterior
        prior: prior distribution
        true_observation: observation where we evaluate the posterior
    """

    # Test normalization.
    prior_sample = prior.sample()

    # Compute unnormalized density, i.e. just the output of the density estimator.
    posterior_likelihood_unnorm = torch.exp(
        posterior.log_prob(prior_sample, normalize_snpe_density=False)
    )
    # Compute the normalized density, scale up output of the density
    # estimator by the ratio of posterior samples within the prior bounds.
    posterior_likelihood_norm = torch.exp(
        posterior.log_prob(prior_sample, normalize_snpe_density=True)
    )

    # Estimate acceptance ratio through rejection sampling.
    acceptance_prob = posterior.get_leakage_correction(x=true_observation)

    return posterior_likelihood_unnorm, posterior_likelihood_norm, acceptance_prob
