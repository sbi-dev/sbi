import torch
import torch.distributions
import sbi.utils as utils
from sbi.simulators.linear_gaussian import (
    get_true_posterior_log_prob_linear_gaussian_n_prior,
    get_true_posterior_log_prob_linear_gaussian_mvn_prior,
)
from sbi.inference.posteriors.sbi_posterior import Posterior
from tests.utils_for_testing.dkl import dkl_via_monte_carlo


def get_dkl_gaussian_prior(
    posterior: Posterior, true_observation: torch.Tensor, num_dim: int
):
    """
    Return the Kullback-Leibler divergence between estimated posterior (with Gaussian
     prior) and ground truth posterior.

    Args:
        posterior: estimated posterior
        true_observation: observation where we evaluate the posterior
        num_dim: dimensionality of the problem
    """
    # load ground truth density
    if num_dim == 1:
        target_dist = get_true_posterior_log_prob_linear_gaussian_n_prior(
            true_observation,
        )
    else:
        target_dist = get_true_posterior_log_prob_linear_gaussian_mvn_prior(
            true_observation,
        )

    # get Kullback-Leibler divergence from the ground truth distribution to the
    # obtained posterior
    dkl = dkl_via_monte_carlo(target_dist, posterior, num_samples=200)

    return dkl


def get_prob_outside_uniform_prior(posterior: Posterior, num_dim: int) -> torch.Tensor:
    """
    Return probability outside of a parameter set outside of the prior support.

    Args:
        posterior: estimated posterior
        num_dim: dimensionality of the problem
    """
    # test whether likelihood outside prior support is zero. Prior bounds are [-1, 1] in
    # each dimension, so tensor of 2s will be outside of bounds.
    sample_outside_support = 2 * torch.ones(num_dim)
    posterior_prob = torch.exp(posterior.log_prob(sample_outside_support))

    return posterior_prob


def get_normalization_uniform_prior(
    posterior: Posterior,
    prior: torch.distributions.Distribution,
    true_observation: torch.Tensor,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Return the unnormalized posterior likelihood, the normalized posterior likelihood,
     and the estimated acceptance probability

    Args:
        posterior: estimated posterior
        prior: prior distribution
        true_observation: observation where we evaluate the posterior
    """

    # test whether normalization works
    prior_sample = prior.sample()
    # compute unnormalized density, i.e. simply the output of the density estimator
    posterior_likelihood_unnorm = torch.exp(
        posterior.log_prob(prior_sample, normalize_snpe_density=False)
    )
    # compute the normalized density, i.e. simply the output of the density
    # estimator scaled up by the ratio of posterior samples within the prior bounds.
    posterior_likelihood_norm = torch.exp(
        posterior.log_prob(prior_sample, normalize_snpe_density=True)
    )

    # estimate acceptance ratio through rejection sampling
    acceptance_prob = posterior.get_leakage_correction(context=true_observation)

    return posterior_likelihood_unnorm, posterior_likelihood_norm, acceptance_prob
