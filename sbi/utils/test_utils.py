import torch
import torch.distributions
import sbi.utils as utils
from sbi.simulators.linear_gaussian import (
    get_true_posterior_log_prob_linear_gaussian_n_prior,
    get_true_posterior_log_prob_linear_gaussian_mvn_prior,
)
from sbi.inference.posteriors.sbi_posterior import Posterior


def dkl_gaussian_prior(
    posterior: Posterior, true_observation: torch.Tensor, num_dim: int
):
    """
    Test whether Kullback-Leibler divergence between estimated posterior (with Gaussian
     prior) and ground truth is below threshold.

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
    dkl = utils.dkl_monte_carlo_estimate(target_dist, posterior, num_samples=200)

    max_dkl = 0.05 if num_dim == 1 else 0.8

    assert (
        dkl < max_dkl
    ), f"D-KL={dkl} is more than 2 stds above the average performance."


def normalization_uniform_prior(
    posterior: Posterior,
    prior: torch.distributions.Distribution,
    true_observation: torch.Tensor,
    num_dim: int,
):
    """
    Check the correctness of the log_prob() function of the posterior with a uniform
     prior.

     Checks whether the returned probability outside of the prior support is zero and
      whether normalization (i.e. scaling up the density due to leakage into regions
      without prior support) scales up the density by the correct factor.

    Args:
        posterior: estimated posterior
        prior: prior distribution
        true_observation: observation where we evaluate the posterior
        num_dim: dimensionality of the problem
    """
    # test whether likelihood outside prior support is zero. Prior bounds are [-1, 1] in
    # each dimension, so tensor of 2s will be outside of bounds.
    sample_outside_support = 2 * torch.ones(num_dim)
    posterior_prob = torch.exp(posterior.log_prob(sample_outside_support))
    assert (
        posterior_prob == 0.0
    ), "The posterior probability outside of the prior support is not zero"

    # test whether normalization works
    prior_sample = prior.sample()
    # compute unnormalized density, i.e. simply the output of the density estimator
    posterior_likelihood_unnorm = torch.exp(
        posterior.log_prob(prior_sample, normalize_snpe=False)
    )
    # compute the normalized density, i.e. simply the output of the density
    # estimator scaled up by the ratio of posterior samples within the prior bounds.
    posterior_likelihood_norm = torch.exp(
        posterior.log_prob(prior_sample, normalize_snpe=True)
    )

    # estimate acceptance ratio through rejection sampling
    acceptance_prob = posterior.get_leakage_correction(context=true_observation)

    # The acceptance probability should be *exactly* the ratio of the unnormalized
    # and the normalized likelihood. However, we allow for an error margin of 1%,
    # since the estimation of the acceptance probability is random (based on
    # rejection sampling).
    assert (
        acceptance_prob * 0.99
        < posterior_likelihood_unnorm / posterior_likelihood_norm
        < acceptance_prob * 1.01
    ), "Normalizing the posterior density using the acceptance probability failed."
