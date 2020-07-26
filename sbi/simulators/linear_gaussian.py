# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Callable, Optional, Union, Dict, Any, Tuple, Union, cast, List, Sequence, TypeVar

import torch
from torch import Tensor
from torch.distributions import Independent, MultivariateNormal, Uniform


def diagonal_linear_gaussian(theta: Tensor, std=1.0) -> Tensor:
    """
    Returns samples drawn from Gaussian likelihood with diagonal covariance.

    Args:
        theta: Parameter set.
        std: Standard deviation.

    Returns: Simulated data.
    """

    return theta + std * torch.randn_like(theta)


def linear_gaussian(
    theta: Tensor,
    likelihood_shift: Tensor,
    likelihood_cov: Tensor,
    num_discarded_dims: int = 0,
) -> Tensor:
    """
    Simulator for linear Gaussian.

    Uses Cholesky decomposition to transform samples from standard Gaussian.

    When `num_discarded_dims>0`, return simulation outputs with as many last dimensions
    discarded. This is implemented by throwing away the last `num_discarded_dims`
    dimensions of theta and then running the linear Gaussian as always.

    Args:
        theta: Parameter sets to be simulated.
        likelihood_shift: The simulator will shift the value of theta by this value.
            Thus, the mean of the Gaussian likelihood will be likelihood_shift+theta.
        likelihood_cov: Covariance matrix of the likelihood.
        num_discarded_dims: Number of output dimensions to discard.

    Returns: Simulated data.
    """

    if num_discarded_dims:
        theta = theta[:, :-num_discarded_dims]

    chol_factor = torch.cholesky(likelihood_cov)

    return likelihood_shift + theta + torch.mm(chol_factor, torch.randn_like(theta).T).T


def true_posterior_linear_gaussian_mvn_prior(
    x_o: Tensor,
    likelihood_shift: Tensor,
    likelihood_cov: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
) -> MultivariateNormal:
    """
    Returns the posterior when likelihood and prior are Gaussian.

    We follow the implementation suggested by rhashimoto here:
    https://math.stackexchange.com/questions/157172 as it requires only one matrix
    inverse.

    Args:
        x_o: The observation.
        likelihood_shift: Mean of the likelihood p(x|theta) is likelihood_shift+theta.
        likelihood_cov: Covariance matrix of likelihood.
        prior_mean: Mean of prior.
        prior_cov: Covariance matrix of prior.

    Returns: Posterior distribution.
    """

    # Let s denote the likelihood_shift:
    # The likelihood has the term (x-(s+theta))^2 in the exponent of the Gaussian.
    # In other words, as a function of x, the mean of the likelihood is s+theta.
    # For computing the posterior we need the likelihood as a function of theta. Hence:
    # (x-(s+theta))^2 = (theta-(-s+x))^2
    # We see that the mean is -s+x = x-s
    likelihood_mean = x_o - likelihood_shift

    product_mean, product_cov = multiply_gaussian_pdfs(
        likelihood_mean, likelihood_cov, prior_mean, prior_cov
    )

    posterior_dist = MultivariateNormal(product_mean, product_cov)

    return posterior_dist


def samples_true_posterior_linear_gaussian_mvn_prior_different_dims(
    x_o: Tensor,
    likelihood_shift: Tensor,
    likelihood_cov: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
    num_discarded_dims: int = 1,
    num_samples: int = 1,
) -> Tensor:
    """
    Returns the posterior when likelihood and prior are Gaussian.

    We follow the implementation suggested by rhashimoto here:
    https://math.stackexchange.com/questions/157172 as it requires only one matrix
    inverse.

    Args:
        x_o: The observation.
        likelihood_shift: Mean of the likelihood p(x|theta) is likelihood_shift+theta.
        likelihood_cov: Covariance matrix of likelihood.
        prior_mean: Mean of prior.
        prior_cov: Covariance matrix of prior.
        num_discarded_dims: Number of output dimensions to discard.
        num_samples: Desired number of samples.

    Returns: Posterior distribution.
    """

    posterior_dist = true_posterior_linear_gaussian_mvn_prior(
        x_o,
        likelihood_shift,
        likelihood_cov,
        prior_mean[:-num_discarded_dims],
        prior_cov[:-num_discarded_dims, :-num_discarded_dims],
    )
    posterior_samples = posterior_dist.sample((num_samples,))

    # Because some dimensions were discarded, these ground truth parameters have to
    # be sampled from the prior and then concatenated to the samples obtained above.
    prior_dist = MultivariateNormal(prior_mean, prior_cov)
    prior_samples = prior_dist.sample((num_samples,))
    relevant_prior_samples = prior_samples[:, -num_discarded_dims:]
    posterior_samples = torch.cat((posterior_samples, relevant_prior_samples), dim=1)

    return posterior_samples


def samples_true_posterior_linear_gaussian_uniform_prior(
    x_o: Tensor,
    likelihood_shift: Tensor,
    likelihood_cov: Tensor,
    prior: Union[Uniform, Independent],
    num_samples: int = 1000,
) -> Tensor:
    """
    Returns ground truth posterior samples for Gaussian likelihood and uniform prior.

    Args:
        x_o: The observation.
        likelihood_shift: Mean of the likelihood p(x|theta) is likelihood_shift+theta.
        likelihood_cov: Covariance matrix of likelihood.
        prior: Uniform prior distribution.
        num_samples: Desired number of samples.

    Returns: Samples from posterior.
    """

    # Let s denote the likelihood_shift:
    # The likelihood has the term (x-(s+theta))^2 in the exponent of the Gaussian.
    # In other words, as a function of x, the mean of the likelihood is s+theta.
    # For computing the posterior we need the likelihood as a function of theta. Hence:
    # (x-(s+theta))^2 = (theta-(-s+x))^2
    # We see that the mean is -s+x = x-s
    likelihood_mean = x_o - likelihood_shift

    posterior = MultivariateNormal(
        loc=likelihood_mean, covariance_matrix=likelihood_cov
    )

    # generate samples from ND Gaussian truncated by prior support
    num_remaining = num_samples
    samples = []

    while num_remaining > 0:
        candidate_samples = posterior.sample(sample_shape=(num_remaining,))
        is_in_prior = torch.isfinite(prior.log_prob(candidate_samples))
        # accept if in prior
        if is_in_prior.sum():
            samples.append(candidate_samples[is_in_prior, :])
            num_remaining -= is_in_prior.sum().item()

    return torch.cat(samples)


def multiply_gaussian_pdfs(mu1, s1, mu2, s2) -> Tuple[Tensor, Tensor]:
    """
    Returns the mean and cov of the Gaussian that is the product of two Gaussian pdfs.

    Args:
        mu1: Mean of first Gaussian.
        s1: Covariance matrix of first Gaussian.
        mu2: Mean of second Gaussian.
        s2: Covariance matrix of second Gaussian.

    Returns: Mean and covariance of the product of the two distributions.
    """

    inv_s1s2 = torch.inverse(s1 + s2)

    # posterior mean = s2 * inv_s1pluss2 * mu1 + s1 * inv_s1pluss2 * mu2
    product_mean = torch.mv(torch.mm(s2, inv_s1s2), mu1) + torch.mv(
        torch.mm(s1, inv_s1s2), mu2
    )

    # posterior cov = s1 * inv_s1pluss2 * s2
    product_cov = torch.mm(torch.mm(s1, inv_s1s2), s2)

    return product_mean, product_cov
