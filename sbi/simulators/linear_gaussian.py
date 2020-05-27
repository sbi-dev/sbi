from typing import Union

import torch
from torch import Tensor
from torch.distributions import Independent, MultivariateNormal, Uniform


def linear_gaussian(theta: Tensor, std=1.0) -> Tensor:

    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    return theta + std * torch.randn_like(theta)


def get_true_posterior_samples_linear_gaussian_mvn_prior(
    x_o: Tensor, num_samples: int = 1000, std: float = 1.0
):
    mean = x_o
    dim = mean.shape[1]
    std = torch.sqrt(torch.tensor([std ** 2 / (std ** 2 + 1)]))
    c = torch.tensor([1.0 / (std ** 2 + 1.0)])
    return c * mean + std * torch.randn(num_samples, dim)


def any_linear_gaussian(theta, likelihood_shift, likelihood_cov):
    """
    Simulator for linear Gaussian.

    Args:
        theta: parameter set.
        likelihood_shift: Mean of the likelihood will be likelihood_shift+theta
        likelihood_cov: Covariance matrix of the likelihood.

    Returns: Simulated data.
    """

    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    # Cholesky decomposition
    L = torch.cholesky(likelihood_cov)

    return likelihood_shift + theta + torch.mm(torch.randn_like(theta), L)


def _true_posterior_any_linear_gaussian_mvn_prior(
    x_o: Tensor,
    likelihood_shift: Tensor,
    likelihood_cov: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
) -> torch.distributions.MultivariateNormal:
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

    posterior_dist = torch.distributions.MultivariateNormal(product_mean, product_cov)

    return posterior_dist


def multiply_gaussian_pdfs(mu1, s1, mu2, s2):
    """
    Returns the Gaussian that is the product of two Gaussian pdfs.

    Args:
        mu1: Mean of first Gaussian.
        s1: Covariance matrix of first Gaussian.
        mu2: Mean of second Gaussian
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


def get_true_posterior_log_prob_linear_gaussian_n_prior(
    x_o: Tensor, std: float = 1.0
) -> torch.distributions.Distribution:
    """
    Returns the ground truth density when using just a single dimension.

    Returns: univariate Gaussian posterior distribution
    """
    mean = x_o
    dim = mean.shape[1]
    std = torch.sqrt(torch.tensor([std ** 2 / (std ** 2 + 1)]))
    c = torch.tensor([1.0 / (std ** 2 + 1.0)])
    target_dist = torch.distributions.Normal(c * mean, std)
    return target_dist


def get_true_posterior_log_prob_linear_gaussian_mvn_prior(
    x_o: Tensor, std: float = 1.0
) -> torch.distributions.Distribution:
    """
    Returns the ground truth density when using more than one dimension.

    Returns: multivariate Gaussian posterior distribution
    """
    mean = x_o
    dim = mean.shape[1]
    std = torch.sqrt(torch.tensor([std ** 2 / (std ** 2 + 1)]))
    c = torch.tensor([1.0 / (std ** 2 + 1.0)])
    target_dist = torch.distributions.MultivariateNormal(c * mean, std * torch.eye(dim))
    return target_dist


def get_true_posterior_samples_linear_gaussian_uniform_prior(
    x_o: Tensor, prior: Union[Uniform, Independent], num_samples: int = 1000, std=1,
):
    mean = x_o
    event_shape = mean.shape[1]
    posterior = MultivariateNormal(
        loc=mean, covariance_matrix=std * torch.eye(event_shape)
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
