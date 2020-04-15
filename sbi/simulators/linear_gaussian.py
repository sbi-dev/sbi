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
