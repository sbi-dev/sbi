# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import Tensor, nn
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    MixtureSameFamily,
    MultivariateNormal,
    Normal,
    Uniform,
    constraints,
)

from sbi.utils.torchutils import BoxUniform

class VectorFieldNet(nn.Module, ABC):
    """Abstract base class for vector field estimation networks.

    Used by both flow matching and score matching approaches.
    """

    @abstractmethod
    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass to compute the vector field.

        Args:
            theta: Parameters
            x: Conditioning information
            t: Time parameter (scalar or batched)

        Returns:
            Vector field evaluation at the provided points
        """
        pass


def _diag_gaussian_log_prob(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    """Log-probability of a diagonal Gaussian evaluated at x."""
    diff = x[:, None, :] - mean[None, :, :]
    log_det = torch.log(var).sum(-1)
    quad = (diff**2 / var).sum(-1)
    d = diff.shape[-1]
    return -0.5 * (d * math.log(2 * math.pi) + log_det + quad)


def _full_gaussian_log_prob(x: Tensor, mean: Tensor, cov: Tensor) -> Tensor:
    """Log-probability of a full-covariance Gaussian evaluated at x."""
    diff = x[:, None, :] - mean[None, :, :]
    chol = torch.linalg.cholesky(cov)
    diff_col = diff[..., None]
    solve = torch.cholesky_solve(diff_col, chol[None, ...])
    quad = (diff_col * solve).sum(-2).squeeze(-1)
    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    d = diff.shape[-1]
    return -0.5 * (d * math.log(2 * math.pi) + log_det + quad)


def _gmm_density(
    x: Tensor,
    weights: Tensor,
    means: Tensor,
    cov: Tensor,
    covariance_type: str,
) -> Tensor:
    """Evaluate a generalized GMM density (unnormalized weights allowed)."""
    if covariance_type == "diag":
        log_probs = _diag_gaussian_log_prob(x, means, cov)
    elif covariance_type == "full":
        log_probs = _full_gaussian_log_prob(x, means, cov)
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")
    return torch.sum(weights[None, :] * torch.exp(log_probs), dim=1)


def _fit_gmm_ratio_impl(
    train_prior: Distribution,
    test_prior: Distribution,
    num_components: int,
    num_steps: int = 10_000,
    batch_size: int = 1_000,
    lr: float = 1e-2,
    covariance_type: str = "diag",
    min_cov: float = 1e-4,
    max_log_ratio: float = 50.0,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Implementation for fitting a generalized GMM to the prior ratio q(theta) / p(theta)."""
    num_components = int(num_components)
    device = torch.device(device)
    if hasattr(train_prior, "to"):
        moved = train_prior.to(device)  # type: ignore
        if moved is not None:
            train_prior = moved
    if hasattr(test_prior, "to"):
        moved = test_prior.to(device)  # type: ignore
        if moved is not None:
            test_prior = moved
    with torch.no_grad():
        init_means = test_prior.sample((num_components,)).to(device)
        init_means = init_means.reshape(num_components, -1)

    d = init_means.shape[-1]
    raw_weights = torch.ones(num_components, device=device, requires_grad=True)
    means = init_means.clone().detach().requires_grad_(True)

    if covariance_type == "diag":
        raw_cov = torch.full(
            (num_components, d),
            math.log(math.exp(1.0) - 1.0),
            device=device,
            requires_grad=True,
        )
    elif covariance_type == "full":
        raw_cov = torch.zeros(num_components, d, d, device=device, requires_grad=True)
        raw_cov.data.copy_(torch.eye(d, device=device)[None, ...])
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    optimizer = torch.optim.Adam([raw_weights, means, raw_cov], lr=lr)

    with torch.enable_grad():
        for _ in range(num_steps):
            theta = train_prior.sample((batch_size,)).to(device)
            theta = theta.reshape(batch_size, -1)
            log_p = train_prior.log_prob(theta).reshape(batch_size, -1).sum(-1)
            log_q = test_prior.log_prob(theta).reshape(batch_size, -1).sum(-1)
            log_p = torch.where(
                torch.isfinite(log_p), log_p, torch.full_like(log_p, -1e8)
            )
            log_q = torch.where(
                torch.isfinite(log_q), log_q, torch.full_like(log_q, -1e8)
            )
            log_ratio = log_q - log_p
            log_ratio = torch.where(
                torch.isfinite(log_ratio), log_ratio, torch.full_like(log_ratio, -1e8)
            )
            log_ratio = torch.clamp(log_ratio, max=max_log_ratio)
            ratio = torch.exp(log_ratio)

            weights = raw_weights
            if covariance_type == "diag":
                cov = F.softplus(raw_cov) + min_cov
            else:
                cov = raw_cov
                cov = cov @ cov.transpose(-1, -2)
                cov = cov + min_cov * torch.eye(d, device=device)[None, ...]

            gmm_val = _gmm_density(theta, weights, means, cov, covariance_type)
            loss = torch.mean((gmm_val - ratio) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    weights = raw_weights.detach()
    means = means.detach()
    if covariance_type == "diag":
        cov = F.softplus(raw_cov).detach() + min_cov
    else:
        cov = raw_cov.detach()
        cov = cov @ cov.transpose(-1, -2)
        cov = cov + min_cov * torch.eye(d, device=device)[None, ...]

    return weights, means, cov


class _HashableById:
    __slots__ = ("obj", "_id")

    def __init__(self, obj: Distribution):
        """Wraps a non-hashable Distribution to make it cache-key compatible."""
        self.obj = obj
        self._id = id(obj)

    def __hash__(self) -> int:
        return self._id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _HashableById) and self._id == other._id


def _ensure_hashable(dist: Distribution) -> Distribution | _HashableById:
    """Return a hashable object for caching, wrapping when needed."""
    try:
        hash(dist)
        return dist
    except TypeError:
        return _HashableById(dist)


@lru_cache(maxsize=32)
def _fit_gmm_ratio_cached(
    train_prior_h: Distribution | _HashableById,
    test_prior_h: Distribution | _HashableById,
    num_components: int,
    num_steps: int = 10_000,
    batch_size: int = 1_000,
    lr: float = 1e-2,
    covariance_type: str = "diag",
    min_cov: float = 1e-4,
    max_log_ratio: float = 50.0,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Cached wrapper for GMM ratio fitting.

    This avoids re-running the costly optimization when the same priors and
    hyperparameters are used repeatedly (e.g. when a guidance adaptor is
    constructed per potential call).
    """
    train_prior = (
        train_prior_h.obj if isinstance(train_prior_h, _HashableById) else train_prior_h
    )
    test_prior = (
        test_prior_h.obj if isinstance(test_prior_h, _HashableById) else test_prior_h
    )
    return _fit_gmm_ratio_impl(
        train_prior,
        test_prior,
        num_components,
        num_steps=num_steps,
        batch_size=batch_size,
        lr=lr,
        covariance_type=covariance_type,
        min_cov=min_cov,
        max_log_ratio=max_log_ratio,
        device=device,
    )


def fit_gmm_ratio(
    train_prior: Distribution,
    test_prior: Distribution,
    num_components: int,
    num_steps: int = 10_000,
    batch_size: int = 1_000,
    lr: float = 1e-2,
    covariance_type: str = "diag",
    min_cov: float = 1e-4,
    max_log_ratio: float = 50.0,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fits a generalized GMM to the prior ratio q(theta) / p(theta).

    Uses an LRU cache keyed by priors and hyperparameters to avoid re-fitting when
    guidance adapters are repeatedly instantiated with the same settings.

    References:
    - "Prior Guidance for Diffusion Models" (arXiv:2510.13763)
    """
    return _fit_gmm_ratio_cached(
        _ensure_hashable(train_prior),
        _ensure_hashable(test_prior),
        num_components,
        num_steps=num_steps,
        batch_size=batch_size,
        lr=lr,
        covariance_type=covariance_type,
        min_cov=min_cov,
        max_log_ratio=max_log_ratio,
        device=device,
    )


# -----------------------------------------------------------------------------
# Moved from score_utils.py
# -----------------------------------------------------------------------------
# Automatic denoising ------------------------------------------------------


def denoise(p: Distribution, m: Tensor, s: Tensor, x_t: Tensor) -> Distribution:
    """Returns the denoised distribution p(X | X_t = x_t).

    This return the posterior distribution p(X | X_t = x_t) given the prior
    distribution p(X) and the following generative process X_t = m * X + s*ε, where
    ε ~ N(0, 1).

    Args:
        p: The prior distribution p(X).
        m: The scaling factor m.
        s: The standard deviation of the noise s.
        x_t: The observation X_t.

    Returns:
        The posterior distribution p(X | X_t = x_t).
    """
    if isinstance(p, Independent):
        return denoise_independent(p, m, s, x_t)
    elif isinstance(p, MixtureSameFamily):
        return denoise_mixture(p, m, s, x_t)
    elif isinstance(p, Normal):
        return denoise_gaussian(p, m, s, x_t)
    elif isinstance(p, MultivariateNormal):
        return denoise_multivariate_gaussian(p, m, s, x_t)
    elif isinstance(p, (Uniform, BoxUniform)):
        return denoise_uniform(p, m, s, x_t)
    else:
        return denoise_general(p, m, s, x_t)


def denoise_independent(
    p: Independent, m: Tensor, s: Tensor, x_t: Tensor
) -> Independent | Distribution:
    """Denoise an independent distribution.

    Args:
        p: The prior independent distribution.
        m: The scaling factor m.
        s: The standard deviation of the noise s.
        x_t: The observation X_t.

    Returns:
        The posterior independent distribution.
    """
    denoised_base_dist = denoise(p.base_dist, m, s, x_t)
    batch_shape = denoised_base_dist.batch_shape
    if len(batch_shape) < p.reinterpreted_batch_ndims:
        return denoised_base_dist
    else:
        return Independent(denoised_base_dist, p.reinterpreted_batch_ndims)


def denoise_gaussian(p: Normal, m: Tensor, s: Tensor, x_t: Tensor) -> Normal:
    """Denoise a Gaussian distribution.

    Args:
        p: The prior Gaussian distribution.
        m: The scaling factor.
        s: The standard deviation of the noise.
        x_t: The observed data.

    Returns:
        The posterior Gaussian distribution.
    """
    mean_0 = p.loc
    std_0 = p.scale

    # Calculate the posterior mean and variance
    posterior_variance = 1 / (1 / std_0**2 + m**2 / s**2)
    posterior_mean = posterior_variance * (mean_0 / std_0**2 + m * x_t / s**2)
    posterior_std = torch.sqrt(posterior_variance)

    return Normal(posterior_mean, posterior_std)


def denoise_multivariate_gaussian(
    p: MultivariateNormal, m: Tensor, s: Tensor, x_t: Tensor
) -> MultivariateNormal:
    """Denoise a multivariate Gaussian distribution.

    Args:
        p: The prior multivariate Gaussian distribution.
        m: The scaling factor.
        s: The standard deviation of the noise.
        x_t: The observed data.

    Returns:
        The posterior multivariate Gaussian distribution.
    """
    mean0 = p.loc
    cov0 = p.covariance_matrix
    n = cov0.size(-1)  # type: ignore
    # Support batch dimensions by expanding the identity matrix to match cov0's
    # batch shape
    batch_shape = cov0.shape[:-2]  # type: ignore
    id_matrix = torch.eye(n, dtype=cov0.dtype, device=cov0.device)  # type: ignore
    id_matrix = id_matrix.expand(*batch_shape, n, n)
    precision_prior = torch.linalg.inv(cov0)
    # Reshape m and s so the operation broadcasts correctly over batch dims
    precision_likelihood = (m**2 / s**2)[..., None, None] * id_matrix
    posterior_cov = torch.linalg.inv(precision_prior + precision_likelihood)
    # unsqueeze mean0 and x_t for proper matrix multiplication
    term1 = torch.matmul(precision_prior, mean0.unsqueeze(-1))
    term2 = (m / s**2)[..., None] * x_t.unsqueeze(-1)
    posterior_mean = torch.matmul(posterior_cov, term1 + term2).squeeze(-1)
    return MultivariateNormal(posterior_mean, covariance_matrix=posterior_cov)


def denoise_mixture(
    p: MixtureSameFamily, m: Tensor, s: Tensor, x_t: Tensor
) -> MixtureSameFamily:
    """Denoise a MixtureSameFamily distribution.

    Args:
        p: The prior MixtureSameFamily distribution.
        m: The scaling factor m.
        s: The standard deviation of the noise.
        x_t: The observed data.

    Returns:
        The posterior MixtureSameFamily distribution.
    """
    mixture_dist = p.mixture_distribution
    component_dist = p.component_distribution
    denoised_components = denoise(component_dist, m, s, x_t)
    mixture_logits = mixture_dist.logits
    # Update the logits to reflect the new likelihoods
    component_loglikelihood = denoised_components.log_prob(x_t)
    # Update the logits to reflect the new likelihoods
    denoised_logits = mixture_logits[None, ...] + component_loglikelihood
    # Normalize the logits
    denoised_logits = denoised_logits - denoised_logits.logsumexp(dim=-1, keepdims=True)
    return MixtureSameFamily(Categorical(logits=denoised_logits), denoised_components)


def denoise_uniform(
    p: Uniform | BoxUniform, m: Tensor, s: Tensor, x_t: Tensor
) -> 'UniformNormalPosterior':
    """Denoise a uniform distribution.

    Args:
        p: The prior uniform distribution.
        m: The scaling factor m.
        s: The standard deviation of the noise s.
        x_t: The observed data.

    Returns:
        The posterior uniform distribution.
    """
    return UniformNormalPosterior(p.low, p.high, m, s, x_t)  # type: ignore


def denoise_general(
    p: Distribution,
    m: Tensor,
    s: Tensor,
    x_t: Tensor,
) -> MixtureSameFamily:
    """Denoise a general distribution.

    This is an approximate method intended for general distributions. It fits a
    Gaussian Mixture Model (GMM) to the empirical distribution, which can then be
    denoised analytically using the denoise_mixture function.

    NOTE: Why GMM not KDE or MC methods? This is because in one intended use case we
    require access to the score function (i.e. the gradient of the log density), and
    this gradient is rather ill-behave for particle-based approximations.

    Args:
        p: The prior empirical distribution.
        m: The scaling factor m.
        s: The standard deviation of the noise s.
        x_t: The observed data.
        num_particles: Number of particles to sample. Defaults to 5000.

    Returns:
        The posterior empirical distribution.
    """
    gmm = fit_gmm(p)
    return denoise_mixture(gmm, m, s, x_t)


# Automatic marginalization -----------------------------------------------


def marginalize(p: Distribution, m: Tensor, s: Tensor) -> Distribution:
    """Given the prior distribution p(X), scaling factor m, and standard deviation of
    the noise s, return the marginal distribution p(X_t).

    Args:
        p: The prior distribution p(X).
        m: The scaling factor m.
        s: The standard deviation of the noise s.

    Raises:
        NotImplementedError: If the distribution is not supported.

    Returns:
        The marginal distribution p(X_t).
    """
    if isinstance(p, Independent):
        return marginalize_independent(p, m, s)
    elif isinstance(p, Normal):
        return marginalize_gaussian(p, m, s)
    elif isinstance(p, MultivariateNormal):
        return marginalize_multivariate_gaussian(p, m, s)
    elif isinstance(p, (Uniform, BoxUniform)):
        return marginalize_uniform(p, m, s)
    elif isinstance(p, MixtureSameFamily):
        return marginalize_mixture(p, m, s)
    else:
        return marginalize_empirical(p, m, s)


def marginalize_independent(
    p: Independent, m: Tensor, s: Tensor
) -> Independent | Distribution:
    """Marginalize an independent distribution.

    Args:
        p: The prior independent distribution.
        m: The scaling factor m.
        s: The standard deviation of the noise s.

    Returns:
        The marginal independent distribution.
    """
    marg_base_dist = marginalize(p.base_dist, m, s)
    batch_shape = marg_base_dist.batch_shape
    if len(batch_shape) < p.reinterpreted_batch_ndims:
        return marg_base_dist
    else:
        return Independent(marg_base_dist, p.reinterpreted_batch_ndims)


def marginalize_mixture(
    p: MixtureSameFamily, m: Tensor, s: Tensor
) -> MixtureSameFamily:
    """Marginalize a MixtureSameFamily distribution.

    Args:
        p: The prior MixtureSameFamily distribution.
        m: The scaling factor m.
        s: The standard deviation of the noise s.

    Returns:
        The marginal MixtureSameFamily distribution.
    """
    return MixtureSameFamily(
        p.mixture_distribution, marginalize(p.component_distribution, m, s)
    )


def marginalize_gaussian(p: Normal, m: Tensor, s: Tensor) -> Normal:
    """Marginalize a Gaussian distribution.

    Args:
        p: The prior Gaussian distribution.
        m: The scaling factor.
        s: The standard deviation of the noise.

    Returns:
        The marginal Gaussian distribution.
    """
    mean_0 = p.loc
    std_0 = p.scale

    # Calculate the marginal mean and variance
    marginal_mean = m * mean_0
    marginal_variance = (m * std_0) ** 2 + s**2
    marginal_std = torch.sqrt(marginal_variance)

    return Normal(marginal_mean, marginal_std)


def marginalize_multivariate_gaussian(
    p: MultivariateNormal, m: Tensor, s: Tensor
) -> MultivariateNormal:
    """Marginalize a multivariate Gaussian distribution.

    Given an observation model xₜ = m * x + ε with independent noise ε ~ N(0, s² I),
    where x ~ p, return the marginal distribution p(xₜ) as a MultivariateNormal.

    Args:
        p: The prior multivariate Gaussian distribution.
        m: The scaling factor.
        s: The standard deviation of the noise.

    Returns:
        The marginal multivariate Gaussian distribution p(xₜ).
    """
    mean_0 = p.loc
    cov_0 = p.covariance_matrix

    marginal_mean = m * mean_0
    marginal_cov = (m**2) * cov_0 + s**2 * torch.eye(
        mean_0.shape[-1],
        dtype=cov_0.dtype,  # type: ignore
        device=cov_0.device,  # type: ignore
    )

    return MultivariateNormal(marginal_mean, covariance_matrix=marginal_cov)


def marginalize_uniform(
    p: Uniform | BoxUniform, m: Tensor, s: Tensor
) -> 'UniformNormalConvolution':
    """Marginalize a uniform distribution.

    Args:
        p: The prior uniform distribution.
        m: The scaling factor.
        s: The standard deviation of the noise.

    Returns:
        The marginal uniform distribution.
    """

    return UniformNormalConvolution(p.low, p.high, m, s)  # type: ignore


def marginalize_empirical(
    p: Distribution, m: Tensor, s: Tensor, num_particles: int = 5000
) -> MixtureSameFamily:
    """Marginalize an empirical distribution.

    Args:
        p: The prior empirical distribution.
        m: The scaling factor m.
        s: The standard deviation of the noise s.
        num_particles: Number of particles to sample. Defaults to 5000.

    Returns:
        The marginal empirical distribution.
    """
    gmm_approx = fit_gmm(p)
    return marginalize_mixture(gmm_approx, m, s)


# Special distributions:
class UniformNormalPosterior(Distribution):
    """Posterior distribution for a uniform prior and normal likelihood.

    Args:
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.
        m: Scaling factor.
        s: Standard deviation of the noise.
        x_t: Observed data.
        validate_args: Whether to validate arguments. Defaults to None.
    """

    arg_constraints = {
        "low": constraints.real,
        "high": constraints.real,
        "m": constraints.real,
        "s": constraints.positive,
        "x_t": constraints.real,
    }  # type: ignore
    has_rsample: bool = False

    def __init__(
        self,
        low: Tensor | float,
        high: Tensor | float,
        m: Tensor | float,
        s: Tensor | float,
        x_t: Tensor | float,
        validate_args: Optional[bool] = None,
    ) -> None:
        self.low = torch.as_tensor(low)
        self.high = torch.as_tensor(high)
        self.m = torch.as_tensor(m)
        self.s = torch.as_tensor(s)
        self.x_t = torch.as_tensor(x_t)

        # Posterior is a truncated normal with:
        self.mu = self.x_t / self.m
        self.sigma = self.s / torch.abs(self.m)

        # Standard Normal Distribution
        self.standard_normal = Normal(torch.tensor(0.0, device=self.sigma.device), 1.0)

        # Standardized truncation limits
        self.alpha = (self.low - self.mu) / self.sigma
        self.beta = (self.high - self.mu) / self.sigma

        # Compute normalization constant Z
        self.a = self.standard_normal.cdf(self.alpha)
        self.b = self.standard_normal.cdf(self.beta)
        self.Z = torch.clamp(self.b - self.a, min=1e-8)  # Avoid division by zero

        batch_shape = torch.broadcast_shapes(
            self.low.shape, self.high.shape, self.m.shape, self.s.shape, self.x_t.shape
        )
        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        # Standard normal pdf values
        phi_alpha = torch.exp(self.standard_normal.log_prob(self.alpha))
        phi_beta = torch.exp(self.standard_normal.log_prob(self.beta))

        return self.mu + self.sigma * (phi_alpha - phi_beta) / self.Z

    @property
    def variance(self) -> Tensor:
        phi_alpha = torch.exp(self.standard_normal.log_prob(self.alpha))
        phi_beta = torch.exp(self.standard_normal.log_prob(self.beta))

        term = (phi_alpha - phi_beta) / self.Z
        variance_adjustment = (
            1 + (self.alpha * phi_alpha - self.beta * phi_beta) / self.Z - term**2
        )

        return self.sigma**2 * variance_adjustment

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        # Inverse CDF sampling for a truncated normal
        u = torch.rand(sample_shape, device=self.low.device) * self.Z + self.a
        sample = self.mu + self.sigma * self.standard_normal.icdf(
            u
        )  # Using icdf for stability
        return torch.clamp(sample, min=self.low, max=self.high)

    def log_prob(self, x: Tensor) -> Tensor:
        lp = Normal(self.mu, self.sigma).log_prob(x) - torch.log(self.Z)
        return torch.where((x >= self.low) & (x <= self.high), lp, -torch.inf)


class UniformNormalConvolution(Distribution):
    """Convolution of a uniform distribution with a normal distribution.

    Args:
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.
        scale: Scaling factor.
        noise: Standard deviation of the noise.
        validate_args: Whether to validate arguments. Defaults to None.
    """

    arg_constraints = {
        "low": constraints.real,
        "high": constraints.real,
        "scale": constraints.real,
        "noise": constraints.positive,
    }  # type: ignore
    support = constraints.real  # type: ignore
    has_rsample: bool = False

    def __init__(
        self,
        low: Tensor | float,
        high: Tensor | float,
        scale: Tensor | float,
        noise: Tensor | float,
        validate_args=None,
    ) -> None:
        self.low = torch.as_tensor(low)
        self.high = torch.as_tensor(high)
        self.scale = torch.as_tensor(scale)
        self.noise = torch.as_tensor(noise)

        # Determine batch_shape from broadcasting low and high (assumes low/high
        # are tensors or scalars)
        batch_shape = self.low.shape

        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = sample_shape + self.batch_shape  # type: ignore
        # Sample uniformly over [low, high]
        x = torch.rand(shape, device=self.low.device)
        x = x * (self.high - self.low) + self.low
        noise_sample = torch.randn(shape, device=self.noise.device) * self.noise
        return self.scale * x + noise_sample

    def log_prob(self, value: Tensor) -> Tensor:
        # Compute p(value) using the convolution formula:
        # p(value) = 1/(b-a) * [Phi((value - scale*a)/noise)
        # - Phi((value - scale*b)/noise)]
        dist0 = Normal(torch.tensor(0.0, device=value.device), 1.0)
        numerator = dist0.cdf((value - self.scale * self.low) / self.noise) - dist0.cdf(
            (value - self.scale * self.high) / self.noise
        )
        denom = (self.high - self.low).clamp_min(1e-30)
        pdf = numerator / denom
        return torch.log(pdf.clamp_min(1e-30))


# Utility functions --------------------------------------------------------


@lru_cache()
def fit_gmm(
    distribution: Distribution,
    num_components: int = 10,
    num_samples: Optional[int] = None,
    random_state: int = 0,
) -> MixtureSameFamily:
    """Fit a Gaussian Mixture Model (GMM) to a distribution.

    Args:
        distribution: The distribution to fit the GMM to.
        num_components: Number of GMM components. Defaults to 10.
        num_samples: Number of samples to draw from the distribution.
            Defaults to 10_000.
        random_state: Random state for reproducibility. Defaults to 0.

    Returns:
        The fitted GMM as a MixtureSameFamily distribution.
    """
    # Sample particles from the distribution
    if num_samples is None:
        d = distribution.event_shape.numel()
        num_samples = 1000 * d * num_components
    samples = distribution.sample((num_samples,))
    # Reshape to 2D array: (num_samples, features)
    samples_np = samples.detach().cpu().numpy().reshape(num_samples, -1)

    # Fit the Gaussian Mixture Model using scikit-learn
    gmm = GaussianMixture(n_components=num_components, random_state=random_state).fit(
        samples_np
    )

    # Convert parameters to torch tensors
    weights = torch.tensor(gmm.weights_, dtype=samples.dtype, device=samples.device)
    means = torch.tensor(gmm.means_, dtype=samples.dtype, device=samples.device)
    covariances = torch.tensor(
        gmm.covariances_, dtype=samples.dtype, device=samples.device
    )

    # Decide on univariate or multivariate based on feature dimension
    d = means.shape[-1]
    if d == 1:
        means = means.squeeze(-1)
        covariances = covariances.squeeze(-1).squeeze(-1)
        std = torch.sqrt(covariances)
        components = Normal(means, std)
    else:
        components = MultivariateNormal(means, covariance_matrix=covariances)

    # Construct and return the MixtureSameFamily distribution
    return MixtureSameFamily(Categorical(probs=weights), components)


def mv_diag_or_dense(A_diag_or_dense: Tensor, b: Tensor, batch_dims: int = 0) -> Tensor:
    """Dot product for diagonal or dense matrices.

    Args:
        A_diag_or_dense: Diagonal or dense matrix.
        b: Dense matrix/vector.
        batch_dims: Number of batch dimensions. Defaults to 0.

    Returns:
        The result of A * b (or A @ b).
    """
    A_dims = A_diag_or_dense.ndim - batch_dims
    if A_dims == 1:
        return A_diag_or_dense * b
    else:
        return torch.einsum('...ij,...j->...i', A_diag_or_dense, b)


def solve_diag_or_dense(
    A_diag_or_dense: Tensor, b: Tensor, batch_dims: int = 0
) -> Tensor:
    """Solve a linear system with a diagonal or dense matrix.

    Args:
        A_diag_or_dense: Diagonal or dense matrix.
        b: Dense matrix/vector.
        batch_dims: Number of batch dimensions. Defaults to 0.

    Returns:
        The solution to the linear system A x = b.
    """
    A_dim = A_diag_or_dense.ndim - batch_dims
    if A_dim == 1:
        return b / A_diag_or_dense
    else:
        return torch.linalg.solve(A_diag_or_dense, b)


def add_diag_or_dense(
    A_diag_or_dense: Tensor, B_diag_or_dense: Tensor, batch_dims: int = 0
) -> Tensor:
    """Add two diagonal or dense matrices, considering batch dimensions.

    Args:
        A_diag_or_dense: Diagonal or dense matrix.
        B_diag_or_dense: Diagonal or dense matrix.
        batch_dims: Number of batch dimensions. Defaults to 0.

    Returns:
        The sum of the two matrices.
    """
    A_ndim = A_diag_or_dense.ndim - batch_dims
    B_ndim = B_diag_or_dense.ndim - batch_dims

    if (A_ndim == 1 and B_ndim == 1) or (A_ndim == 2 and B_ndim == 2):
        return A_diag_or_dense + B_diag_or_dense
    elif A_ndim == 2 and B_ndim == 1:
        return A_diag_or_dense + torch.diag_embed(B_diag_or_dense)
    elif A_ndim == 1 and B_ndim == 2:
        return torch.diag_embed(A_diag_or_dense) + B_diag_or_dense
    else:
        raise ValueError("Incompatible dimensions for addition")
