# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.simulators.linear_gaussian import true_posterior_linear_gaussian_mvn_prior
from sbi.types import Shape, TorchTransform
from sbi.utils import BoxUniform, within_support
from sbi.utils.metrics import c2st
from sbi.utils.torchutils import ensure_theta_batched


def kl_d_via_monte_carlo(
    p: Union[NeuralPosterior, Distribution],
    q: Union[NeuralPosterior, Distribution],
    num_samples: int = 1000,
) -> Tensor:
    r"""
    Returns Monte-Carlo estimate of the Kullback-Leibler divergence of distributions p,
    q.

    Unlike torch.distributions.kl.kl_divergence(p, q), this function does not require p
    and q to be `torch.Distribution` objects, but just to provide `sample()` and
    `log_prob()` methods.

    For added flexibility, we squeeze the output of log_prob() and hence can handle
    outputs such as torch.tensor([[p_1], [p_2], [p_3]]), instead of just
    torch.tensor([p_1, p_2, p_3]) (like torch.distributions.kl.kl_divergence(p, q)),
    with p_n being probabilities.

    Computes $D = \int p(x) * log(p(x)/q(x)) dx \approx 1/N * log(p(x)/q(x))$
    Args:
        p, q: distribution-like objects with sample() and log_prob() methods
        num_samples: number of samples that the Monte-Carlo estimate is based on
    """

    cumulative_log_ratio = torch.tensor([0.0])
    for _ in range(num_samples):
        target_sample = p.sample()
        # squeeze to make the shapes match. The output from log_prob() is either
        # torch.tensor([[p_1], [p_2], [p_3]]) or torch.tensor([p_1, p_2, p_3]), so we
        # squeeze to make both of them torch.tensor([p_1, p_2, p_3])
        cumulative_log_ratio += torch.squeeze(
            p.log_prob(target_sample)
        ) - torch.squeeze(q.log_prob(target_sample))

    dkl = cumulative_log_ratio / num_samples

    return dkl


def get_dkl_gaussian_prior(
    posterior: NeuralPosterior,
    x_o: Tensor,
    likelihood_shift: Tensor,
    likelihood_cov: Tensor,
    prior_mean: Tensor,
    prior_cov: Tensor,
    num_samples: int = 200,
) -> Tensor:
    """
    Return the Kullback-Leibler divergence between estimated posterior (with Gaussian
    prior) and ground-truth target posterior.

    Args:
        posterior: The estimated posterior.
        x_o: The observation where we evaluate the posterior.
        likelihood_shift: Mean of the likelihood p(x|theta) is likelihood_shift+theta.
        likelihood_cov: Covariance matrix of likelihood.
        prior_mean: Mean of prior.
        prior_cov: Covariance matrix of prior.
        num_samples: number of samples that the Monte-Carlo estimate is based on
    """

    target_dist = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )

    return kl_d_via_monte_carlo(target_dist, posterior, num_samples=num_samples)


def get_prob_outside_uniform_prior(
    posterior: NeuralPosterior, prior: BoxUniform, num_dim: int
) -> Tensor:
    """
    Return posterior probability for a parameter set outside of the prior support.

    Args:
        posterior: estimated posterior
        num_dim: dimensionality of the problem
    """
    # Test whether likelihood outside prior support is zero.
    assert isinstance(prior, BoxUniform)
    sample_outside_support = 1.1 * prior.base_dist.low
    assert not within_support(
        prior, sample_outside_support
    ).all(), "Samples must be outside of support."

    return torch.exp(posterior.log_prob(sample_outside_support))


def get_normalization_uniform_prior(
    posterior: DirectPosterior,
    prior: Distribution,
    x: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Return the unnormalized posterior likelihood, the normalized posterior likelihood,
    and the estimated acceptance probability.

    Args:
        posterior: estimated posterior
        prior: prior distribution
    """

    # Test normalization.
    prior_sample = prior.sample()

    # Compute unnormalized density, i.e. just the output of the density estimator.
    posterior_likelihood_unnorm = torch.exp(
        posterior.log_prob(prior_sample, norm_posterior=False)
    )
    # Compute the normalized density, scale up output of the density
    # estimator by the ratio of posterior samples within the prior bounds.
    posterior_likelihood_norm = torch.exp(
        posterior.log_prob(prior_sample, norm_posterior=True)
    )

    # Estimate acceptance ratio through rejection sampling.
    acceptance_prob = posterior.leakage_correction(x=x)

    return posterior_likelihood_unnorm, posterior_likelihood_norm, acceptance_prob


def check_c2st(x: Tensor, y: Tensor, alg: str, tol: float = 0.1) -> None:
    """Compute classification based two-sample test accuracy and assert it close to
    chance."""

    score = c2st(x, y).item()

    print(f"c2st for {alg} is {score:.2f}.")

    assert (
        (0.5 - tol) <= score <= (0.5 + tol)
    ), f"{alg}'s c2st={score:.2f} is too far from the desired near-chance performance."


class PosteriorPotential(BasePotential):
    allow_iid_x = False  # type: ignore

    def __init__(
        self,
        posterior: Distribution,
        prior: Distribution,
        x_o: Optional[Tensor] = None,
        device: str = "cpu",
    ):
        r"""Returns the potential for a closed-form posterior.

        The potential is the same as the log-probability of the posterior,
        but it is set to $-\inf$ outside of the prior bounds.

        Args:
            posterior: The posterior distribution
            prior: The prior distribution.
            x_o: The observed data at which to evaluate the posterior.

        Returns:
            The potential function.
        """
        super().__init__(prior, x_o, device)

        assert (
            x_o is None
        ), "No need to pass x_o, passed Posterior must be fixed to x_o."
        self.posterior = posterior

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        r"""Returns the potential for posterior-based methods.

        Args:
            theta: The parameter set at which to evaluate the potential function.
            track_gradients: Whether to track the gradients.

        Returns:
            The potential.
        """

        theta = ensure_theta_batched(torch.as_tensor(theta))

        with torch.set_grad_enabled(track_gradients):
            posterior_log_prob = self.posterior.log_prob(theta)

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta)

            posterior_log_prob = torch.where(
                in_prior_support,
                posterior_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self.device),
            )
        return posterior_log_prob


class TractablePosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x_o)$ with `log_prob()` and `sample()` methods, built from a
    potential function with tractable posterior distribution.<br/><br/>"""

    def __init__(
        self,
        potential_fn: Callable,
        theta_transform: Optional[TorchTransform] = None,
        device: Optional[str] = "cpu",
        x_shape: Optional[torch.Size] = None,
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples.
            theta_transform: Transformation that will be applied during sampling.
                Allows to perform, e.g. MCMC in unconstrained space.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            x_shape: Shape of the observed data.
        """
        assert isinstance(potential_fn, PosteriorPotential)
        super().__init__(potential_fn, theta_transform, device, x_shape)

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """See child classes for docstring."""

        return self.potential_fn.posterior.sample(sample_shape)

    def log_prob(
        self,
        theta: Tensor,
    ) -> Tensor:
        r"""Returns the log-probability of the posterior $p(\theta|x)$.

        Args:
            theta: Parameters $\theta$.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.
        """
        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.potential_fn.posterior.log_prob(theta)

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        force_update: bool = False,
    ) -> Tensor:
        """Returns stored maximum-a-posterior estimate (MAP), otherwise calculates it.

        See child classes for docstring.
        """

        return super().map(
            x=x,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            init_method=init_method,
            num_init_samples=num_init_samples,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
            force_update=force_update,
        )
