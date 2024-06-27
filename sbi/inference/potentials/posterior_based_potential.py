# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.density_estimators import DensityEstimator
from sbi.neural_nets.density_estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.sbi_types import TorchTransform
from sbi.utils import mcmc_transform
from sbi.utils.sbiutils import within_support


def posterior_estimator_based_potential(
    posterior_estimator: DensityEstimator,
    prior: Distribution,
    x_o: Optional[Tensor],
    enable_transform: bool = True,
) -> Tuple[Callable, TorchTransform]:
    r"""Returns the potential for posterior-based methods.

    It also returns a transformation that can be used to transform the potential into
    unconstrained space.

    The potential is the same as the log-probability of the `posterior_estimator`, but
    it is set to $-\inf$ outside of the prior bounds.

    Args:
        posterior_estimator: The neural network modelling the posterior.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the posterior.
        enable_transform: Whether to transform parameters to unconstrained space.
            When False, an identity transform will be returned for `theta_transform`.

    Returns:
        The potential function and a transformation that maps
        to unconstrained space.
    """

    device = str(next(posterior_estimator.parameters()).device)

    potential_fn = PosteriorBasedPotential(
        posterior_estimator, prior, x_o, device=device
    )

    theta_transform = mcmc_transform(
        prior, device=device, enable_transform=enable_transform
    )

    return potential_fn, theta_transform


class PosteriorBasedPotential(BasePotential):
    def __init__(
        self,
        posterior_estimator: DensityEstimator,
        prior: Distribution,
        x_o: Optional[Tensor],
        device: str = "cpu",
    ):
        r"""Returns the potential for posterior-based methods.

        The potential is the same as the log-probability of the `posterior_estimator`,
        but it is set to $-\inf$ outside of the prior bounds.

        Args:
            posterior_estimator: The neural network modelling the posterior.
            prior: The prior distribution.
            x_o: The observed data at which to evaluate the posterior.

        Returns:
            The potential function.
        """
        super().__init__(prior, x_o, device)
        self.posterior_estimator = posterior_estimator
        self.posterior_estimator.eval()

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        r"""Returns the potential for posterior-based methods.

        Args:
            theta: The parameter set at which to evaluate the potential function.
            track_gradients: Whether to track the gradients.

        Returns:
            The potential.
        """

        if self._x_o is None:
            raise ValueError(
                "No observed data x_o is available. Please reinitialize \
                the potential or manually set self._x_o."
            )
        # Force probability to be zero outside prior support.

        in_prior_support = within_support(self.prior, theta)
        if self.x_is_iid:
            # Calculate posterior probabilities over trials and in one batch.
            posterior_log_prob = _log_posteriors_over_trials(
                x=self._x_o,
                theta=theta.to(self.device),
                estimator=self.posterior_estimator,
                track_gradients=track_gradients,
            )
        else:
            # Calculate posterior probabilities over batches.
            posterior_log_prob = _log_posteriors_over_batches(
                x=self._x_o,
                theta=theta.to(self.device),
                estimator=self.posterior_estimator,
                track_gradients=track_gradients,
            )

        with torch.set_grad_enabled(track_gradients):
            posterior_log_prob = torch.where(
                in_prior_support,
                posterior_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self.device),
            )
        return posterior_log_prob


def _log_posteriors_over_trials(
    x: Tensor, theta: Tensor, estimator: DensityEstimator, track_gradients: bool = False
) -> Tensor:
    r"""Return log posterior probabilities for batch trials of `x`.

    Note: `x` can be a batch with batch size larger 1. Batches in `x` are assumed
    to be iid trials, i.e., data generated based on the same paramters /
    experimental conditions.

    Repeats `x` and $\theta$ to cover all their combinations of batch entries.

    Args:
        x: Batch of iid data of shape `(iid_dim, *event_shape)`.
        theta: Batch of parameters of shape `(batch_dim, *event_shape)`.
        estimator: DensityEstimator.
        track_gradients: Whether to track gradients.

    Returns:
        posterior_log_prob: log posterior probability for each parameter, summed over
        all batch entries (iid trials) in `x`.
    """
    theta = reshape_to_sample_batch_event(
        theta, event_shape=theta.shape[1:], leading_is_sample=False
    )
    x = reshape_to_batch_event(x, event_shape=x.shape[1:])

    # Match the number of `x` to the number of conditions (`theta`). This is important
    # if the potential is simulataneously evaluated at multiple `theta` (e.g.
    # multi-chain MCMC).
    theta_batch_size = theta.shape[1]
    trailing_minus_ones = [-1 for _ in range(x.dim() - 1)]
    x = x.expand(theta_batch_size, *trailing_minus_ones)

    assert (
        next(estimator.parameters()).device == x.device and x.device == theta.device
    ), f"""device mismatch: estimator, x, theta: \
        {next(estimator.parameters()).device}, {x.device},
        {theta.device}."""

    with torch.set_grad_enabled(track_gradients):
        posterior_log_prob = estimator.log_prob(theta, condition=x).sum(0)

    return posterior_log_prob


def _log_posteriors_over_batches(
    x: Tensor, theta: Tensor, estimator: DensityEstimator, track_gradients: bool = False
) -> Tensor:
    r"""Return log posterior probabilities for batch trials of `x`.

    Note: `x` can be a batch with batch size larger 1. Batches in `x` are assumed
    to be independent queries, e.g. MCMC chains with different target distributions.

    If `x` and $\theta$ have the same batch dimension, return the log posteriors.
    Otherwise, assume that the batch dimension of $\theta$ is a multiple of the batch
    dimension of `x`. In this case, repeat `x` in order to cover all combinations of
    `x`, $\theta$.

    Args:
        x: Batch of iid data of shape `(iid_dim, *event_shape)`.
        theta: Batch of parameters of shape `(batch_dim, *event_shape)`.
        estimator: DensityEstimator.
        track_gradients: Whether to track gradients.

    Returns:
        posterior_log_prob: log posterior probability for each parameter, conditioned on
        the corresponding entry of `x`.
    """
    # Shape of `x` is (condition_batch_dim, *event_shape).
    x = reshape_to_batch_event(x, event_shape=x.shape[1:])

    # Shape of `theta` is (batch_dim, *event_shape). Therefore, the call below should
    # not change anything, and we just have it as "best practice" before calling
    # `DensityEstimator.log_prob`.
    theta = reshape_to_sample_batch_event(
        theta, event_shape=theta.shape[1:], leading_is_sample=False
    )
    x_batch_size = x.shape[0]
    theta_batch_size = theta.shape[1]
    theta_per_x = theta_batch_size // x_batch_size
    assert (
        theta_batch_size % x_batch_size == 0
    ), f"Batch size mismatch: {theta.shape[0]} and {x.shape[0]}.\
        When performing batched mcmc sampling, the batch size of `theta` must be a\
        multiple of the batch size of `x`."

    x = torch.repeat_interleave(x, theta_per_x, dim=0)
    # density estimator expects (sample_dim, batch_dim, *event_shape)
    with torch.set_grad_enabled(track_gradients):
        posterior_log_prob = estimator.log_prob(theta, condition=x)

    return posterior_log_prob
