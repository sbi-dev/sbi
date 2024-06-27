# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.density_estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.sbi_types import TorchTransform
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes, mcmc_transform
from sbi.utils.torchutils import atleast_2d


def ratio_estimator_based_potential(
    ratio_estimator: nn.Module,
    prior: Distribution,
    x_o: Optional[Tensor],
    enable_transform: bool = True,
) -> Tuple[Callable, TorchTransform]:
    r"""Returns the potential for ratio-based methods.

    It also returns a transformation that can be used to transform the potential into
    unconstrained space.

    Args:
        ratio_estimator: The neural network modelling likelihood-to-evidence ratio.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the likelihood-to-evidence ratio.
        enable_transform: Whether to transform parameters to unconstrained space.
            When False, an identity transform will be returned for `theta_transform`.

    Returns:
        The potential function and a transformation that maps
        to unconstrained space.
    """

    device = str(next(ratio_estimator.parameters()).device)

    potential_fn = RatioBasedPotential(ratio_estimator, prior, x_o, device=device)
    theta_transform = mcmc_transform(
        prior, device=device, enable_transform=enable_transform
    )

    return potential_fn, theta_transform


class RatioBasedPotential(BasePotential):
    def __init__(
        self,
        ratio_estimator: nn.Module,
        prior: Distribution,
        x_o: Optional[Tensor],
        device: str = "cpu",
    ):
        r"""Returns the potential for ratio-based methods.

        Args:
            ratio_estimator: The neural network modelling likelihood-to-evidence ratio.
            prior: The prior distribution.
            x_o: The observed data at which to evaluate the likelihood-to-evidence
                ratio.

        Returns:
            The potential function.
        """
        super().__init__(prior, x_o, device)
        self.ratio_estimator = ratio_estimator
        self.ratio_estimator.eval()

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        r"""Returns the potential for likelihood-ratio-based methods.

        Args:
            theta: The parameter set at which to evaluate the potential function.
            x_is_iid: Whether the observed data is iid.
            track_gradients: Whether to track the gradients.

        Returns:
            The potential.
        """
        if self.x_is_iid:
            # Calculate likelihood ratios over trials and in one batch.
            log_ratio_trial_sum = _log_ratios_over_trials(
                x=self.x_o,
                theta=theta.to(self.device),
                net=self.ratio_estimator,
                track_gradients=track_gradients,
            )

            # Move to cpu for comparison with prior.
            return log_ratio_trial_sum + self.prior.log_prob(theta)  # type: ignore
        else:
            # Calculate likelihood over batches.
            log_ratio_batches = _log_ratios_over_batches(
                x=self.x_o,
                theta=theta.to(self.device),
                net=self.ratio_estimator,
                track_gradients=track_gradients,
            )
            return log_ratio_batches + self.prior.log_prob(theta)  # type: ignore


def _log_ratios_over_trials(
    x: Tensor, theta: Tensor, net: nn.Module, track_gradients: bool = False
) -> Tensor:
    r"""Return log ratios summed over iid trials of `x`.

    Note: `x` can be a batch with batch size larger 1. Batches in x are assumed to
    be iid trials, i.e., data generated based on the same paramters / experimental
    conditions.

    Repeats `x` and $\theta$ to cover all their combinations of batch entries.

    Args:
        x: Batch of iid data of shape `(iid_dim, *event_shape)`.
        theta: Batch of parameters of shape `(batch_dim, *event_shape)`.
        net: neural net representing the classifier to approximate the ratio.
        track_gradients: Whether to track gradients.

    Returns:
        log_ratio_trial_sum: log ratio for each parameter, summed over all
            batch entries (iid trials) in `x`.
    """
    theta_repeated, x_repeated = match_theta_and_x_batch_shapes(
        theta=atleast_2d(theta), x=atleast_2d(x)
    )
    assert (
        x_repeated.shape[0] == theta_repeated.shape[0]
    ), "x and theta must match in batch shape."
    assert (
        next(net.parameters()).device == x.device and x.device == theta.device
    ), f"""device mismatch: net, x, theta: {next(net.parameters()).device}, {x.device},
        {theta.device}."""

    # Calculate ratios in one batch.
    with torch.set_grad_enabled(track_gradients):
        log_ratio_trial_batch = net([theta_repeated, x_repeated])
        # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
        log_ratio_trial_sum = log_ratio_trial_batch.reshape(x.shape[0], -1).sum(0)

    return log_ratio_trial_sum


def _log_ratios_over_batches(
    x: Tensor, theta: Tensor, net: nn.Module, track_gradients: bool = False
) -> Tensor:
    r"""Return log ratios for batch trials of `x`.

    Note: `x` can be a batch with batch size larger 1. Batches in `x` are assumed
    to be independent queries, e.g. MCMC chains with different target distributions.

    If `x` and $\theta$ have the same batch dimension, return the log likelihoods.
    Otherwise, assume that the batch dimension of $\theta$ is a multiple of the batch
    dimension of `x`. In this case, repeat `x` in order to cover all combinations of
    `x`, $\theta$.

    Args:
        x: Batch of iid data of shape `(iid_dim, *event_shape)`.
        theta: Batch of parameters of shape `(batch_dim, *event_shape)`.
        net: neural net representing the classifier to approximate the ratio.
        track_gradients: Whether to track gradients.

    Returns:
        log_ratio_trial_batch: log ratio for each parameter, conditioned on the
        corresponding entry of `x`.
    """
    # Shape of `x` is (condition_batch_dim, *event_shape).
    x = reshape_to_sample_batch_event(
        x, event_shape=x.shape[1:], leading_is_sample=True
    )

    # Shape of `theta` is (batch_dim, *event_shape). Therefore, the call below should
    # not change anything, and we just have it as "best practice" before calling
    # `DensityEstimator.log_prob`.
    theta = reshape_to_batch_event(theta, event_shape=theta.shape[1:])

    x_batch_size = x.shape[0]
    theta_batch_size = theta.shape[0]
    theta_per_x = theta_batch_size // x_batch_size
    assert (
        theta_batch_size % x_batch_size == 0
    ), f"Batch size mismatch: {theta.shape[0]} and {x.shape[0]}.\
        When performing batched mcmc sampling, the batch size of `theta` must be a\
        multiple of the batch size of `x`."

    x = torch.repeat_interleave(x, theta_per_x, dim=0)
    # density estimator expects (sample_dim, batch_dim, *event_shape)
    x = x.transpose(0, 1)
    x = x[0]

    # Calculate ratios in one batch.
    with torch.set_grad_enabled(track_gradients):
        log_ratio_trial_batch = net([theta, x])
        # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
        log_ratio_trial_batch = log_ratio_trial_batch.reshape(
            -1,
        )

    return log_ratio_trial_batch
