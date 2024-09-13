# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.estimators import (
    ConditionalDensityEstimator,
    MixedDensityEstimator,
)
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.sbi_types import TorchTransform
from sbi.utils.sbiutils import mcmc_transform


def likelihood_estimator_based_potential(
    likelihood_estimator: ConditionalDensityEstimator,
    prior: Distribution,
    x_o: Optional[Tensor],
    enable_transform: bool = True,
) -> Tuple[Callable, TorchTransform]:
    r"""Returns potential $\log(p(x_o|\theta)p(\theta))$ for likelihood-based methods.

    It also returns a transformation that can be used to transform the potential into
    unconstrained space.

    Args:
        likelihood_estimator: The density estimator modelling the likelihood.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the likelihood.
        enable_transform: Whether to transform parameters to unconstrained space.
             When False, an identity transform will be returned for `theta_transform`.

    Returns:
        The potential function $p(x_o|\theta)p(\theta)$ and a transformation that maps
        to unconstrained space.
    """

    device = str(next(likelihood_estimator.parameters()).device)

    potential_fn = LikelihoodBasedPotential(
        likelihood_estimator, prior, x_o, device=device
    )
    theta_transform = mcmc_transform(
        prior, device=device, enable_transform=enable_transform
    )

    return potential_fn, theta_transform


class LikelihoodBasedPotential(BasePotential):
    def __init__(
        self,
        likelihood_estimator: ConditionalDensityEstimator,
        prior: Distribution,
        x_o: Optional[Tensor] = None,
        device: str = "cpu",
    ):
        r"""Returns the potential function for likelihood-based methods.

        Args:
            likelihood_estimator: The density estimator modelling the likelihood.
            prior: The prior distribution.
            x_o: The observed data at which to evaluate the likelihood.
            device: The device to which parameters and data are moved before evaluating
                the `likelihood_nn`.

        Returns:
            The potential function $p(x_o|\theta)p(\theta)$.
        """

        super().__init__(prior, x_o, device)
        self.likelihood_estimator = likelihood_estimator
        self.likelihood_estimator.eval()

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        r"""Returns the potential $\log(p(x_o|\theta)p(\theta))$.

        Args:
            theta: The parameter set at which to evaluate the potential function.
            track_gradients: Whether to track the gradients.

        Returns:
            The potential $\log(p(x_o|\theta)p(\theta))$.
        """
        if self.x_is_iid:
            # For each theta, calculate the likelihood sum over all x in batch.
            log_likelihood_trial_sum = _log_likelihoods_over_trials(
                x=self.x_o,
                theta=theta.to(self.device),
                estimator=self.likelihood_estimator,
                track_gradients=track_gradients,
            )
            return log_likelihood_trial_sum + self.prior.log_prob(theta)  # type: ignore
        else:
            # Calculate likelihood for each (theta,x) pair separately
            theta_batch_size = theta.shape[0]
            x_batch_size = self.x_o.shape[0]
            assert (
                theta_batch_size == x_batch_size
            ), f"Batch size mismatch: {theta_batch_size} and {x_batch_size}.\
                When performing batched sampling for multiple `x`, the batch size of\
                `theta` must match the batch size of `x`."
            x = self.x_o.unsqueeze(0)
            with torch.set_grad_enabled(track_gradients):
                log_likelihood_batches = self.likelihood_estimator.log_prob(
                    x, condition=theta
                )
            return log_likelihood_batches + self.prior.log_prob(theta)  # type: ignore


def _log_likelihoods_over_trials(
    x: Tensor,
    theta: Tensor,
    estimator: ConditionalDensityEstimator,
    track_gradients: bool = False,
) -> Tensor:
    r"""Return log likelihoods summed over iid trials of `x`.

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
        log_likelihood_trial_sum: log likelihood for each parameter, summed over all
            batch entries (iid trials) in `x`.
    """
    # Shape of `x` is (iid_dim, *event_shape).
    x = reshape_to_sample_batch_event(
        x, event_shape=x.shape[1:], leading_is_sample=True
    )

    # Match the number of `x` to the number of conditions (`theta`). This is important
    # if the potential is simulataneously evaluated at multiple `theta` (e.g.
    # multi-chain MCMC).
    theta_batch_size = theta.shape[0]
    trailing_minus_ones = [-1 for _ in range(x.dim() - 2)]
    x = x.expand(-1, theta_batch_size, *trailing_minus_ones)

    assert (
        next(estimator.parameters()).device == x.device and x.device == theta.device
    ), f"""device mismatch: estimator, x, theta: \
        {next(estimator.parameters()).device}, {x.device},
        {theta.device}."""

    # Shape of `theta` is (batch_dim, *event_shape). Therefore, the call below should
    # not change anything, and we just have it as "best practice" before calling
    # `DensityEstimator.log_prob`.
    theta = reshape_to_batch_event(theta, event_shape=theta.shape[1:])

    # Calculate likelihood in one batch.
    with torch.set_grad_enabled(track_gradients):
        log_likelihood_trial_batch = estimator.log_prob(x, condition=theta)
        # Sum over trial-log likelihoods.
        log_likelihood_trial_sum = log_likelihood_trial_batch.sum(0)

    return log_likelihood_trial_sum


def mixed_likelihood_estimator_based_potential(
    likelihood_estimator: MixedDensityEstimator,
    prior: Distribution,
    x_o: Optional[Tensor],
) -> Tuple[Callable, TorchTransform]:
    r"""Returns $\log(p(x_o|\theta)p(\theta))$ for mixed likelihood-based methods.

    It also returns a transformation that can be used to transform the potential into
    unconstrained space.

    Args:
        likelihood_estimator: The neural network modelling the likelihood.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the likelihood.

    Returns:
        The potential function $p(x_o|\theta)p(\theta)$ and a transformation that maps
        to unconstrained space.
    """

    device = str(next(likelihood_estimator.discrete_net.parameters()).device)

    potential_fn = MixedLikelihoodBasedPotential(
        likelihood_estimator, prior, x_o, device=device
    )
    theta_transform = mcmc_transform(prior, device=device)

    return potential_fn, theta_transform


class MixedLikelihoodBasedPotential(LikelihoodBasedPotential):
    def __init__(
        self,
        likelihood_estimator: MixedDensityEstimator,
        prior: Distribution,
        x_o: Optional[Tensor],
        device: str = "cpu",
    ):
        super().__init__(likelihood_estimator, prior, x_o, device)

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        prior_log_prob = self.prior.log_prob(theta)  # type: ignore

        # Shape of `x` is (iid_dim, *event_shape)
        theta = reshape_to_batch_event(theta, event_shape=theta.shape[1:])
        x = reshape_to_sample_batch_event(
            self.x_o, event_shape=self.x_o.shape[1:], leading_is_sample=True
        )
        theta_batch_dim = theta.shape[0]
        # Match the number of `x` to the number of conditions (`theta`). This is
        # importantif the potential is simulataneously evaluated at multiple `theta`
        # (e.g. multi-chain MCMC).
        trailing_minus_ones = [-1 for _ in range(x.dim() - 2)]
        x = x.expand(-1, theta_batch_dim, *trailing_minus_ones)

        # Calculate likelihood in one batch.
        with torch.set_grad_enabled(track_gradients):
            # Call the specific log prob method of the mixed likelihood estimator as
            # this optimizes the evaluation of the discrete data part.
            # TODO log_prob_iid
            log_likelihood_trial_batch = self.likelihood_estimator.log_prob(
                input=x,
                condition=theta.to(self.device),
            )
            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
                self.x_o.shape[0], -1
            ).sum(0)

        return log_likelihood_trial_sum + prior_log_prob
