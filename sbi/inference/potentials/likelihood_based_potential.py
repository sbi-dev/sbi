# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Callable, List, Optional, Tuple, Union

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

    def to(self, device: Union[str, torch.device]) -> None:
        """
        Move likelihood_estimator, prior and x_o to the given device.

        It also set the device attribute to the given device.

        Args:
            device: Device to move the likelihood_estimator, prior and x_o to.
        """
        self.device = device
        self.likelihood_estimator.to(device)
        self.prior.to(device)
        if self._x_o:
            self._x_o = self._x_o.to(device)

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
            assert theta_batch_size == x_batch_size, (
                f"Batch size mismatch: {theta_batch_size} and {x_batch_size}.\
                When performing batched sampling for multiple `x`, the batch size of\
                `theta` must match the batch size of `x`."
            )
            x = self.x_o.unsqueeze(0)
            with torch.set_grad_enabled(track_gradients):
                log_likelihood_batches = self.likelihood_estimator.log_prob(
                    x, condition=theta
                )
            return log_likelihood_batches + self.prior.log_prob(theta)  # type: ignore

    def condition_on_theta(
        self, local_theta: Tensor, dims_global_theta: List[int]
    ) -> Callable:
        r"""Returns a potential function conditioned on a subset of theta dimensions.

        The goal of this function is to divide the original `theta` into a
        `global_theta` we do inference over, and a `local_theta` we condition on (in
        addition to conditioning on `x_o`). Thus, the returned potential function will
        calculate $\prod_{i=1}^{N}p(x_i | local_theta_i, \global_theta)$, where `x_i`
        and `local_theta_i` are fixed and `global_theta` varies at inference time.

        Args:
            local_theta: The condition values to be conditioned.
            dims_global_theta: The indices of the columns in `theta` that will be
                sampled, i.e., that *not* conditioned. For example, if original theta
                has shape `(batch_dim, 3)`, and `dims_global_theta=[0, 1]`, then the
                potential will set `theta[:, 3] = local_theta` at inference time.

        Returns:
            A potential function conditioned on the `local_theta`.
        """

        assert self.x_is_iid, "Conditioning is only supported for iid data."

        def conditioned_potential(
            theta: Tensor, x_o: Optional[Tensor] = None, track_gradients: bool = True
        ) -> Tensor:
            assert len(dims_global_theta) == theta.shape[-1], (
                "dims_global_theta must match the number of parameters to sample."
            )
            if theta.dim() > 2:
                assert theta.shape[0] == 1, (
                    "condition_on_theta does not support sample shape for theta."
                )
                theta = theta.squeeze(0)
            global_theta = theta[:, dims_global_theta]
            x_o = x_o if x_o is not None else self.x_o
            # x needs shape (sample_dim (iid), batch_dim (xs), *event_shape)
            if x_o.dim() < 3:
                x_o = reshape_to_sample_batch_event(
                    x_o, event_shape=x_o.shape[1:], leading_is_sample=self.x_is_iid
                )

            return _log_likelihood_over_iid_trials_and_local_theta(
                x=x_o.to(self.device),
                global_theta=global_theta.to(self.device),
                local_theta=local_theta.to(self.device),
                estimator=self.likelihood_estimator,
                track_gradients=track_gradients,
            )

        return conditioned_potential


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


def _log_likelihood_over_iid_trials_and_local_theta(
    x: Tensor,
    global_theta: Tensor,
    local_theta: Tensor,
    estimator: ConditionalDensityEstimator,
    track_gradients: bool = False,
) -> Tensor:
    """Returns $\\prod_{i=1}^N \\log(p(x_i|\theta, local_theta_i)$.

    `x` is a batch of iid data, and `local_theta` is a matching batch of condition
    values that were part of `theta` but are treated as local iid variables at inference
    time.

    This function is different from `_log_likelihoods_over_trials` in that it moves the
    iid batch dimension of `x` onto the batch dimension of `theta`. This is needed when
    the likelihood estimator is conditioned on a batch of conditions that are iid with
    the batch of `x`. It avoids the evaluation of the likelihood for every combination
    of `x` and `local_theta`.

    Args:
        x: data with shape `(sample_dim, x_batch_dim, *x_event_shape)`, where sample_dim
            holds the i.i.d. trials and batch_dim holds a batch of xs, e.g., non-iid
            observations.
        global_theta: Batch of parameters `(theta_batch_dim,
            num_parameters)`.
        local_theta: Batch of conditions of shape `(sample_dim, num_local_thetas)`, must
            match x's `sample_dim`.
        estimator: DensityEstimator.
        track_gradients: Whether to track gradients.

    Returns:
        log_likelihood: log likelihood for each x in x_batch_dim, for each theta in
            theta_batch_dim, summed over all iid trials. Shape `(x_batch_dim,
            theta_batch_dim)`.
    """
    assert x.dim() > 2, "x must have shape (sample_dim, batch_dim, *event_shape)."
    assert local_theta.dim() == 2, (
        "condition must have shape (sample_dim, num_conditions)."
    )
    assert global_theta.dim() == 2, "theta must have shape (batch_dim, num_parameters)."
    num_trials, num_xs = x.shape[:2]
    num_thetas = global_theta.shape[0]
    assert local_theta.shape[0] == num_trials, (
        "Condition batch size must match the number of iid trials in x."
    )
    if num_xs > 1:
        raise NotImplementedError(
            "Batched sampling for multiple `x` is not supported for iid conditions."
        )

    # move the iid batch dimension onto the batch dimension of theta and repeat it there
    x_repeated = torch.transpose(x, 0, 1).repeat_interleave(num_thetas, dim=1)

    # construct theta and condition to cover all trial-theta combinations
    theta_with_condition = torch.cat(
        [
            global_theta.repeat(num_trials, 1),  # repeat ABAB
            local_theta.repeat_interleave(num_thetas, dim=0),  # repeat AABB
        ],
        dim=-1,
    )

    with torch.set_grad_enabled(track_gradients):
        # Calculate likelihood in one batch. Returns (1, num_trials * num_theta)
        log_likelihood_trial_batch = estimator.log_prob(
            x_repeated, condition=theta_with_condition
        )
        # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
        log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
            num_xs, num_trials, num_thetas
        ).sum(1)

    # remove xs batch dimension
    return log_likelihood_trial_sum.squeeze(0)


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

    warnings.warn(
        "This function is deprecated and will be removed in a future release. Use "
        "`likelihood_estimator_based_potential` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

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

        warnings.warn(
            "This function is deprecated and will be removed in a future release. Use "
            "`LikelihoodBasedPotential` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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
            log_likelihood_trial_batch = self.likelihood_estimator.log_prob(
                input=x,
                condition=theta.to(self.device),
            )
            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
                self.x_o.shape[0], -1
            ).sum(0)

        return log_likelihood_trial_sum + prior_log_prob
