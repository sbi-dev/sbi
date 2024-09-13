# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.sbi_types import TorchTransform
from sbi.utils.sbiutils import mcmc_transform, within_support
from sbi.utils.torchutils import ensure_theta_batched


def posterior_estimator_based_potential(
    posterior_estimator: ConditionalDensityEstimator,
    prior: Distribution,
    x_o: Optional[Tensor],
    enable_transform: bool = True,
) -> Tuple[PosteriorBasedPotential, TorchTransform]:
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
        posterior_estimator: ConditionalDensityEstimator,
        prior: Distribution,
        x_o: Optional[Tensor] = None,
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

    def set_x(self, x_o: Optional[Tensor], x_is_iid: Optional[bool] = False):
        """
        Check the shape of the observed data and, if valid, set it.
        For posterior-based methods, `x_o` is not allowed to be iid, as we assume that
        iid `x` is handled by a Permutation Invariant embedding net.
        """
        if x_is_iid and x_o is not None and x_o.shape[0] > 1:
            raise NotImplementedError(
                "For NPE, iid `x` must be handled by a permutation invariant embedding "
                "net. Therefore, the iid dimension of `x` is added to the event "
                "dimension of `x`. Please set `x_is_iid=False`."
            )
        else:
            super().set_x(x_o, x_is_iid=False)

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

        with torch.set_grad_enabled(track_gradients):
            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta)
            x = reshape_to_batch_event(
                self.x_o, event_shape=self.posterior_estimator.condition_shape
            )
            theta = ensure_theta_batched(torch.as_tensor(theta)).to(self.device)
            theta_batch_size = theta.shape[0]
            x_batch_size = x.shape[0]

            assert (
                theta_batch_size == x_batch_size or x_batch_size == 1
            ), f"Batch size mismatch: {theta_batch_size} and {x_batch_size}.\
                When performing batched sampling for multiple `x`, the batch size of\
                `theta` must match the batch size of `x`."

            if x_batch_size == 1:
                # If a single `x` is passed (i.e. batchsize==1), we squeeze
                # the batch dimension of the log-prob with `.squeeze(dim=1)`.
                theta = reshape_to_sample_batch_event(
                    theta, event_shape=theta.shape[1:], leading_is_sample=True
                )

                posterior_log_prob = self.posterior_estimator.log_prob(
                    theta, condition=x
                )
                posterior_log_prob = posterior_log_prob.squeeze(1)
            else:
                # If multiple `x` are passed, we return the log-probs for each (x,theta)
                # pair, and do not squeeze the batch dimension.
                theta = theta.unsqueeze(0)
                posterior_log_prob = self.posterior_estimator.log_prob(
                    theta, condition=x
                )
            posterior_log_prob = torch.where(
                in_prior_support,
                posterior_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self.device),
            )

        return posterior_log_prob
