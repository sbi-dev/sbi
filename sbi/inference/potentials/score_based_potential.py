# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.vf_estimators.score_estimator import ScoreEstimator
from sbi.sbi_types import TorchTransform
from sbi.utils import mcmc_transform

# NOTE: rough draft for score_based potentials!


def score_estimator_based_potential(
    score_estimator: ScoreEstimator,
    prior: Distribution,
    x_o: Optional[Tensor],
    x_o_shape: Optional[Tuple[int, ...]] = None,
    diffusion_length: Optional[int] = None,
    enable_transform: bool = True,
) -> Tuple[Callable, TorchTransform]:
    device = str(next(score_estimator.parameters()).device)

    potential_fn = ScoreBasedPotential(
        score_estimator, prior, x_o, x_o_shape, diffusion_length, device=device
    )

    # TODO Disabling transform for now, need to think how this affects the score
    theta_transform = mcmc_transform(prior, device=device, enable_transform=False)

    return potential_fn, theta_transform


class ScoreBasedPotential(BasePotential):
    allow_iid_x = True  # type: ignore

    def __init__(
        self,
        score_estimator: ScoreEstimator,
        prior: Distribution,
        x_o: Optional[Tensor],
        x_o_shape: Optional[Tuple[int, ...]] = None,
        diffusion_length: Optional[float] = None,
        device: str = "cpu",
    ):
        r"""Returns the score function for score-based methods.

        Args:
            score_estimator: The neural network modelling the score.
            prior: The prior distribution.
            x_o: The observed data at which to evaluate the posterior.
            device: The device on which to evaluate the potential.
        """

        super().__init__(prior, x_o, device=device)

        self.score_estimator = score_estimator
        self.diffusion_length = diffusion_length
        self.x_o_shape = x_o_shape

    def __call__(
        self, theta: Tensor, diffusion_time: Tensor, track_gradients: bool = True
    ) -> Tensor:
        r"""Returns the potential function for score-based methods.

        Args:
            theta: The parameters at which to evaluate the potential.
            track_gradients: Whether to track gradients.

        Returns:
            The potential function.
        """
        # theta shape (batch, (iid), event_dim)

        # could also have a more lenient check here (just the length)
        # this assumes that there is only one batch dimension
        # is this assumption always valid?
        if self.x_o.shape[1:] == self.x_o_shape:
            score_trial_sum = self.score_estimator.forward(
                input=theta, condition=self.x_o, times=diffusion_time
            )
        else:
            assert self.diffusion_length is not None
            # Diffusion length is required for Geffner bridge
            score_trial_sum = _bridge(
                x=self.x_o,
                theta=theta.to(self.device),
                estimator=self.score_estimator,
                diffusion_time=diffusion_time,
                prior=self.prior,
                diffusion_lentgh=self.diffusion_length,
                track_gradients=track_gradients,
            )

        return score_trial_sum


def _bridge(
    x: Tensor,
    x_shape: Tuple[int, ...],
    theta: Tensor,
    estimator: ScoreEstimator,
    diffusion_time: Tensor,
    prior: Distribution,
    diffusion_lentgh: Optional[float],
    track_gradients: bool = False,
):
    r"""
    Note: `x` can be a batch with batch size larger 1. Batches in `x` are assumed
    to be iid trials, i.e., data generated based on the same paramters /
    experimental conditions.
    """

    assert (
        next(estimator.parameters()).device == x.device and x.device == theta.device
    ), f"""device mismatch: estimator, x, theta: \
        {next(estimator.parameters()).device}, {x.device},
        {theta.device}."""

    # Get number of observations which are left from event_shape if they exist.
    num_obs = x.shape[-len(x_shape) - 1]

    # Calculate likelihood in one batch.
    # TODO we need to figure out the axis where we sum or manually reshape to a
    # compatible input for the score estimtor and then reshape and summing after
    # obtaining the score.
    with torch.set_grad_enabled(track_gradients):
        score_trial_batch = estimator.forward(
            input=theta, condition=x, times=diffusion_time
        )

        score_trial_sum = score_trial_batch.sum(0)

    return score_trial_sum + _get_prior_contribution(
        diffusion_time, prior, theta, diffusion_lentgh, num_obs
    )


def _get_prior_contribution(
    diffusion_time: float,
    prior: Distribution,
    theta: Tensor,
    diffusion_length: Optional[float],
    num_obs: int,
):
    # This method can be used to add several different bridges (Sharrock, Linhart)
    # to obtain the posterior for multiple IID observations.
    # For now, it only implements the approach by Geffner et al.

    # TODO Check if prior has the grad property else use torch autograd
    # For now just use autograd.

    log_prob_theta = prior.log_prob(theta)

    grad_log_prob_theta = torch.autograd.grad(
        log_prob_theta,
        theta,
        grad_outputs=torch.ones_like(log_prob_theta),
        create_graph=True,
    )[0]

    return (
        (1 - num_obs) * (1.0 - diffusion_time / diffusion_length)
    ) * grad_log_prob_theta
