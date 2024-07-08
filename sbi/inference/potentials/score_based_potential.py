# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.vf_estimators import VectorFieldEstimator
from sbi.types import TorchTransform
from sbi.utils import mcmc_transform

# NOTE: rough draft for score_based potentials!


def score_estimator_based_potential(
    score_estimator: VectorFieldEstimator,
    prior: Distribution,
    x_o: Optional[Tensor],
    diffusion_time: float,
    enable_transform: bool = True,
) -> Tuple[Callable, TorchTransform]:
    device = str(next(score_estimator.parameters()).device)

    potential_fn = ScoreBasedPotential(
        score_estimator, prior, diffusion_time, x_o, device=device
    )

    # TODO Disabling transform for now, need to think how this affects the score
    theta_transform = mcmc_transform(prior, device=device, enable_transform=False)

    return potential_fn, theta_transform


class ScoreBasedPotential(BasePotential):
    allow_iid_x = True  # type: ignore

    def __init__(
        self,
        score_estimator: VectorFieldEstimator,
        prior: Distribution,
        diffusion_time: float,
        x_o: Optional[Tensor],
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
        self.diffusion_time = diffusion_time
        self.diffusion_length = diffusion_length

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        r"""Returns the potential function for score-based methods.

        Args:
            theta: The parameters at which to evaluate the potential.
            track_gradients: Whether to track gradients.

        Returns:
            The potential function.
        """

        # TODO check if x_o consists of multiple observations
        # Is this a robust check that corresponds to the ones in NLE?

        if self.x_o.shape[0] > 1:
            assert self.diffusion_length is not None
            score_trial_sum = _bridge(
                x=self.x_o,
                theta=theta.to(self.device),
                estimator=self.score_estimator,
                diffusion_time=self.diffusion_time,
                prior=self.prior,
                diffusion_lentgh=self.diffusion_length,
                track_gradients=track_gradients,
            )
        else:
            score_trial_sum = self.score_estimator.forward(
                self.x_o, self.diffusion_time, condition=theta
            )

        return score_trial_sum


def _bridge(
    x: Tensor,
    theta: Tensor,
    estimator: VectorFieldEstimator,
    diffusion_time: float,
    prior: Distribution,
    diffusion_lentgh: Optional[float],
    track_gradients: bool = False,
):
    r"""
    Note: `x` can be a batch with batch size larger 1. Batches in `x` are assumed
    to be iid trials, i.e., data generated based on the same paramters /
    experimental conditions.
    """

    # NOTE: Should conform to
    # (batch_size1, 1, input_size) + (batch_size2, *condition_shape)
    # unsqueeze to ensure that the x-batch dimension is the first dimension for the
    # broadcasting of the density estimator.

    x = torch.as_tensor(x).reshape(-1, x.shape[-1]).unsqueeze(1)
    num_obs = x.shape[0]
    assert (
        next(estimator.parameters()).device == x.device and x.device == theta.device
    ), f"""device mismatch: estimator, x, theta: \
        {next(estimator.parameters()).device}, {x.device},
        {theta.device}."""

    # Calculate likelihood in one batch.
    with torch.set_grad_enabled(track_gradients):
        score_trial_batch = estimator.forward(x, diffusion_time, condition=theta)
        # Reshape to (-1, theta_batch_size), sum over trial-log likelihoods.
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

    # TODO check if this has the grad property else use torch autograd
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
