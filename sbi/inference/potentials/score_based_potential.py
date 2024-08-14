# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.potentials.base_potential import BasePotentialGradient
from sbi.neural_nets.estimators.score_estimator import ConditionalScoreEstimator
from sbi.sbi_types import TorchTransform
from sbi.utils import mcmc_transform


def score_estimator_based_potential_gradient(
    score_estimator: ConditionalScoreEstimator,
    prior: Optional[Distribution],
    x_o: Optional[Tensor],
    enable_transform: bool = False,
) -> Tuple["PosteriorScoreBasedPotentialGradient", TorchTransform]:
    r"""Returns the potential function gradient for score estimators.

    Args:
        score_estimator: The neural network modelling the score.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the score.
        enable_transform: Whether to enable transforms. Not supported yet.

    """
    device = str(next(score_estimator.parameters()).device)

    potential_fn = PosteriorScoreBasedPotentialGradient(
        score_estimator, prior, x_o, device=device
    )

    # TODO Add issue
    assert (
        enable_transform is False
    ), "Transforms are not yet supported for score estimators."

    if prior is not None:
        theta_transform = mcmc_transform(
            prior, device=device, enable_transform=enable_transform
        )
    else:
        theta_transform = torch.distributions.transforms.identity_transform

    return potential_fn, theta_transform


class PosteriorScoreBasedPotentialGradient(BasePotentialGradient):
    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Optional[Distribution],
        x_o: Optional[Tensor],
        iid_method: str = "iid_bridge",
        device: str = "cpu",
    ):
        r"""Returns the score function for score-based methods.

        Args:
            score_estimator: The neural network modelling the score.
            prior: The prior distribution.
            x_o: The observed data at which to evaluate the posterior.
            x_o_shape: The shape of the observed data.
            iid_method: Which method to use for computing the score. Currently, only
                `iid_bridge` as proposed in Geffner et al. is implemented.
            device: The device on which to evaluate the potential.
        """

        super().__init__(prior, x_o, device=device)
        self.score_estimator = score_estimator
        self.score_estimator.eval()
        self.iid_method = iid_method

    def __call__(
        self, theta: Tensor, time: Tensor, track_gradients: bool = True
    ) -> Tensor:
        r"""Returns the potential function gradient for score-based methods.

        Args:
            theta: The parameters at which to evaluate the potential.
            diffusion_time: The diffusion time.
            track_gradients: Whether to track gradients.

        Returns:
            The potential function.
        """
        if self._x_o is None:
            raise ValueError(
                "No observed data x_o is available. Please reinitialize \
                the potential or manually set self._x_o."
            )

        with torch.set_grad_enabled(track_gradients):
            if not self.x_is_iid or self._x_o.shape[0] == 1:
                score = self.score_estimator.forward(
                    input=theta, condition=self.x_o, time=time
                )
            else:
                if self.prior is None:
                    raise ValueError(
                        "Prior must be provided when interpreting the data as IID."
                    )

                if self.iid_method == "iid_bridge":
                    score = _iid_bridge(
                        theta=theta,
                        xos=self.x_o,
                        time=time,
                        score_estimator=self.score_estimator,
                        prior=self.prior,
                    )
                else:
                    raise NotImplementedError(
                        f"Method {self.iid_method} not implemented."
                    )

        return score


def _iid_bridge(
    theta: Tensor,
    xos: Tensor,
    time: Tensor,
    score_estimator: ConditionalScoreEstimator,
    prior: Distribution,
):
    r"""
    Returns the score-based potential for multiple IID observations. This can require a
    special solver to obtain the correct tall posterior.

    Args:
        input: The parameter values at which to evaluate the potential.
        condition: The observed data at which to evaluate the potential.
        time: The diffusion time.
        score_estimator: The neural network modelling the score.
        prior: The prior distribution.
    """

    assert (
        next(score_estimator.parameters()).device == xos.device
        and xos.device == theta.device
    ), f"""device mismatch: estimator, x, theta: \
        {next(score_estimator.parameters()).device}, {xos.device},
        {theta.device}."""

    # Get number of observations which are left from event_shape if they exist.
    condition_shape = score_estimator.condition_shape
    num_obs = xos.shape[-len(condition_shape) - 1]

    # Calculate likelihood in one batch.

    score_trial_batch = score_estimator.forward(input=theta, condition=xos, time=time)

    score_trial_sum = score_trial_batch.sum(0)

    return score_trial_sum + _get_prior_contribution(time, prior, theta, num_obs)


def _get_prior_contribution(
    diffusion_time: Tensor,
    prior: Distribution,
    theta: Tensor,
    num_obs: int,
):
    r"""
    Returns the prior contribution for multiple IID observations.

    Args:
        diffusion_time: The diffusion time.
        prior: The prior distribution.
        theta: The parameter values at which to evaluate the prior contribution.
        num_obs: The number of independent observations.
    """
    # This method can be used to add several different bridges
    # to obtain the posterior for multiple IID observations.
    # For now, it only implements the approach by Geffner et al.

    # TODO Check if prior has the grad property else use torch autograd.
    # For now just use autograd.

    log_prob_theta = prior.log_prob(theta)

    grad_log_prob_theta = torch.autograd.grad(
        log_prob_theta,
        theta,
        grad_outputs=torch.ones_like(log_prob_theta),
        create_graph=True,
    )[0]

    return ((1 - num_obs) * (1.0 - diffusion_time)) * grad_log_prob_theta
