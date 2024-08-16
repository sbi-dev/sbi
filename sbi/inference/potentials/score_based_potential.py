# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution
from zuko.transforms import FreeFormJacobianTransform

from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.estimators.score_estimator import ConditionalScoreEstimator
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.sbi_types import TorchTransform
from sbi.utils import mcmc_transform
from sbi.utils.sbiutils import within_support
from sbi.utils.torchutils import ensure_theta_batched


def score_estimator_based_potential(
    score_estimator: ConditionalScoreEstimator,
    prior: Optional[Distribution],
    x_o: Optional[Tensor],
    enable_transform: bool = False,
) -> Tuple["PosteriorScoreBasedPotential", TorchTransform]:
    r"""Returns the potential function gradient for score estimators.

    Args:
        score_estimator: The neural network modelling the score.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the score.
        enable_transform: Whether to enable transforms. Not supported yet.
    """
    device = str(next(score_estimator.parameters()).device)

    potential_fn = PosteriorScoreBasedPotential(
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


class PosteriorScoreBasedPotential(BasePotential):
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
        self,
        theta: Tensor,
        track_gradients: bool = True,
        atol: float = 1e-5,
        rtol: float = 1e-6,
        exact: bool = True,
    ):
        """Return the potential via probability flow ODE."""
        theta = ensure_theta_batched(torch.as_tensor(theta))
        theta_density_estimator = reshape_to_sample_batch_event(
            theta, theta.shape[1:], leading_is_sample=True
        )
        x_density_estimator = reshape_to_batch_event(
            self.x_o, event_shape=self.score_estimator.condition_shape
        )
        assert (
            x_density_estimator.shape[0] == 1
        ), ".log_prob() supports only `batchsize == 1`."

        self.score_estimator.eval()

        # Compute the base density
        mean_T = self.score_estimator.mean_T
        std_T = self.score_estimator.std_T
        base_density = torch.distributions.Normal(mean_T, std_T)
        for _ in range(len(self.score_estimator.input_shape)):
            base_density = torch.distributions.Independent(base_density, 1)
        # Build the freeform jacobian transformation by probability flow ODEs
        transform = build_freeform_jacobian_transform(
            self.score_estimator, x_density_estimator, atol=atol, rtol=rtol, exact=exact
        )

        with torch.set_grad_enabled(track_gradients):
            eps_samples, logabsdet = transform.call_and_ladj(  # type: ignore
                theta_density_estimator
            )
            base_log_prob = base_density.log_prob(eps_samples)
            log_probs = base_log_prob + logabsdet
            log_probs = log_probs.squeeze(-1)

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta)

            masked_log_prob = torch.where(
                in_prior_support,
                log_probs,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self.device),
            )
            return masked_log_prob

    def gradient(
        self, theta: Tensor, time: Optional[Tensor] = None, track_gradients: bool = True
    ) -> Tensor:
        r"""Returns the potential function gradient for score-based methods.

        Args:
            theta: The parameters at which to evaluate the potential.
            diffusion_time: The diffusion time. If None, then `T_min` of the
                self.score_estimator is used (i.e. we evaluate the gradient of the
                actual data distribution).
            track_gradients: Whether to track gradients.

        Returns:
            The potential function.
        """
        if time is None:
            time = torch.tensor([self.score_estimator.T_min])

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


def build_freeform_jacobian_transform(
    score_estimator,
    x_o: Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-6,
    exact: bool = True,
):
    # Create a freeform jacobian transformation
    phi = score_estimator.parameters()

    def f(t, x):
        score = score_estimator(input=x, condition=x_o, time=t)
        f = score_estimator.drift_fn(x, t)
        g = score_estimator.diffusion_fn(x, t)
        v = f - 0.5 * g**2 * score
        return v

    transform = FreeFormJacobianTransform(
        f=f,
        t0=score_estimator.T_min,
        t1=score_estimator.T_max,
        phi=phi,
        atol=atol,
        rtol=rtol,
        exact=exact,
    )

    return transform


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
