# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.density_estimators import DensityEstimator
from sbi.neural_nets.mnle import MixedDensityEstimator
from sbi.sbi_types import TorchTransform
from sbi.utils import mcmc_transform


def likelihood_estimator_based_potential(
    likelihood_estimator: DensityEstimator,
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
    allow_iid_x = True  # type: ignore

    def __init__(
        self,
        likelihood_estimator: DensityEstimator,
        prior: Distribution,
        x_o: Optional[Tensor],
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

        # Calculate likelihood over trials and in one batch.
        log_likelihood_trial_sum = _log_likelihoods_over_trials(
            x=self.x_o,
            theta=theta.to(self.device),
            estimator=self.likelihood_estimator,
            track_gradients=track_gradients,
        )

        return log_likelihood_trial_sum + self.prior.log_prob(theta)  # type: ignore


def _log_likelihoods_over_trials(
    x: Tensor, theta: Tensor, estimator: DensityEstimator, track_gradients: bool = False
) -> Tensor:
    r"""Return log likelihoods summed over iid trials of `x`.

    Note: `x` can be a batch with batch size larger 1. Batches in `x` are assumed
    to be iid trials, i.e., data generated based on the same paramters /
    experimental conditions.

    Repeats `x` and $\theta$ to cover all their combinations of batch entries.

    Args:
        x: batch of iid data.
        theta: batch of parameters.
        estimator: DensityEstimator.
        track_gradients: Whether to track gradients.

    Returns:
        log_likelihood_trial_sum: log likelihood for each parameter, summed over all
            batch entries (iid trials) in `x`.
    """
    # unsqueeze to ensure that the x-batch dimension is the first dimension for the
    # broadcasting of the density estimator.
    x = torch.as_tensor(x).reshape(-1, x.shape[-1]).unsqueeze(1)
    assert (
        next(estimator.parameters()).device == x.device and x.device == theta.device
    ), f"""device mismatch: estimator, x, theta: \
        {next(estimator.parameters()).device}, {x.device},
        {theta.device}."""

    # Calculate likelihood in one batch.
    with torch.set_grad_enabled(track_gradients):
        log_likelihood_trial_batch = estimator.log_prob(x, condition=theta)
        # Reshape to (-1, theta_batch_size), sum over trial-log likelihoods.
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
        likelihood_estimator: MixedDensityEstimator,  # type: ignore TODO fix pyright
        prior: Distribution,
        x_o: Optional[Tensor],
        device: str = "cpu",
    ):
        # TODO Fix pyright issue by making MixedDensityEstimator a subclass
        # of DensityEstimator
        super().__init__(likelihood_estimator, prior, x_o, device)  # type: ignore

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        # Calculate likelihood in one batch.
        with torch.set_grad_enabled(track_gradients):
            # Call the specific log prob method of the mixed likelihood estimator as
            # this optimizes the evaluation of the discrete data part.
            # TODO: how to fix pyright issues?
            log_likelihood_trial_batch = self.likelihood_estimator.log_prob_iid(
                x=self.x_o,
                context=theta.to(self.device),
            )  # type: ignore
            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
                self.x_o.shape[0], -1
            ).sum(0)

        return log_likelihood_trial_sum + self.prior.log_prob(theta)  # type: ignore
