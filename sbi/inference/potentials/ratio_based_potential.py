# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from sbi.inference.potentials.base_potential import BasePotential
from sbi.types import TorchTransform
from sbi.utils import mcmc_transform
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes
from sbi.utils.torchutils import atleast_2d


def ratio_estimator_based_potential(
    ratio_estimator: nn.Module,
    prior: Distribution,
    x_o: Optional[Tensor],
) -> Tuple[Callable, TorchTransform]:
    r"""Returns the potential for ratio-based methods.

    It also returns a transformation that can be used to transform the potential into
    unconstrained space.

    Args:
        ratio_estimator: The neural network modelling likelihood-to-evidence ratio.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the likelihood-to-evidence ratio.

    Returns:
        The potential function and a transformation that maps
        to unconstrained space.
    """

    device = str(next(ratio_estimator.parameters()).device)

    potential_fn = RatioBasedPotential(ratio_estimator, prior, x_o, device=device)
    theta_transform = mcmc_transform(prior, device=device)

    return potential_fn, theta_transform


class RatioBasedPotential(BasePotential):
    allow_iid_x = True  # type: ignore

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
            track_gradients: Whether to track the gradients.

        Returns:
            The potential.
        """

        # Calculate likelihood over trials and in one batch.
        log_likelihood_trial_sum = _log_ratios_over_trials(
            x=self.x_o,
            theta=theta.to(self.device),
            net=self.ratio_estimator,
            track_gradients=track_gradients,
        )

        # Move to cpu for comparison with prior.
        return log_likelihood_trial_sum + self.prior.log_prob(theta)


def _log_ratios_over_trials(
    x: Tensor, theta: Tensor, net: nn.Module, track_gradients: bool = False
) -> Tensor:
    r"""Return log ratios summed over iid trials of `x`.

    Note: `x` can be a batch with batch size larger 1. Batches in x are assumed to
    be iid trials, i.e., data generated based on the same paramters / experimental
    conditions.

    Repeats `x` and $\theta$ to cover all their combinations of batch entries.

    Args:
        x: batch of iid data.
        theta: batch of parameters
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
