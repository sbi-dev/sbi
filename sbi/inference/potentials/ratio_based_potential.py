# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributions.transforms as torch_tf
from torch import Tensor, nn

from sbi.utils import mcmc_transform
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes
from sbi.utils.torchutils import atleast_2d


def ratio_potential(
    ratio_model: nn.Module,
    prior: Any,
    x_o: Tensor,
) -> Tuple[Callable, torch_tf.Transform]:
    r"""
    Returns the potential for ratio-based methods.

    It also returns a transformation that can be used to transform the potential into
    unconstrained space.

    Args:
        ratio_model: The neural network modelling likelihood-to-evidence ratio.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the likelihood-to-evidence ratio.

    Returns:
        The potential function and a transformation that maps
        to unconstrained space.
    """

    device = str(next(ratio_model.parameters()).device)

    potential_fn = _build_potential_fn(prior, ratio_model, x_o, device=device)
    theta_transform = mcmc_transform(prior, device=device)

    return potential_fn, theta_transform


def _build_potential_fn(prior, likelihood_nn: nn.Module, x_o: Tensor, device: str):
    r"""
    Returns the potential for ratio-based methods.

    Args:
        ratio_model: The neural network modelling likelihood-to-evidence ratio.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the likelihood-to-evidence ratio.

    Returns:
        The potential function.
    """
    likelihood_nn.eval()

    def ratio_potential(theta: Tensor, track_gradients: bool = True):
        # Calculate likelihood over trials and in one batch.
        log_likelihood_trial_sum = _log_ratios_over_trials(
            x=x_o.to(device),
            theta=theta.to(device),
            net=likelihood_nn,
            track_gradients=track_gradients,
        )

        # Move to cpu for comparison with prior.
        return log_likelihood_trial_sum + prior.log_prob(theta)

    return ratio_potential


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
    ), f"device mismatch: net, x, theta: {next(net.parameters()).device}, {x.device}, {theta.device}."

    # Calculate ratios in one batch.
    with torch.set_grad_enabled(track_gradients):
        log_ratio_trial_batch = net([theta_repeated, x_repeated])
        # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
        log_ratio_trial_sum = log_ratio_trial_batch.reshape(x.shape[0], -1).sum(0)

    return log_ratio_trial_sum
