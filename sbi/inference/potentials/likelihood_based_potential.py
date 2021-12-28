# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils.torchutils import atleast_2d
from sbi.utils import mcmc_transform
from sbi.types import TorchModule
import torch.distributions.transforms as torch_tf
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes


def likelihood_potential(
    likelihood_model: TorchModule,
    prior: Any,
    xo: Tensor,
) -> Tuple[Callable, torch_tf.Transform]:
    r"""
    Build posterior from the neural density estimator.

    SNLE trains a neural network to approximate the likelihood $p(x|\theta)$. The
    `LikelihoodBasedPosterior` class wraps the trained network such that one can
    directly evaluate the unnormalized posterior log probability
    $p(\theta|x) \propto p(x|\theta) \cdot p(\theta)$ and draw samples from the
    posterior with MCMC.

    Args:
        density_estimator: The density estimator that the posterior is based on.
            If `None`, use the latest neural density estimator that was trained.
        sample_with: Method to use for sampling from the posterior. Must be one of
            [`mcmc` | `rejection`].

    Returns:
        Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
        (the returned log-probability is unnormalized).
    """

    device = str(next(likelihood_model.parameters()).device)

    potential_fn = _build_potential_fn(prior, likelihood_model, xo, device=device)
    potential_tf = mcmc_transform(prior, device=device)

    return potential_fn, potential_tf


def _build_potential_fn(prior, likelihood_nn: nn.Module, xo: Tensor, device: str):
    # TODO Train exited here, entered after sampling?
    likelihood_nn.eval()

    def likelihood_potential(theta: Tensor, track_gradients: bool = True):
        # Calculate likelihood over trials and in one batch.
        log_likelihood_trial_sum = _log_likelihoods_over_trials(
            x=xo.to(device),
            theta=theta.to(device),
            net=likelihood_nn,
            track_gradients=track_gradients,
        )

        # Move to cpu for comparison with prior.
        return log_likelihood_trial_sum + prior.log_prob(theta)

    return likelihood_potential


def _log_likelihoods_over_trials(
    x: Tensor, theta: Tensor, net: Any, track_gradients: bool = False
) -> Tensor:
    r"""Return log likelihoods summed over iid trials of `x`.

    Note: `x` can be a batch with batch size larger 1. Batches in `x` are assumed
    to be iid trials, i.e., data generated based on the same paramters /
    experimental conditions.

    Repeats `x` and $\theta$ to cover all their combinations of batch entries.

    Args:
        x: batch of iid data.
        theta: batch of parameters
        net: neural net with .log_prob()
        track_gradients: Whether to track gradients.

    Returns:
        log_likelihood_trial_sum: log likelihood for each parameter, summed over all
            batch entries (iid trials) in `x`.
    """

    # Repeat `x` in case of evaluation on multiple `theta`. This is needed below in
    # when calling nflows in order to have matching shapes of theta and context x
    # at neural network evaluation time.
    theta_repeated, x_repeated = match_theta_and_x_batch_shapes(
        theta=atleast_2d(theta), x=atleast_2d(x)
    )
    assert (
        x_repeated.shape[0] == theta_repeated.shape[0]
    ), "x and theta must match in batch shape."
    assert (
        next(net.parameters()).device == x.device and x.device == theta.device
    ), f"device mismatch: net, x, theta: {next(net.parameters()).device}, {x.device}, {theta.device}."

    # Calculate likelihood in one batch.
    with torch.set_grad_enabled(track_gradients):
        log_likelihood_trial_batch = net.log_prob(x_repeated, theta_repeated)
        # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
        log_likelihood_trial_sum = log_likelihood_trial_batch.reshape(
            x.shape[0], -1
        ).sum(0)

    return log_likelihood_trial_sum
