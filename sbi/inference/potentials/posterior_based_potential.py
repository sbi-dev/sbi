# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributions.transforms as torch_tf
from torch import Tensor, nn

from sbi.utils import mcmc_transform
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes, within_support
from sbi.utils.torchutils import atleast_2d, ensure_theta_batched


def posterior_potential(
    posterior_model: nn.Module,
    prior: Any,
    x_o: Tensor,
) -> Tuple[Callable, torch_tf.Transform]:
    r"""
    Returns the potential for posterior-based methods.

    It also returns a transformation that can be used to transform the potential into
    unconstrained space.

    The potential is the same as the log-probability of the `posterior_model`, but it
    is set to $-\inf$ outside of the prior bounds.

    Args:
        posterior_model: The neural network modelling the posterior.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the posterior.

    Returns:
        The potential function and a transformation that maps
        to unconstrained space.
    """

    device = str(next(posterior_model.parameters()).device)

    potential_fn = _build_potential_fn(posterior_model, prior, x_o, device=device)
    theta_transform = mcmc_transform(prior, device=device)

    return potential_fn, theta_transform


def _build_potential_fn(
    posterior_model: nn.Module, prior: Any, x_o: Tensor, device: str
) -> Callable:
    r"""
    Returns the potential for posterior-based methods.

    The potential is the same as the log-probability of the `posterior_model`, but it
    is set to $-\inf$ outside of the prior bounds.

    Args:
        posterior_model: The neural network modelling the posterior.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the posterior.

    Returns:
        The potential function.
    """
    posterior_model.eval()

    def posterior_potential(theta: Tensor, track_gradients: bool = True):

        theta = ensure_theta_batched(torch.as_tensor(theta))
        theta, x_repeated = match_theta_and_x_batch_shapes(theta, x_o)
        theta, x_repeated = theta.to(device), x_repeated.to(device)

        with torch.set_grad_enabled(track_gradients):
            posterior_log_prob = posterior_model.log_prob(theta, context=x_repeated)

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(prior, theta)

            posterior_log_prob = torch.where(
                in_prior_support,
                posterior_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=device),
            )
        return posterior_log_prob

    return posterior_potential
