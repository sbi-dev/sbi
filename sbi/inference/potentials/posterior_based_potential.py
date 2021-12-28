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
from sbi.utils.sbiutils import within_support, match_theta_and_x_batch_shapes
from sbi.utils.torchutils import atleast_2d, ensure_theta_batched


def posterior_potential(
    posterior_model: TorchModule,
    prior: Any,
    xo: Tensor,
) -> Tuple[Callable, torch_tf.Transform]:
    r"""
    Build posterior from the neural density estimator.

    This is to be used with SNPE. It is not used in the standard case in which the
    posterior-net of SNPE is used to sample the posterior. It is only used when one
    wants to sample the SNPE posterior with VI, MCMC, or standard rejection sampling.

    Args:
        density_estimator: The density estimator that the posterior is based on.
            If `None`, use the latest neural density estimator that was trained.
        sample_with: Method to use for sampling from the posterior. Must be one of
            [`mcmc` | `rejection`].

    Returns:
        Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
        (the returned log-probability is unnormalized).
    """

    device = str(next(posterior_model.parameters()).device)

    potential_fn = _build_potential_fn(prior, posterior_model, xo, device=device)
    potential_tf = mcmc_transform(prior, device=device)

    return potential_fn, potential_tf


def _build_potential_fn(prior, posterior_model: nn.Module, xo: Tensor, device: str):
    # TODO Train exited here, entered after sampling?
    posterior_model.eval()

    def direct_potential(theta: Tensor, track_gradients: bool = True):

        theta = ensure_theta_batched(torch.as_tensor(theta))
        theta, x_repeated = match_theta_and_x_batch_shapes(theta, xo)
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

    return direct_potential
