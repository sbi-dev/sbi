from typing import Callable, Dict, Union

import numpy as np
import torch
import torch.distributions.transforms as torch_tf
from torch import Tensor

from sbi.utils.torchutils import ensure_theta_batched


def transformed_potential(
    theta: Union[Tensor, np.ndarray],
    potential_fn: Callable,
    theta_transform: torch_tf.Transform,
    device: str,
    track_gradients: bool = False,
) -> Tensor:
    """Return potential after a transformation by adding the log-abs-determinant.

    In addition, this function takes care of moving the parameters to the correct
    device.

    Args:
        theta:  Parameters $\theta$ in transformed space.
        potential_fn: Potential function.
        theta_transform: Transformation applied before evaluating the `potential_fn`
        device: The device to which to move the parameters before evaluation.
        track_gradients: Whether to track the gradients of the `potential_fn`
            evaluation.
    """

    # Device is the same for net and prior.
    transformed_theta = ensure_theta_batched(
        torch.as_tensor(theta, dtype=torch.float32)
    ).to(device)
    # Transform `theta` from transformed (i.e. unconstrained) to untransformed
    # space.
    theta = theta_transform.inv(transformed_theta)  # type: ignore
    log_abs_det = theta_transform.log_abs_det_jacobian(theta, transformed_theta)

    posterior_potential = potential_fn(theta, track_gradients=track_gradients)
    posterior_potential_transformed = posterior_potential - log_abs_det
    return posterior_potential_transformed


def pyro_potential_wrapper(theta: Dict[str, Tensor], potential: Callable) -> Callable:
    r"""Evaluate pyro-based `theta` under the negative `potential`.

        Args:
        theta: Parameters $\theta$. The tensor's shape will be
            (1, shape_of_single_theta) if running a single chain or just
            (shape_of_single_theta) for multiple chains.
        potential: Potential which to evaluate.

    Returns:
        The negative potential $-[\log r(x_o, \theta) + \log p(\theta)]$.
    """

    theta_tensor = next(iter(theta.values()))

    # Note the minus to match the pyro potential function requirements.
    return -potential(theta_tensor)
