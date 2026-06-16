# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Binding utilities for creating sampler-ready potential functions.

This module provides Layer 2 utilities for binding observations to stateless
potential functions, making them ready for MCMC/VI samplers.
"""

from typing import Union

import torch
from torch import Tensor

from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference.potentials.protocol import PotentialFunction


def bind_observation(
    potential: PotentialFunction,
    x_o: Tensor,
    sum_iid: bool = True,
) -> "BoundPotential":
    """Create a sampler-compatible theta -> log_prob function.

    This utility binds observed data to a stateless potential function, producing
    a callable ready for use with MCMC, VI, or rejection sampling algorithms.

    The function infers the device from x_o and validates compatibility with
    the potential's device requirements (if the potential has a device attribute).

    Args:
        potential: Stateless (theta, x) -> log_prob function satisfying the
            PotentialFunction protocol.
        x_o: Observed data tensor. Shape should be (obs_batch, *obs_shape) for
            IID observations, or (*obs_shape) for single observation.
        sum_iid: If True, sum log probabilities over the observation batch dimension.
            This is the typical behavior for IID observations where we want to
            compute log p(x_o | theta) = sum_i log p(x_i | theta).

    Returns:
        A BoundPotential instance that takes theta (parameters) and returns the
        log-probability log p(theta | x_o) + log p(theta). The returned object
        is compatible with samplers that expect set_x() method.

    Example:
        >>> # Create stateless potential
        >>> def likelihood_potential(theta: Tensor, x: Tensor) -> Tensor:
        ...     return estimator.log_prob(x, context=theta) + prior.log_prob(theta)
        >>>
        >>> # Bind observation for sampling
        >>> bound_fn = bind_observation(likelihood_potential, x_o, sum_iid=True)
        >>> log_probs = bound_fn(theta_samples)  # Ready for MCMC/VI
    """
    device = x_o.device

    if hasattr(potential, "device"):
        potential_device = potential.device
        if isinstance(potential_device, str):
            potential_device = torch.device(potential_device)
        if potential_device != device:
            raise ValueError(
                f"Device mismatch: x_o is on {device}, but potential expects "
                f"{potential_device}. Use x_o.to('{potential_device}') before "
                f"binding, or ensure the potential was created on the same device "
                f"as x_o."
            )

    x_o = x_o.to(device)

    return BoundPotential(potential, x_o, sum_iid)


class BoundPotential(BasePotential):
    """Wrapper class for binding observation to a potential function.

    This provides an alternative to the functional bind_observation, maintaining
    compatibility with existing code patterns while supporting the new stateless
    protocol internally.
    """

    def __init__(
        self,
        potential: PotentialFunction,
        x_o: Tensor,
        sum_iid: bool = True,
    ):
        self._potential = potential
        self._x_o = x_o.to(x_o.device)
        self._sum_iid = sum_iid
        self._device = x_o.device
        self._x_is_iid = True

    @property
    def device(self) -> torch.device:
        return self._device

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        theta = theta.to(self._device)

        with torch.set_grad_enabled(track_gradients):
            log_prob = self._potential(theta, self._x_o)

            if self._sum_iid and self._x_o.ndim > 1:
                log_prob = log_prob.sum(dim=0)

        return log_prob.reshape(-1) if log_prob.ndim > 0 else log_prob.unsqueeze(0)

    def set_x(self, x_o: Tensor, x_is_iid: bool = True) -> None:
        """Set observation for the potential function.

        For BoundPotential, x_o is already bound in __init__, so this is a no-op
        for backward compatibility with samplers.
        """
        pass

    def return_x_o(self) -> Tensor:
        """Return the bound observation."""
        return self._x_o

    def to(self, device: Union[str, torch.device]) -> "BoundPotential":
        self._device = torch.device(device)
        self._x_o = self._x_o.to(device)
        return self


def bind_observation_class(
    potential: PotentialFunction,
    x_o: Tensor,
    sum_iid: bool = True,
) -> BoundPotential:
    """Create a BoundPotential instance for sampler-compatible usage.

    This is an alternative to bind_observation that returns a class instance
    instead of a simple function. Useful when you need method chaining or
    want to maintain compatibility with code that expects an object.

    Args:
        potential: Stateless (theta, x) -> log_prob function.
        x_o: Observed data tensor.
        sum_iid: If True, sum log probabilities over IID batch dimension.

    Returns:
        BoundPotential instance that can be called like a function.
    """
    return BoundPotential(potential, x_o, sum_iid)
