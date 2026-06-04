# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Protocol definitions for stateless potential functions.

This module provides the core protocol (Layer 1) for potential functions,
enabling stateless, composable potential evaluation.
"""

from typing import Protocol

import torch
from torch import Tensor


class PotentialFunction(Protocol):
    """Stateless callable protocol for potential functions.

    A PotentialFunction takes parameters theta and observed data x, and returns
    the unnormalized log-probability log p(theta, x). This is the core interface
    that all potential functions must satisfy.

    The protocol is stateless - all inputs (theta, x) are passed as arguments,
    with no hidden state dependencies. This enables:
    - Easy testing (same inputs -> same outputs)
    - Simple serialization
    - Composable inference workflows
    - Integration with external samplers (NumPyro, PyMC, custom PyTorch)

    Example:
        >>> def my_potential(theta: Tensor, x: Tensor) -> Tensor:
        ...     ll = density_estimator.log_prob(x, context=theta)
        ...     return ll + prior.log_prob(theta)
        >>> # Satisfies PotentialFunction protocol
    """

    def __call__(self, theta: Tensor, x: Tensor) -> Tensor:
        """Evaluate the potential function at theta given observed data x.

        Args:
            theta: Parameter samples with shape (batch_size, *param_shape).
            x: Observed data with shape (obs_batch, *obs_shape).

        Returns:
            Log-probability with shape (batch_size,) or (obs_batch, batch_size)
            depending on the potential implementation.
        """
        ...


class PotentialFunctionWithDevice(Protocol):
    """Extended protocol that reports device requirements.

    This protocol adds a device property to enable automatic device validation
    and inference in bind_observation.
    """

    def __call__(self, theta: Tensor, x: Tensor) -> Tensor:
        """Evaluate the potential function at theta given observed data x."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the expected device for inputs.

        This property allows bind_observation to validate device compatibility
        and provide clear error messages for device mismatches.
        """
        ...


def validate_potential(potential: PotentialFunction) -> bool:
    """Validate that a potential function satisfies the protocol.

    This helper checks that the potential is callable and has the correct
    signature. It performs runtime validation since Protocol typing
    doesn't catch errors at definition time.

    Args:
        potential: The potential function to validate.

    Returns:
        True if the potential satisfies the protocol.

    Raises:
        TypeError: If potential doesn't satisfy the protocol.
        ValueError: If potential is callable but has wrong signature.
    """
    if not callable(potential):
        raise TypeError(
            "Potential must be callable. "
            "Expected a function or object with __call__ method."
        )

    import inspect

    sig = inspect.signature(potential.__call__)
    params = list(sig.parameters.keys())

    if len(params) < 2:
        raise ValueError(
            f"Potential __call__ must have at least 2 parameters (theta, x), "
            f"got signature: {sig}"
        )

    return True
