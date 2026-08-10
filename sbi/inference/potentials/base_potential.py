# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from abc import ABCMeta, abstractmethod
from typing import Optional, Protocol, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.utils.user_input_checks import process_x


class BasePotential(metaclass=ABCMeta):
    def __init__(
        self,
        prior: Optional[Distribution],
        x_o: Optional[Tensor] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize potential function.

        This parent class takes care of setting `x_o`.

        Args:
            prior: Prior distribution.
            x_o: Observed data.
            device: Device on which to evaluate the potential function.
        """
        self.device = device
        self.prior = prior
        if x_o is not None:
            x_o = process_x(x_o).to(self.device)
        self._x_o = x_o
        self._x_is_iid = True

    @abstractmethod
    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        raise NotImplementedError

    def gradient(
        self, theta: Tensor, time: Optional[Tensor] = None, track_gradients: bool = True
    ) -> Tensor:
        raise NotImplementedError

    @property
    def x_is_iid(self) -> bool:
        """If x has batch dimension greater than 1, whether to intepret the batch as iid
        samples or batch of data points."""
        if self._x_is_iid is not None:
            return self._x_is_iid
        else:
            raise ValueError(
                "No observed data is available. Use `potential_fn.bind(x_o)`."
            )

    def set_x(self, x_o: Optional[Tensor], x_is_iid: Optional[bool] = True):
        """Check the shape of the observed data and, if valid, set it.

        DEPRECATED: Use bind() instead. This method delegates to bind() internally.
        """
        warnings.warn(
            "set_x() is deprecated, use bind() instead",
            FutureWarning,
            stacklevel=2,
        )
        bound = self.bind(x_o, x_is_iid=x_is_iid)
        self._x_o = bound._x_o
        self._x_is_iid = bound._x_is_iid

    @property
    def x_o(self) -> Tensor:
        """Return the observed data at which the potential is evaluated."""
        if self._x_o is not None:
            return self._x_o
        else:
            raise ValueError(
                "No observed data is available. Use `potential_fn.bind(x_o)`."
            )

    def bind(self, x_o: Tensor, x_is_iid: bool = True) -> "BasePotential":
        """Create new potential with x bound, without mutable state.

        Args:
            x_o: Observed data to bind.
            x_is_iid: Whether x represents iid observations.

        Returns:
            New potential instance with x bound.

        Subclasses must implement this method.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement bind()")

    def return_x_o(self) -> Optional[Tensor]:
        """Return the observed data at which the potential is evaluated.

        Difference to the `x_o` property is that it will not raise an error if
        `self._x_o` is `None`.
        """
        return self._x_o

    def to(self, device: Union[str, torch.device]) -> "BasePotential":
        """Deprecated: Do not call .to() on potentials.

        Potentials are immutable after creation. To move to a different device:
        - Option 1: Move estimator, prior, and x first, then build potential
        - Option 2: Use posterior.to(device) which handles this internally

        Args:
            device: Device (unused, kept for API compatibility).

        Returns:
            Self for method chaining.
        """
        warnings.warn(
            "Calling .to() on a potential is deprecated and will be removed. "
            "Move estimator, prior, and x to the device first, then build the "
            "potential, or use posterior.to(device) which handles this internally.",
            FutureWarning,
            stacklevel=2,
        )
        return self


class CustomPotential(Protocol):
    """Protocol for custom potential functions."""

    def __call__(self, theta: Tensor, x_o: Tensor) -> Tensor:
        """Call the potential function on given theta and observed data."""
        ...


class CustomPotentialWrapper(BasePotential):
    """If `potential_fn` is a callable it gets wrapped as this."""

    def __init__(
        self,
        potential_fn: CustomPotential,  # type: ignore
        prior: Optional[Distribution],  # type: ignore
        x_o: Optional[Tensor] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """Wraps a callable potential function.

        Args:
            potential_fn: Custom potential function following the CustomPotential
                protocol, i.e., the function must have exactly two positional arguments
                where the first is theta and the second is the x_o.
            prior: Prior distribution, optional at init, but needed at inference time.
            x_o: Observed data, optional at init, but needed at inference time.
            device: Device on which to evaluate the potential function.

        """
        super().__init__(prior, x_o, device)

        self.potential_fn = potential_fn

    def __call__(self, theta, track_gradients: bool = True):
        """Calls the custom potential function on given theta.

        Note, x_o is re-used from the initialization of the potential function.
        """
        with torch.set_grad_enabled(track_gradients):
            return self.potential_fn(theta, self.x_o)

    def bind(self, x_o: Tensor, x_is_iid: bool = True) -> "CustomPotentialWrapper":
        """Create new potential with x bound, without mutable state."""
        return CustomPotentialWrapper(
            potential_fn=self.potential_fn,
            prior=self.prior,
            x_o=x_o,
            device=self.device,
        )
