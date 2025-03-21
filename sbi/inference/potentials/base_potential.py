# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

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
        device: str = "cpu",
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
        self.set_x(x_o)

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
                "No observed data is available. Use `potential_fn.set_x(x_o)`."
            )

    def set_x(self, x_o: Optional[Tensor], x_is_iid: Optional[bool] = True):
        """Check the shape of the observed data and, if valid, set it."""
        if x_o is not None:
            x_o = process_x(x_o).to(self.device)
        self._x_o = x_o
        self._x_is_iid = x_is_iid

    @property
    def x_o(self) -> Tensor:
        """Return the observed data at which the potential is evaluated."""
        if self._x_o is not None:
            return self._x_o
        else:
            raise ValueError(
                "No observed data is available. Use `potential_fn.set_x(x_o)`."
            )

    @x_o.setter
    def x_o(self, x_o: Optional[Tensor]) -> None:
        """Check the shape of the observed data and, if valid, set it."""
        self.set_x(x_o)

    def return_x_o(self) -> Optional[Tensor]:
        """Return the observed data at which the potential is evaluated.

        Difference to the `x_o` property is that it will not raise an error if
        `self._x_o` is `None`.
        """
        return self._x_o


class CustomPotential(Protocol):
    """Protocol for custom potential functions."""

    def __call__(self, theta: Tensor, x_o: Tensor) -> Tensor:
        """Call the potential function on given theta and observed data."""
        ...


class CustomPotentialWrapper(BasePotential):
    """If `potential_fn` is a callable it gets wrapped as this."""

    def __init__(
        self,
        potential_fn: CustomPotential,
        prior: Optional[Distribution],
        x_o: Optional[Tensor] = None,
        device: str = "cpu",
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

    def to(self, device: Union[str, torch.device]) -> None:
        """
        Move prior and x_o to the given device.

        It also set the device attribute to the given device.

        Args:
            device: Device to move the prior and x_o to.
        """
        self.device = device
        if self.prior:
            self.prior.to(device)
        if self._x_o:
            self._x_o = self._x_o.to(device)
        super().__init__(self.prior, self._x_o, device)

    def __call__(self, theta, track_gradients: bool = True):
        """Calls the custom potential function on given theta.

        Note, x_o is re-used from the initialization of the potential function.
        """
        with torch.set_grad_enabled(track_gradients):
            return self.potential_fn(theta, self.x_o)
