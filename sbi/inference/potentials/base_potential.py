from abc import ABCMeta, abstractmethod
from typing import Optional

from torch import Tensor
from torch.distributions import Distribution

from sbi.utils.user_input_checks import process_x


class BasePotential(metaclass=ABCMeta):
    def __init__(
        self, prior: Distribution, x_o: Optional[Tensor] = None, device: str = "cpu"
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

    @property
    @abstractmethod
    def allow_iid_x(self) -> bool:
        raise NotImplementedError

    def set_x(self, x_o: Optional[Tensor]):
        """Check the shape of the observed data and, if valid, set it."""
        if x_o is not None:
            x_o = process_x(x_o, allow_iid_x=self.allow_iid_x).to(self.device)
        self._x_o = x_o

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
