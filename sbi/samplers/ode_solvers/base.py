from abc import abstractmethod
from typing import Callable

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from zuko.lazy import LazyDistribution

NeuralODEFuncType = Callable[[Tensor, Tensor, Tensor], Tensor]


class NeuralODE(LazyDistribution):
    r"""
    Base class for Neural ODEs that implements the LazyDistribution
    interface by `zuko`. Given the condition :math:`x_o`,
    it returns the distribution object with `log_prob` and
    `sample` methods. These methods internally solve the ODE
    for the provided function :math:`f(x_t, t, x_o)` and
    track the log-determinant of the Jacobian when
    calculating the `log_prob`.
    """

    def __init__(
        self,
        f: NeuralODEFuncType,
        net: nn.Module,
        mean_base: Tensor,
        std_base: Tensor,
        t_min: float = 0.0,
        t_max: float = 1.0,
        **kwargs,
    ):
        r"""
        Initialize the NeuralODE class.

        Args:
            f: The function to be integrated.
                Must accept three arguments in the order:
                    - input (Tensor): The input state tensor :math:`\theta_t`
                    - condition (Tensor): The conditioning tensor :math:`x_o`
                    - time (Tensor): The time parameter tensor :math:`t`
            net: The neural network that is used by the function :math:`f`.
                This is never called explicitly by the NeuralODE class,
                but is used to track the parameters of the neural network.
            mean_base: The mean of the base distribution.
                Expected shape: (1, theta_dim).
            std_base: The std of the base distribution.
                Expected shape: (1, theta_dim).
            t_min: The minimum time value for the ODE solver.
            t_max: The maximum time value for the ODE solver.
            **kwargs: Additional arguments for the ODE solver.
        """
        super().__init__()
        self.f = f
        self.net = net
        self.t_min = t_min
        self.t_max = t_max
        self.mean_base = mean_base
        self.std_base = std_base
        self.kwargs = kwargs

    @abstractmethod
    def forward(self, condition: Tensor, **kwargs) -> Distribution:
        pass
