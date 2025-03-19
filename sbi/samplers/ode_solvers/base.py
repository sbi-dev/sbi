from abc import abstractmethod
from typing import Callable

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from zuko.lazy import LazyDistribution

NeuralODEFuncType = Callable[[Tensor, Tensor, Tensor], Tensor]
NEURAL_ODE_FUNC_DOCS = """Neural ODE function that computes the time derivative.
Must accept three arguments in the order:
    - input (Tensor): The input state tensor
    - condition (Tensor): The conditioning tensor
    - time (Tensor): The time parameter tensor

Returns:
    Tensor: The computed time derivative
"""


class NeuralODE(LazyDistribution):
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
