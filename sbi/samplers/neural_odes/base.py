
from typing import Callable, Tuple
from abc import ABC, abstractmethod

import torch
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
            t0: float = 0.0,
            t1: float = 1.0,
            **kwargs
        ):
        super().__init__()
        self.f = f
        self.net = net
        self.t0 = t0
        self.t1 = t1

    @abstractmethod
    def forward(self, condition: Tensor) -> Distribution:
        pass
