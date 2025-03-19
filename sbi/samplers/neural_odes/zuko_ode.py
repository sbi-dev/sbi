from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from zuko.transforms import FreeFormJacobianTransform
from zuko.distributions import NormalizingFlow, DiagNormal

from sbi.samplers.neural_odes.base import NeuralODE


class ZukoNeuralODE(NeuralODE): 
    def forward(self, condition: Tensor) -> Distribution:
        transform = FreeFormJacobianTransform(
            f=lambda t, input: self.f(input, condition, t),
            t0=condition.new_tensor(self.t0),
            t1=condition.new_tensor(self.t1),
            phi=(condition, *self.net.parameters()),
        )

        return NormalizingFlow(
            transform=transform,
            base=DiagNormal(self.zeros, self.ones).expand(condition.shape[:-1]),
        )
