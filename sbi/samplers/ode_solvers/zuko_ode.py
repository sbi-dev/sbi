import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.transforms import FreeFormJacobianTransform

from sbi.samplers.ode_solvers.base import NeuralODE, NeuralODEFuncType


class ZukoNeuralODE(NeuralODE):
    def __init__(
        self,
        f: NeuralODEFuncType,
        net: nn.Module,
        mean_base: Tensor,
        std_base: Tensor,
        t_min: float = 0.0,
        t_max: float = 1.0,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        exact: bool = True,
    ) -> None:
        super().__init__(
            f, net, mean_base, std_base, t_min, t_max, atol=atol, rtol=rtol, exact=exact
        )

    def forward(self, condition: Tensor, **kwargs) -> Distribution:
        ode_kwargs = dict(self.kwargs)
        ode_kwargs.update(kwargs)

        transform = FreeFormJacobianTransform(
            f=lambda t, input: self.f(input, condition, t),
            t0=condition.new_tensor(self.t_min),
            t1=condition.new_tensor(self.t_max),
            phi=(condition, *self.net.parameters()),
            **ode_kwargs,
        )

        return NormalizingFlow(
            transform=transform,
            base=DiagNormal(self.mean_base, self.std_base).expand(condition.shape[:-1]),
        )
