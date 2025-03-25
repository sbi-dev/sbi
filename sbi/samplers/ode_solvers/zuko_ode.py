# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""
Zuko ODE solver.
"""

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.transforms import FreeFormJacobianTransform

from sbi.samplers.ode_solvers.base import NeuralODE, NeuralODEFunc


class ZukoNeuralODE(NeuralODE):
    def __init__(
        self,
        f: NeuralODEFunc,
        net: nn.Module,
        mean_base: Tensor,
        std_base: Tensor,
        t_min: float = 0.0,
        t_max: float = 1.0,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        exact: bool = True,
    ) -> None:
        r"""
        Initialize the ZukoNeuralODE class.

        References:
        ----------
        .. [1] FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative
           Models (Grathwohl et al., 2018)
           https://arxiv.org/abs/1810.01367

        Args:
            f: The function to be integrated that implements the `NeuralODEFunc`
                protocol. Must accept three arguments in the order:
                - input (Tensor): The input state tensor :math:`\theta_t`
                - condition (Tensor): The conditioning tensor :math:`x_o`
                - times (Tensor): The time parameter tensor :math:`t`
            net: The neural network that is used by the function :math:`f`.
                This is never called explicitly by the NeuralODE class,
                but is used to track the parameters of the neural network.
            mean_base: The mean of the base distribution.
                Expected shape: (1, theta_dim).
            std_base: The std of the base distribution.
                Expected shape: (1, theta_dim).
            t_min: The minimum time value for the ODE solver.
            t_max: The maximum time value for the ODE solver.
            atol: The absolute tolerance for the ODE solver.
            rtol: The relative tolerance for the ODE solver.
            exact: Whether the exact log-determinant of the Jacobian or an unbiased
            stochastic estimate thereof is calculated.
        """

        super().__init__(
            f,
            net,
            mean_base,
            std_base,
            t_min,
            t_max,
            atol=atol,
            rtol=rtol,
            exact=exact,
        )

    def get_distribution(self, condition: Tensor, **kwargs) -> Distribution:
        """
        Get the distribution that wraps the ODE solver.

        Args:
            condition: The condition tensor.
            **kwargs: Additional arguments for the ODE solver.

        Returns:
            The distribution object with `log_prob` and
            `sample` methods that wraps the ODE solver.
        """
        transform = FreeFormJacobianTransform(
            f=lambda t, input: self.f(input, condition, t),
            t0=condition.new_tensor(self.t_min),
            t1=condition.new_tensor(self.t_max),
            phi=(condition, *self.net.parameters()),
            **kwargs,
        )

        return NormalizingFlow(
            transform=transform,
            base=DiagNormal(self.mean_base, self.std_base).expand(condition.shape[:-1]),
        )
