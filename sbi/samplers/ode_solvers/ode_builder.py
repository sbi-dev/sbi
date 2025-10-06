# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""
Builder for ODE solvers.
"""

from torch import Tensor, nn

from sbi.samplers.ode_solvers.base import NeuralODE, NeuralODEFunc
from sbi.samplers.ode_solvers.zuko_ode import ZukoNeuralODE


def build_neural_ode(
    f: NeuralODEFunc,
    net: nn.Module,
    mean_base: Tensor,
    std_base: Tensor,
    backend: str = "zuko",
    t_min: float = 0.0,
    t_max: float = 1.0,
    **kwargs,
) -> NeuralODE:
    r"""
    Build a NeuralODE from a function and a neural network.

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
        std_base: The std of the base distribution.
        backend: The backend to be used. Currently only "zuko" is supported.
        t_min: The minimum time value.
        t_max: The maximum time value.
        **kwargs: Additional arguments provided to the backend.

    Returns:
        A NeuralODE object.

    Raises:
        ValueError: If the backend is not supported.
    """
    if backend == "zuko":
        return ZukoNeuralODE(f, net, mean_base, std_base, t_min, t_max, **kwargs)
    else:
        raise ValueError(f"Backend {backend} not supported")
