"""
Neural ODEs for sampling.
"""

import torch.nn as nn
from torch import Tensor
from sbi.samplers.ode_solvers.base import NeuralODE, NeuralODEFuncType
from sbi.samplers.ode_solvers.zuko_ode import ZukoNeuralODE


__all__ = [
    "NeuralODE",
    "NeuralODEFuncType",
    "ZukoNeuralODE",
    "build_neural_ode",
]


def build_neural_ode(
        f: NeuralODEFuncType, 
        net: nn.Module,
        mean_base: Tensor,
        std_base: Tensor,
        backend: str = "zuko", 
        **kwargs
    ) -> NeuralODE:
    """
    Build a NeuralODE from a function and a neural network.

    Args:
        f: The function to be integrated.
        net: The neural network to be used.
        mean_base: The mean of the base distribution.
        std_base: The std of the base distribution.
        backend: The backend to be used. Currently only "zuko" is supported.
        **kwargs: Additional arguments provided to the backend.

    Returns:
        A NeuralODE object.
    
    Raises:
        ValueError: If the backend is not supported.
    """
    if backend == "zuko":
        return ZukoNeuralODE(f, net, mean_base, std_base, **kwargs)
    else:
        raise ValueError(f"Backend {backend} not supported")
