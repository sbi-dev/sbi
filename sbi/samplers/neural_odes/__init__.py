"""
Neural ODEs for sampling.
"""

import torch.nn as nn

from sbi.samplers.neural_odes.base import NeuralODE, NeuralODEFuncType
from sbi.samplers.neural_odes.zuko_ode import ZukoNeuralODE


__all__ = [
    "NeuralODE",
    "NeuralODEFuncType",
    "ZukoNeuralODE",
    "build_neural_ode",
]


def build_neural_ode(f: NeuralODEFuncType, net: nn.Module, backend: str = "zuko", **kwargs) -> NeuralODE:
    """
    Build a NeuralODE from a function and a neural network.

    Args:
        f: The function to be integrated.
        net: The neural network to be used.
        backend: The backend to be used. Currently only "zuko" is supported.
        **kwargs: Additional arguments provided to the backend.

    Returns:
        A NeuralODE object.
    
    Raises:
        ValueError: If the backend is not supported.
    """
    if backend == "zuko":
        return ZukoNeuralODE(f, net, **kwargs)
    else:
        raise ValueError(f"Backend {backend} not supported")
