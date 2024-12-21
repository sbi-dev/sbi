from .gaussian_linear import GaussianLinear
from .linear_mvg import LinearMVG2d
from .slcp import Slcp
from .two_moons import TwoMoons


def get_task(name: str):
    if name == "two_moons":
        return TwoMoons()
    elif name == "linear_mvg_2d":
        return LinearMVG2d()
    elif name == "gaussian_linear":
        return GaussianLinear()
    elif name == "slcp":
        return Slcp()
    else:
        raise ValueError(f"Unknown task {name}")
