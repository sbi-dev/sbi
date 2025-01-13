from .gaussian_linear import GaussianLinear
from .linear_mvg import LinearMVG2d
from .slcp import Slcp
from .two_moons import TwoMoons


def get_task(name: str):
    """
    Retrieve a task instance based on the given name.

    Args:
        name (str): The name of the task to retrieve.
                    Possible values are "two_moons", "linear_mvg_2d",
                    "gaussian_linear", and "slcp".

    Returns:
        object: An instance of the corresponding task class.

    Raises:
        ValueError: If the provided task name is unknown.
    """
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
