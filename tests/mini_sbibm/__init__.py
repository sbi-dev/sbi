# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
# NOTE: This is inspired by the sbibm-package <https://github.com/sbi-benchmark/sbibm>

from .base_task import Task
from .gaussian_linear import GaussianLinear
from .linear_mvg import LinearMVG2d
from .slcp import Slcp
from .two_moons import TwoMoons

TASKS = {
    "two_moons": TwoMoons,
    "linear_mvg_2d": LinearMVG2d,
    "gaussian_linear": GaussianLinear,
    "slcp": Slcp,
}


def get_task(name: str, *args, **kwargs) -> Task:
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
    try:
        return TASKS[name](*args, **kwargs)
    except KeyError as err:
        raise ValueError(f"Unknown task {name}") from err
