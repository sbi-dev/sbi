# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
# NOTE: This is inspired by the sbibm-package <https://github.com/sbi-benchmark/sbibm>

import math
from typing import Callable

import torch
from torch.distributions import Distribution, Independent, Normal, Uniform

from .base_task import Task


def _map_fun(parameters: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Maps the parameters and points to a new space using a rotation.

    Args:
        parameters (torch.Tensor): The parameters for the mapping.
        p (torch.Tensor): The points to be mapped.

    Returns:
        torch.Tensor: The mapped points.
    """
    ang = torch.tensor(
        [-math.pi / 4.0], device=parameters.device, dtype=parameters.dtype
    )
    c = torch.cos(ang)
    s = torch.sin(ang)
    z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
    z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)
    p = p.to(parameters.device)
    return p + torch.cat((-torch.abs(z0), z1), dim=1)


def simulator(
    parameters: torch.Tensor,
    r_loc: float = 0.1,
    r_scale: float = 0.01,
    a_low: float = -math.pi / 2.0,
    a_high: float = math.pi / 2.0,
    base_offset: float = 0.25,
    device=None,
) -> torch.Tensor:
    """
    Simulator function for the Two Moons task.

    Args:
        parameters (torch.Tensor): The parameters for the simulator.
        r_loc (float, optional): The mean of the radius distribution. Defaults to 0.1.
        r_scale (float, optional): The standard deviation of the radius distribution.
        Defaults to 0.01.
        a_low (float, optional): The lower bound of the angle distribution. Defaults to
            -math.pi / 2.0.
        a_high (float, optional): The upper bound of the angle distribution. Defaults to
            math.pi / 2.0.
        base_offset (float, optional): The base offset for the points. Defaults to 0.25.

    Returns:
        torch.Tensor: The simulated data.
    """
    num_samples = parameters.shape[0]
    parameters = parameters.to(device) if device is not None else parameters
    a_dist = Uniform(
        low=torch.tensor(a_low, device=device) if device is not None else a_low,
        high=torch.tensor(a_high, device=device) if device is not None else a_high,
    )
    a = a_dist.sample((num_samples, 1))

    r_dist = Normal(
        torch.tensor(r_loc, device=device) if device is not None else r_loc,
        torch.tensor(r_scale, device=device) if device is not None else r_scale,
    )
    r = r_dist.sample((num_samples, 1))

    p = torch.cat(
        (
            torch.cos(a) * r
            + (
                torch.tensor(base_offset, device=device)
                if device is not None
                else base_offset
            ),
            torch.sin(a) * r,
        ),
        dim=1,
    )

    return _map_fun(parameters, p)


class TwoMoons(Task):
    """
    Task for the Two Moons model.

    This task uses a uniform prior and a custom simulator.
    """

    def __init__(self):
        """
        Initializes the TwoMoons task.
        """
        super().__init__("two_moons")

    def theta_dim(self) -> int:
        """
        Returns the dimensionality of the parameter space.

        Returns:
            int: The dimensionality of the parameter space.
        """
        return 2

    def x_dim(self) -> int:
        """
        Returns the dimensionality of the observation space.

        Returns:
            int: The dimensionality of the observation space.
        """
        return 2

    def get_prior(self, device=None) -> Distribution:
        """
        Returns the prior distribution over parameters.

        Args:
            device (str or torch.device, optional): The device to initialize tensors on.

        Returns:
            Distribution: The prior distribution.
        """
        low = -torch.ones(2, device=device) if device is not None else -torch.ones(2)
        high = torch.ones(2, device=device) if device is not None else torch.ones(2)
        return Independent(Uniform(low, high), 1)

    def get_simulator(self, device=None) -> Callable:
        """
        Returns the simulator function.

        Args:
            device (str or torch.device, optional): The device to initialize tensors on.

        Returns:
            Callable: The simulator function.
        """

        def sim_with_device(parameters, *args, **kwargs):
            parameters = parameters.to(device) if device is not None else parameters
            return simulator(parameters, *args, **kwargs)

        return sim_with_device
