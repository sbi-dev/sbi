
import math
import os
from typing import Callable

import torch
from torch.distributions import Distribution, Independent, Normal, Uniform

from .base_task import Task

PATH = os.path.dirname(__file__)


def _map_fun(parameters: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    ang = torch.tensor([-math.pi / 4.0])
    c = torch.cos(ang)
    s = torch.sin(ang)
    z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
    z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)
    return p + torch.cat((-torch.abs(z0), z1), dim=1)


def simulator(
    parameters,
    r_loc=0.1,
    r_scale=0.01,
    a_low=-math.pi / 2.0,
    a_high=math.pi / 2.0,
    base_offset=0.25,
):
    num_samples = parameters.shape[0]

    a_dist = Uniform(
        low=a_low,
        high=a_high,
    )
    a = a_dist.sample((num_samples, 1))

    r_dist = Normal(r_loc, r_scale)
    r = r_dist.sample((num_samples, 1))

    p = torch.cat(
        (
            torch.cos(a) * r + base_offset,
            torch.sin(a) * r,
        ),
        dim=1,
    )

    return _map_fun(parameters, p)


class TwoMoons(Task):
    def __init__(self):
        super().__init__("two_moons")

    def theta_dim(self) -> int:
        return 2

    def x_dim(self) -> int:
        return 2

    def get_prior(self) -> Distribution:
        return Independent(Uniform(-torch.ones(2), torch.ones(2)), 1)

    def get_simulator(self) -> Callable:
        return simulator
