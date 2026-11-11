# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
# NOTE: This is inspired by the sbibm-package <https://github.com/sbi-benchmark/sbibm>

from typing import Callable

import torch
from torch.distributions import Distribution, Independent, MultivariateNormal, Uniform

from .base_task import Task


def simulator(theta, num_data=4):
    num_samples = theta.shape[0]

    m = torch.stack((theta[:, [0]].squeeze(), theta[:, [1]].squeeze())).T
    if m.dim() == 1:
        m.unsqueeze_(0)

    s1 = theta[:, [2]].squeeze() ** 2
    s2 = theta[:, [3]].squeeze() ** 2
    rho = torch.nn.Tanh()(theta[:, [4]]).squeeze()

    S = torch.empty((num_samples, 2, 2))
    S[:, 0, 0] = s1**2
    S[:, 0, 1] = rho * s1 * s2
    S[:, 1, 0] = rho * s1 * s2
    S[:, 1, 1] = s2**2

    # Add eps to diagonal to ensure PSD
    eps = 0.000001
    S[:, 0, 0] += eps
    S[:, 1, 1] += eps

    data_dist = MultivariateNormal(m, S)
    xs = data_dist.sample((num_data,))
    xs = xs.permute(1, 0, 2)

    return xs.reshape(num_samples, num_data * 2)


class Slcp(Task):
    def __init__(self):
        super().__init__("slcp")

    def theta_dim(self) -> int:
        return 5

    def x_dim(self) -> int:
        return 8

    def get_prior(self) -> Distribution:
        return Independent(Uniform(-3 * torch.ones(5), 3 * torch.ones(5)), 1)

    def get_simulator(self) -> Callable:
        return simulator
