from functools import partial
from typing import Callable

import torch
from torch.distributions import Distribution, MultivariateNormal

from sbi.simulators.linear_gaussian import (
    diagonal_linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)

from .base_task import Task


class GaussianLinear(Task):
    def __init__(self):
        self.simulator_scale = 0.1
        self.dim = 5
        super().__init__("gaussian_linear")

    def theta_dim(self) -> int:
        return self.dim

    def x_dim(self) -> int:
        return self.dim

    def get_reference_posterior_samples(self, idx: int) -> torch.Tensor:
        x_o = self.get_observation(idx)
        posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o,
            torch.zeros(self.dim),
            self.simulator_scale * torch.eye(self.dim),
            torch.zeros(self.dim),
            torch.eye(self.dim),
        )

        return posterior.sample((10_000,))

    def get_true_parameters(self, idx: int) -> torch.Tensor:
        torch.manual_seed(idx)
        return self.get_prior().sample()

    def get_observation(self, idx: int) -> torch.Tensor:
        theta_o = self.get_true_parameters(idx)
        x_o = self.get_simulator()(theta_o[None, :])[0]
        return x_o

    def get_prior(self) -> Distribution:
        return MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))

    def get_simulator(self) -> Callable:
        return partial(
            diagonal_linear_gaussian,
            std=self.simulator_scale,
        )
