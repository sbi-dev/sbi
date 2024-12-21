from functools import partial
from typing import Callable

import torch
from torch.distributions import Distribution, MultivariateNormal

from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)

from .base_task import Task


class LinearMVG2d(Task):
    def __init__(self):
        self.likelihood_shift = torch.tensor([-1.0, 1.0])
        self.likelihood_cov = torch.tensor([[0.6, 0.5], [0.5, 0.6]])
        super().__init__("linear_mvg_2d")

    def theta_dim(self) -> int:
        return 2

    def x_dim(self) -> int:
        return 2

    def get_reference_posterior_samples(self, idx: int) -> torch.Tensor:
        x_o = self.get_observation(idx)
        posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o,
            self.likelihood_shift,
            self.likelihood_cov,
            torch.zeros(2),
            torch.eye(2),
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
        return MultivariateNormal(torch.zeros(2), torch.eye(2))

    def get_simulator(self) -> Callable:
        return partial(
            linear_gaussian,
            likelihood_shift=self.likelihood_shift,
            likelihood_cov=self.likelihood_cov,
        )
