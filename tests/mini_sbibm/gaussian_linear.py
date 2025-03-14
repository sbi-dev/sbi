# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
# NOTE: This is inspired by the sbibm-package <https://github.com/sbi-benchmark/sbibm>

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
    """
    Task for the Gaussian Linear model.

    This task uses a linear Gaussian model with a multivariate normal prior.
    """

    def __init__(self):
        """
        Initializes the GaussianLinear task.
        """
        self.simulator_scale = 0.1
        self.dim = 5
        super().__init__("gaussian_linear")

    def theta_dim(self) -> int:
        """
        Returns the dimensionality of the parameter space.

        Returns:
            int: The dimensionality of the parameter space.
        """
        return self.dim

    def x_dim(self) -> int:
        """
        Returns the dimensionality of the observation space.

        Returns:
            int: The dimensionality of the observation space.
        """
        return self.dim

    def get_reference_posterior_samples(self, idx: int) -> torch.Tensor:
        """
        Generates reference posterior samples for a specific observation.

        Args:
            idx (int): The index of the observation.

        Returns:
            torch.Tensor: The reference posterior samples.
        """
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
        """
        Generates the true parameters for a specific observation.

        Args:
            idx (int): The index of the observation.

        Returns:
            torch.Tensor: The true parameters.
        """
        torch.manual_seed(idx)
        return self.get_prior().sample()

    def get_observation(self, idx: int) -> torch.Tensor:
        """
        Generates an observation for a specific set of true parameters.

        Args:
            idx (int): The index of the observation.

        Returns:
            torch.Tensor: The observation.
        """
        theta_o = self.get_true_parameters(idx)
        x_o = self.get_simulator()(theta_o[None, :])[0]
        return x_o

    def get_prior(self) -> Distribution:
        """
        Returns the prior distribution over parameters.

        Returns:
            Distribution: The prior distribution.
        """
        return MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))

    def get_simulator(self) -> Callable:
        """
        Returns the simulator function.

        Returns:
            Callable: The simulator function.
        """
        return partial(
            diagonal_linear_gaussian,
            std=self.simulator_scale,
        )
