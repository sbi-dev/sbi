# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
# NOTE: This is inspired by the sbibm-package <https://github.com/sbi-benchmark/sbibm>

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
    """
    Task for the Linear Multivariate Gaussian (MVG) model in 2D.

    This task uses a linear Gaussian model with a multivariate normal prior.
    """

    def __init__(self):
        """
        Initializes the LinearMVG2d task.
        """
        self.likelihood_shift = torch.tensor([-1.0, 1.0])
        self.likelihood_cov = torch.tensor([[0.6, 0.5], [0.5, 0.6]])
        super().__init__("linear_mvg_2d")

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
            self.likelihood_shift,
            self.likelihood_cov,
            torch.zeros(2),
            torch.eye(2),
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
        return MultivariateNormal(torch.zeros(2), torch.eye(2))

    def get_simulator(self) -> Callable:
        """
        Returns the simulator function.

        Returns:
            Callable: The simulator function.
        """
        return partial(
            linear_gaussian,
            likelihood_shift=self.likelihood_shift,
            likelihood_cov=self.likelihood_cov,
        )
