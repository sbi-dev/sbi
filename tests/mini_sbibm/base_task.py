# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
# NOTE: This is inspired by the sbibm-package <https://github.com/sbi-benchmark/sbibm>

import os
from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch.distributions import Distribution

PATH = os.path.dirname(__file__)


class Task(ABC):
    """
    Abstract base class for a task in the SBI benchmark.

    Args:
        name (str): The name of the task.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def theta_dim(self) -> int:
        """
        Returns the dimensionality of the parameter space.

        Returns:
            int: The dimensionality of the parameter space.
        """
        pass

    @abstractmethod
    def x_dim(self) -> int:
        """
        Returns the dimensionality of the observation space.

        Returns:
            int: The dimensionality of the observation space.
        """
        pass

    @abstractmethod
    def get_prior(self) -> Distribution:
        """
        Returns the prior distribution over parameters.

        Returns:
            Distribution: The prior distribution.
        """
        pass

    @abstractmethod
    def get_simulator(self) -> Callable:
        """
        Returns the simulator function.

        Returns:
            Callable: The simulator function.
        """
        pass

    def get_data(self, num_sims: int):
        """
        Generates data by sampling from the prior and simulating observations.

        Args:
            num_sims (int): The number of simulations to run.

        Returns:
            tuple: A tuple containing the sampled parameters and simulated observations.
        """
        thetas = self.get_prior().sample((num_sims,))
        xs = self.get_simulator()(thetas)
        return thetas, xs

    def get_observation(self, idx: int) -> torch.Tensor:
        """
        Loads a specific observation from file.

        Args:
            idx (int): The index of the observation to load.

        Returns:
            torch.Tensor: The loaded observation.
        """
        x_o = torch.load(
            PATH + os.sep + "files" + os.sep + f"{self.name}{os.sep}x_o_{idx}.pt",
            weights_only=False,
        )
        return x_o

    def get_true_parameters(self, idx: int) -> torch.Tensor:
        """
        Loads the true parameters for a specific observation from file.

        Args:
            idx (int): The index of the parameters to load.

        Returns:
            torch.Tensor: The loaded true parameters.
        """
        theta = torch.load(
            PATH + os.sep + "files" + os.sep + f"{self.name}{os.sep}theta_{idx}.pt"
        )
        return theta

    def get_reference_posterior_samples(self, idx: int) -> torch.Tensor:
        """
        Loads reference posterior samples for a specific observation from file.

        Args:
            idx (int): The index of the posterior samples to load.

        Returns:
            torch.Tensor: The loaded posterior samples.
        """
        posterior_samples = torch.load(
            PATH + os.sep + "files" + os.sep + f"{self.name}{os.sep}samples_{idx}.pt",
            weights_only=False,
        )
        return posterior_samples
