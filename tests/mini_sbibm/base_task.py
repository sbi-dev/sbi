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
    def get_prior(self, device=None) -> Distribution:
        """
        Returns the prior distribution over parameters.

        Args:
            device (str or torch.device, optional): The device to initialize tensors on.

        Returns:
            Distribution: The prior distribution.
        """
        pass

    @abstractmethod
    def get_simulator(self, device=None) -> Callable:
        """
        Returns the simulator function.

        Args:
            device (str or torch.device, optional): The device to initialize tensors on.

        Returns:
            Callable: The simulator function.
        """
        pass

    def get_data(self, num_sims: int, device=None):
        """
        Generates data by sampling from the prior and simulating observations.

        Args:
            num_sims (int): The number of simulations to run.
            device (str or torch.device, optional): The device to initialize tensors on.

        Returns:
            tuple: A tuple containing the sampled parameters and simulated observations.
        """
        thetas = self.get_prior(device=device).sample((num_sims,))
        xs = self.get_simulator(device=device)(thetas)
        return thetas, xs

    def get_observation(self, idx: int, device=None) -> torch.Tensor:
        """
        Loads a specific observation from file.

        Args:
            idx (int): The index of the observation to load.
            device (str or torch.device, optional): The device to move the tensor to.

        Returns:
            torch.Tensor: The loaded observation.
        """
        x_o = torch.load(
            PATH + os.sep + "files" + os.sep + f"{self.name}{os.sep}x_o_{idx}.pt",
            weights_only=False,
        )
        if device is not None:
            x_o = x_o.to(device)
        return x_o

    def get_true_parameters(self, idx: int, device=None) -> torch.Tensor:
        """
        Loads the true parameters for a specific observation from file.

        Args:
            idx (int): The index of the observation to load.
            device (str or torch.device, optional): The device to move the tensor to.

        Returns:
            torch.Tensor: The loaded true parameters.
        """
        theta_o = torch.load(
            PATH + os.sep + "files" + os.sep + f"{self.name}{os.sep}theta_o_{idx}.pt",
            weights_only=False,
        )
        if device is not None:
            theta_o = theta_o.to(device)
        return theta_o

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
