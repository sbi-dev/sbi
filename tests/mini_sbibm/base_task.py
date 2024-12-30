import os
from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch.distributions import Distribution

PATH = os.path.dirname(__file__)


class Task(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def theta_dim(self) -> int:
        pass

    @abstractmethod
    def x_dim(self) -> int:
        pass

    @abstractmethod
    def get_prior(self) -> Distribution:
        pass

    @abstractmethod
    def get_simulator(self) -> Callable:
        pass

    def get_data(self, num_sims: int):
        thetas = self.get_prior().sample((num_sims,))
        xs = self.get_simulator()(thetas)
        return thetas, xs

    def get_observation(self, idx: int) -> torch.Tensor:
        x_o = torch.load(
            PATH + os.sep + "files" + os.sep + f"{self.name}{os.sep}x_o_{idx}.pt"
        )
        return x_o

    def get_true_parameters(self, idx: int) -> torch.Tensor:
        theta = torch.load(
            PATH + os.sep + "files" + os.sep + f"{self.name}{os.sep}theta_{idx}.pt"
        )
        return theta

    def get_reference_posterior_samples(self, idx: int) -> torch.Tensor:
        posterior_samples = torch.load(
            PATH + os.sep + "files" + os.sep + f"{self.name}{os.sep}samples_{idx}.pt"
        )
        return posterior_samples
