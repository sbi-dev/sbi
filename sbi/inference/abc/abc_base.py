import torch
from typing import Callable, Optional, Union

from sbi.simulators.simutils import simulate_in_batches


class ABCBASE:
    def __init__(
        self,
        simulator,
        prior,
        x_o,
        distance: Union[str, Callable] = "l2",
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        show_progress_bars: bool = True,
    ) -> None:

        # TODO: User input checks

        self.prior = prior
        self._simulator = simulator
        self.x_o = x_o
        self._show_progress_bars = show_progress_bars
        if type(distance) == str:
            distances = ["l1", "l2", "mse"]
            assert (
                distance in distances
            ), f"Distance function str must be one of {distances}."
            self.distance = eval(f"self.{distance}")

        self._batched_simulator = lambda theta: simulate_in_batches(
            self._simulator,
            theta,
            simulation_batch_size,
            num_workers,
            self._show_progress_bars,
        )

    @staticmethod
    def mse(observation: torch.Tensor, simulated_data: torch.Tensor) -> torch.Tensor:
        """Take mean squared distance over batch dimension

        Args:
            observation: observed data, could be 1D
            simulated_data: batch of simulated data, has batch dimension

        Returns:
            Torch tensor with batch of distances
        """
        assert simulated_data.ndim == 2, "simulated data needs batch dimension"

        return torch.mean((observation - simulated_data) ** 2, dim=-1)

    @staticmethod
    def l2(observation: torch.Tensor, simulated_data: torch.Tensor) -> torch.Tensor:
        """Take L2 distance over batch dimension

        Args:
            observation: observed data, could be 1D
            simulated_data: batch of simulated data, has batch dimension

        Returns:
            Torch tensor with batch of distances
        """
        assert (
            simulated_data.ndim == 2
        ), f"Simulated data needs batch dimension, is {simulated_data.shape}."

        return torch.norm((observation - simulated_data), dim=-1)

    @staticmethod
    def l1(observation: torch.Tensor, simulated_data: torch.Tensor) -> torch.Tensor:
        """Take mean absolute distance over batch dimension

        Args:
            observation: observed data, could be 1D
            simulated_data: batch of simulated data, has batch dimension

        Returns:
            Torch tensor with batch of distances
        """
        assert (
            simulated_data.ndim == 2
        ), f"Simulated data needs batch dimension, is {simulated_data.shape}."

        return torch.mean(abs(observation - simulated_data), dim=-1)
