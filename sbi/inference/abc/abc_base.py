
from abc import ABC
from typing import Callable, Optional, Union, Dict, Any, Tuple, Union, cast, List, Sequence, TypeVar

import torch
from torch import Tensor

from sbi.simulators.simutils import simulate_in_batches


class ABCBASE(ABC):
    def __init__(
        self,
        simulator: Callable,
        prior,
        distance: Union[str, Callable] = "l2",
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        show_progress_bars: bool = True,
    ) -> None:
        r"""Base class for Approximate Bayesian Computation methods.

        Args:
            simulator: A function that takes parameters $\theta$ and maps them to
                simulations, or observations, `x`, $\mathrm{sim}(\theta)\to x$. Any
                regular Python callable (i.e. function or class with `__call__` method)
                can be used.
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            distance: Distance function to compare observed and simulated data. Can be
                a custom function or one of `l1`, `l2`, `mse`.
            num_workers: Number of parallel workers to use for simulations.
            simulation_batch_size: Number of parameter sets that the simulator
                maps to data x at once. If None, we simulate all parameter sets at the
                same time. If >= 1, the simulator has to process data of shape
                (simulation_batch_size, parameter_dimension).
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
        """

        self.prior = prior
        self._simulator = simulator
        self._show_progress_bars = show_progress_bars

        # Select distance function.
        if type(distance) == str:
            distances = ["l1", "l2", "mse"]
            assert (
                distance in distances
            ), f"Distance function str must be one of {distances}."
            self.distance = self.choose_distance_function(distance_type=distance)

        self._batched_simulator = lambda theta: simulate_in_batches(
            simulator=self._simulator,
            theta=theta,
            sim_batch_size=simulation_batch_size,
            num_workers=num_workers,
            show_progress_bars=self._show_progress_bars,
        )

    @staticmethod
    def choose_distance_function(distance_type: str = "l2") -> Callable:
        """Return distance function for given distance type."""

        if distance_type == "mse":
            distance = lambda xo, x: torch.mean((xo - x) ** 2, dim=-1)
        elif distance_type == "l2":
            distance = lambda xo, x: torch.norm((xo - x), dim=-1)
        elif distance_type == "l1":
            distance = lambda xo, x: torch.mean(abs(xo - x), dim=-1)
        else:
            raise ValueError(r"Distance {distance_type} not supported.")

        def distance_fun(observed_data: Tensor, simulated_data: Tensor) -> Tensor:
            """Return distance over batch dimension.

            Args:
                observed_data: Observed data, could be 1D.
                simulated_data: Batch of simulated data, has batch dimension.

            Returns:
                Torch tensor with batch of distances.
            """
            assert simulated_data.ndim == 2, "simulated data needs batch dimension"

            return distance(observed_data, simulated_data)

        return distance_fun
