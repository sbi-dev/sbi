from __future__ import annotations
from abc import ABC

from pyro.distributions.empirical import Empirical
from torch import log
from sbi.inference.abc.abc_base import ABCBASE
from typing import Callable, Optional, Union

import torch


class MCABC(ABCBASE, ABC):
    def __init__(
        self,
        simulator,
        prior,
        x_o,
        distance: Union[str, Callable] = "l2",
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        show_progress_bars: bool = True,
    ):
        """ Base class for Sequential Neural Posterior Estimation methods.

        density_estimator: Density estimator that can `.log_prob()` and `.sample()`.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            simulator=simulator,
            prior=prior,
            x_o=x_o,
            distance=distance,
            num_workers=num_workers,
            simulation_batch_size=simulation_batch_size,
            show_progress_bars=show_progress_bars,
        )

    def __call__(
        self,
        num_simulations: int,
        eps: Optional[float] = None,
        quantile: Optional[float] = None,
        return_distances: bool = False,
    ):
        # Exactly one of eps or quantile needs to be passed.
        assert (eps is not None) ^ (
            quantile is not None
        ), "Eps xor quantile must be passed."

        # Simulate and calculate distances.
        theta = self.prior.sample((num_simulations,))
        x = self._batched_simulator(theta)
        distances = self.distance(self.x_o, x)

        if eps is not None:
            is_accepted = distances < eps
            num_accepted = is_accepted.sum().item()
            assert num_accepted > 0, f"No parameters accepted, eps={eps} too small"

            theta_accepted = theta[is_accepted]
            distances_accepted = distances[is_accepted]

        elif quantile is not None:
            num_top_samples = int(num_simulations * quantile)
            sort_idx = torch.argsort(distances)
            theta_accepted = theta[sort_idx][:num_top_samples]
            distances_accepted = distances[sort_idx][:num_top_samples]

        else:
            raise ValueError("One of epsilon or quantile has to be passed.")

        posterior = Empirical(
            theta_accepted, log_weights=torch.ones(theta_accepted.shape[0])
        )

        if return_distances:
            return posterior, distances_accepted
        else:
            return posterior
