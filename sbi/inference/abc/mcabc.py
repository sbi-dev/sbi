from __future__ import annotations

from pyro.distributions.empirical import Empirical
from torch.distributions.distribution import Distribution
from sbi.inference.abc.abc_base import ABCBASE
from typing import Callable, Optional, Union, Tuple

import torch
from torch import Tensor, ones
from numpy import ndarray


class MCABC(ABCBASE):
    def __init__(
        self,
        simulator: Callable,
        prior,
        x_o: Union[Tensor, ndarray],
        distance: Union[str, Callable] = "l2",
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        show_progress_bars: bool = True,
    ):
        """Monte-Carlo Approximate Bayesian Computation (Rejection ABC).

        Args:

        See docstring of `ABCBASE` class for all arguments.
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
    ) -> Union[Distribution, Tuple[Distribution, Tensor]]:
        r"""Run MCABC.
        
        Args:
            num_simulations: Number of simulations to run.
            eps: Acceptance threshold $\epsilon$ for distance between observed and
                simulated data.
            quantile: Upper quantile of smallest distances for which the corresponding
                parameters are returned. Exactly one of quantile or `eps` have to be
                passed.
            return_distances: Whether to return the distances corresponding to the
                selected parameters.
        Returns:
            posterior: Empirical distribution based on selected parameters.
            distances: Tensor of distances of the selected parameters.
        """
        # Exactly one of eps or quantile need to be passed.
        assert (eps is not None) ^ (
            quantile is not None
        ), "Eps xor quantile must be passed."

        # Simulate and calculate distances.
        theta = self.prior.sample((num_simulations,))
        x = self._batched_simulator(theta)
        distances = self.distance(self.x_o, x)

        # Select based on acceptance threshold epsilon.
        if eps is not None:
            is_accepted = distances < eps
            num_accepted = is_accepted.sum().item()
            assert num_accepted > 0, f"No parameters accepted, eps={eps} too small"

            theta_accepted = theta[is_accepted]
            distances_accepted = distances[is_accepted]

        # Select based on quantile on sorted distances.
        elif quantile is not None:
            num_top_samples = int(num_simulations * quantile)
            sort_idx = torch.argsort(distances)
            theta_accepted = theta[sort_idx][:num_top_samples]
            distances_accepted = distances[sort_idx][:num_top_samples]

        else:
            raise ValueError("One of epsilon or quantile has to be passed.")

        posterior = Empirical(theta_accepted, log_weights=ones(theta_accepted.shape[0]))

        if return_distances:
            return posterior, distances_accepted
        else:
            return posterior
