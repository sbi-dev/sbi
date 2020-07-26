
from typing import Callable, Optional, Union, Dict, Any, Tuple, Union, cast, List, Sequence, TypeVar

import torch
from numpy import ndarray
from pyro.distributions.empirical import Empirical
from torch import Tensor, ones
from torch.distributions.distribution import Distribution

from sbi.inference.abc.abc_base import ABCBASE
from sbi.user_input.user_input_checks import process_x


class MCABC(ABCBASE):
    def __init__(
        self,
        simulator: Callable,
        prior,
        distance: Union[str, Callable] = "l2",
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        show_progress_bars: bool = True,
    ):
        """Monte-Carlo Approximate Bayesian Computation (Rejection ABC) [1].

        [1] Pritchard, J. K., Seielstad, M. T., Perez-Lezaun, A., & Feldman, M. W.
        (1999). Population growth of human Y chromosomes: a study of Y chromosome
        microsatellites. Molecular biology and evolution, 16(12), 1791-1798.

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

        super().__init__(
            simulator=simulator,
            prior=prior,
            distance=distance,
            num_workers=num_workers,
            simulation_batch_size=simulation_batch_size,
            show_progress_bars=show_progress_bars,
        )

    def __call__(
        self,
        x_o: Union[Tensor, ndarray],
        num_simulations: int,
        eps: Optional[float] = None,
        quantile: Optional[float] = None,
        return_distances: bool = False,
    ) -> Union[Distribution, Tuple[Distribution, Tensor]]:
        r"""Run MCABC.

        Args:
            x_o: Observed data.
            num_simulations: Number of simulations to run.
            eps: Acceptance threshold $\epsilon$ for distance between observed and
                simulated data.
            quantile: Upper quantile of smallest distances for which the corresponding
                parameters are returned, e.g, q=0.01 will return the top 1%. Exactly
                one of quantile or `eps` have to be passed.
            return_distances: Whether to return the distances corresponding to the
                selected parameters.
        Returns:
            posterior: Empirical distribution based on selected parameters.
            distances: Tensor of distances of the selected parameters.
        """
        # Exactly one of eps or quantile need to be passed.
        assert (eps is not None) ^ (
            quantile is not None
        ), "Eps or quantile must be passed, but not both."

        # Simulate and calculate distances.
        theta = self.prior.sample((num_simulations,))
        x = self._batched_simulator(theta)

        # Infer shape of x to test and set x_o.
        self.x_shape = x[0].unsqueeze(0).shape
        self.x_o = process_x(x_o, self.x_shape)

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
