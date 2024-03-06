"""Base class for Approximate Bayesian Computation methods."""

import logging
from typing import Callable, Union

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch import Tensor

from sbi.simulators.simutils import simulate_in_batches


class ABCBASE:
    """Base class for Approximate Bayesian Computation methods."""

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
                a custom callable function or one of `l1`, `l2`, `mse`.
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

        self.x_o = None
        self.x_shape = None

        # Select distance function.
        self.distance = self.get_distance_function(distance)

        self._batched_simulator = lambda theta: simulate_in_batches(
            simulator=self._simulator,
            theta=theta,
            sim_batch_size=simulation_batch_size,
            num_workers=num_workers,
            show_progress_bars=self._show_progress_bars,
        )

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_distance_function(distance_type: Union[str, Callable] = "l2") -> Callable:
        """Return distance function for given distance type.

        Args:
            distance_type: string indicating the distance type, e.g., 'l2', 'l1',
                'mse'. Note that the returned distance function averages over the last
                dimension, e.g., over the summary statistics.

        Returns:
            distance_fun: distance functions built from passe string. Returns
                distance_type is callable.
        """

        if isinstance(distance_type, Callable):
            return distance_type

        # Select distance function.
        implemented_distances = ["l1", "l2", "mse"]
        assert (
            distance_type in implemented_distances
        ), f"{distance_type} must be one of {implemented_distances}."

        def mse_distance(xo, x):
            return torch.mean((xo - x) ** 2, dim=-1)

        def l2_distance(xo, x):
            return torch.norm((xo - x), dim=-1)

        def l1_distance(xo, x):
            return torch.mean(abs(xo - x), dim=-1)

        distance_functions = {"mse": mse_distance, "l2": l2_distance, "l1": l1_distance}

        try:
            distance = distance_functions[distance_type]
        except KeyError as exc:
            raise KeyError(f"Distance {distance_type} not supported.") from exc

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

    @staticmethod
    def get_sass_transform(
        theta: torch.Tensor,
        x: torch.Tensor,
        expansion_degree: int = 1,
        sample_weight=None,
    ) -> Callable:
        """Return semi-automatic summary statitics function.

        Running weighted linear regressin as in
        Fearnhead & Prandle 2012: https://arxiv.org/abs/1004.1112

        Following implementation in
        https://abcpy.readthedocs.io/en/latest/_modules/abcpy/statistics.html#Identity
        and
        https://pythonhosted.org/abcpy/_modules/abcpy/summaryselections.html#Semiautomatic
        """
        expansion = PolynomialFeatures(degree=expansion_degree, include_bias=False)
        # Transform x, remove intercept.
        x_expanded = expansion.fit_transform(x)
        sumstats_map = np.zeros((x_expanded.shape[1], theta.shape[1]))

        for parameter_idx in range(theta.shape[1]):
            regression_model = LinearRegression(fit_intercept=True)
            regression_model.fit(
                X=x_expanded, y=theta[:, parameter_idx], sample_weight=sample_weight
            )
            sumstats_map[:, parameter_idx] = regression_model.coef_

        sumstats_map = torch.tensor(sumstats_map, dtype=torch.float32)

        def sumstats_transform(x):
            x_expanded = torch.tensor(expansion.fit_transform(x), dtype=torch.float32)
            return x_expanded.mm(sumstats_map)

        return sumstats_transform

    @staticmethod
    def run_lra(
        theta: torch.Tensor,
        x: torch.Tensor,
        observation: torch.Tensor,
        sample_weight=None,
    ) -> torch.Tensor:
        """Return parameters adjusted with linear regression adjustment.

        Implementation as in Beaumont et al. 2002: https://arxiv.org/abs/1707.01254
        """

        theta_adjusted = theta
        for parameter_idx in range(theta.shape[1]):
            regression_model = LinearRegression(fit_intercept=True)
            regression_model.fit(
                X=x,
                y=theta[:, parameter_idx],
                sample_weight=sample_weight,
            )
            theta_adjusted[:, parameter_idx] += regression_model.predict(
                observation.reshape(1, -1)
            )
            theta_adjusted[:, parameter_idx] -= regression_model.predict(x)

        return theta_adjusted
