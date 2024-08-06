# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Monte-Carlo Approximate Bayesian Computation (Rejection ABC)."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from numpy import ndarray
from torch import Tensor

from sbi.inference.abc.abc_base import ABCBASE
from sbi.utils.kde import KDEWrapper, get_kde
from sbi.utils.user_input_checks import process_x


class MCABC(ABCBASE):
    """Monte-Carlo Approximate Bayesian Computation (Rejection ABC)."""

    def __init__(
        self,
        simulator: Callable,
        prior,
        distance: Union[str, Callable] = "l2",
        requires_iid_data: Optional[None] = None,
        distance_kwargs: Optional[Dict] = None,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        distance_batch_size: int = -1,
        show_progress_bars: bool = True,
    ):
        r"""Monte-Carlo Approximate Bayesian Computation (Rejection ABC) [1].

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
                a custom callable function or one of `l1`, `l2`, `mse`,
                `mmd`, `wasserstein`.
            requires_iid_data: Whether to allow conditioning on iid sampled data or not.
                Typically, this information is inferred by the choice of the distance,
                but in case a custom distance is used, this information is pivotal.
            distance_kwargs: Configurations parameters for the distances. In particular
                useful for the MMD and Wasserstein distance.
            num_workers: Number of parallel workers to use for simulations.
            simulation_batch_size: Number of parameter sets that the simulator
                maps to data x at once. If None, we simulate all parameter sets at the
                same time. If >= 1, the simulator has to process data of shape
                (simulation_batch_size, parameter_dimension).
            distance_batch_size: Number of simulations that the distance function
                evaluates against the reference observations at once. If -1, we evaluate
                all simulations at the same time.
        """

        super().__init__(
            simulator=simulator,
            prior=prior,
            distance=distance,
            requires_iid_data=requires_iid_data,
            distance_kwargs=distance_kwargs,
            num_workers=num_workers,
            simulation_batch_size=simulation_batch_size,
            distance_batch_size=distance_batch_size,
            show_progress_bars=show_progress_bars,
        )

    def __call__(
        self,
        x_o: Union[Tensor, ndarray],
        num_simulations: int,
        eps: Optional[float] = None,
        quantile: Optional[float] = None,
        lra: bool = False,
        sass: bool = False,
        sass_fraction: float = 0.25,
        sass_expansion_degree: int = 1,
        kde: bool = False,
        kde_kwargs: Optional[Dict[str, Any]] = None,
        return_summary: bool = False,
        num_iid_samples: int = 1,
    ) -> Union[Tuple[Tensor, dict], Tuple[KDEWrapper, dict], Tensor, KDEWrapper]:
        r"""Run MCABC and return accepted parameters or KDE object fitted on them.

        Args:
            x_o: Observed data.
            num_simulations: Number of simulations to run.
            eps: Acceptance threshold $\epsilon$ for distance between observed and
                simulated data.
            quantile: Upper quantile of smallest distances for which the corresponding
                parameters are returned, e.g, q=0.01 will return the top 1%. Exactly
                one of quantile or `eps` have to be passed.
            lra: Whether to run linear regression adjustment as in Beaumont et al. 2002
            sass: Whether to determine semi-automatic summary statistics as in
                Fearnhead & Prangle 2012.
            sass_fraction: Fraction of simulation budget used for the initial sass run.
            sass_expansion_degree: Degree of the polynomial feature expansion for the
                sass regression, default 1 - no expansion.
            kde: Whether to run KDE on the accepted parameters to return a KDE
                object from which one can sample.
            kde_kwargs: kwargs for performing KDE:
                'bandwidth='; either a float, or a string naming a bandwidth
                heuristics, e.g., 'cv' (cross validation), 'silvermann' or 'scott',
                default 'cv'.
                'transform': transform applied to the parameters before doing KDE.
                'sample_weights': weights associated with samples. See 'get_kde' for
                more details
            return_summary: Whether to return the distances and data corresponding to
                the accepted parameters.
            num_iid_samples: Number of simulations per parameter. Choose
                `num_iid_samples>1`, if you have chosen a statistical distance that
                evaluates sets of simulations against a set of reference observations
                instead of a single data-point comparison.

        Returns:
            theta (if kde False): accepted parameters
            kde (if kde True): KDE object based on accepted parameters from which one
                can .sample() and .log_prob().
            summary (if summary True): dictionary containing the accepted paramters (if
                kde True), distances and simulated data x.
        """

        # Exactly one of eps or quantile need to be passed.
        assert (eps is not None) ^ (
            quantile is not None
        ), "Eps or quantile must be passed, but not both."
        if kde_kwargs is None:
            kde_kwargs = {}

        # Run SASS and change the simulator and x_o accordingly.
        if sass:
            num_pilot_simulations = int(sass_fraction * num_simulations)
            self.logger.info(
                "Running SASS with %s pilot samples.", num_pilot_simulations
            )
            num_simulations -= num_pilot_simulations

            pilot_theta = self.prior.sample((num_pilot_simulations,))
            pilot_x = self._batched_simulator(pilot_theta)

            sass_transform = self.get_sass_transform(
                pilot_theta, pilot_x, sass_expansion_degree
            )

            # Add sass transform to simulator and x_o.
            def simulator(theta):
                return sass_transform(self._batched_simulator(theta))

            x_o = sass_transform(x_o)
        else:
            simulator = self._batched_simulator

        # Simulate and calculate distances.
        theta = self.prior.sample((num_simulations,))
        theta_repeat = theta.repeat_interleave(num_iid_samples, dim=0)
        x = simulator(theta_repeat)
        x = x.reshape((
            num_simulations,
            num_iid_samples,
            -1,
        ))  # Dim(num_initial_pop, num_iid_samples, -1)

        # Infer x shape to test and set x_o.
        if not self.distance.requires_iid_data:
            x = x.squeeze(1)
            self.x_shape = x[0].shape
            self.x_o = process_x(x_o, self.x_shape)
        else:
            self.x_shape = x[0, 0].shape
            self.x_o = process_x(x_o, self.x_shape)

        distances = self.distance(self.x_o, x)

        # Select based on acceptance threshold epsilon.
        if eps is not None:
            is_accepted = distances < eps
            num_accepted = is_accepted.sum().item()
            assert num_accepted > 0, f"No parameters accepted, eps={eps} too small"

            theta_accepted = theta[is_accepted]
            distances_accepted = distances[is_accepted]
            x_accepted = x[is_accepted]

        # Select based on quantile on sorted distances.
        elif quantile is not None:
            num_top_samples = int(num_simulations * quantile)
            sort_idx = torch.argsort(distances)
            theta_accepted = theta[sort_idx][:num_top_samples]
            distances_accepted = distances[sort_idx][:num_top_samples]
            x_accepted = x[sort_idx][:num_top_samples]

        else:
            raise ValueError("One of epsilon or quantile has to be passed.")

        # Maybe adjust theta with LRA.
        if lra:
            self.logger.info("Running Linear regression adjustment.")
            final_theta = self.run_lra(theta_accepted, x_accepted, observation=self.x_o)
        else:
            final_theta = theta_accepted

        if kde:
            self.logger.info(
                """KDE on %s samples with bandwidth option
                {kde_kwargs["bandwidth"] if "bandwidth" in kde_kwargs else "cv"}.
                Beware that KDE can give unreliable results when used with too few
                samples and in high dimensions.""",
                final_theta.shape[0],
            )

            kde_dist = get_kde(final_theta, **kde_kwargs)

            if return_summary:
                return (
                    kde_dist,
                    dict(theta=final_theta, distances=distances_accepted, x=x_accepted),
                )
            else:
                return kde_dist
        elif return_summary:
            return final_theta, dict(distances=distances_accepted, x=x_accepted)
        else:
            return final_theta
