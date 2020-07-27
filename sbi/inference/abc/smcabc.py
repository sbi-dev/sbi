
import logging
from typing import Callable, Optional, Union, Dict, Any, Tuple, Union, cast, List, Sequence, TypeVar

import torch
from numpy import ndarray
from pyro.distributions import Uniform
from pyro.distributions.empirical import Empirical
from torch import Tensor, ones, tensor
from torch.distributions import Distribution, Multinomial, MultivariateNormal

from sbi.inference.abc.abc_base import ABCBASE
from sbi.user_input.user_input_checks import process_x


class SMCABC(ABCBASE):
    def __init__(
        self,
        simulator: Callable,
        prior: Distribution,
        distance: Union[str, Callable] = "l2",
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        show_progress_bars: bool = True,
        kernel: Optional[str] = "gaussian",
        algorithm_variant: str = "C",
    ):
        r"""Sequential Monte Carlo Approximate Bayesian Computation.

        We distinguish between three different SMC methods here:
            - A: Toni et al. 2010 (Phd Thesis)
            - B: Sisson et al. 2007 (with correction from 2009)
            - C: Beaumont et al. 2009

        In Toni et al. 2010 we find an overview of the differences on page 34:
            - B: same as A except for resampling of weights if the effective sampling
                size is too small.
            - C: same as A except for calculation of the covariance of the perturbation
                kernel: the kernel covariance is a scaled version of the covariance of
                the previous population.

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
            kernel: Perturbation kernel.
            algorithm_variant: Indicating the choice of algorithm variant, A, B, or C.

        """

        super().__init__(
            simulator=simulator,
            prior=prior,
            distance=distance,
            num_workers=num_workers,
            simulation_batch_size=simulation_batch_size,
            show_progress_bars=show_progress_bars,
        )

        kernels = ("gaussian", "uniform")
        assert (
            kernel in kernels
        ), f"Kernel '{kernel}' not supported. Choose one from {kernels}."
        self.kernel = kernel

        algorithm_variants = ("A", "B", "C")
        assert algorithm_variant in algorithm_variants, (
            f"SMCABC variant '{algorithm_variant}' not supported, choose one from"
            " {algorithm_variants}."
        )
        self.algorithm_variant = algorithm_variant
        self.distance_to_x0 = None
        self.simulation_counter = 0
        self.num_simulations = 0

        self.logger = logging.getLogger(__name__)

        # Define simulator that keeps track of budget.
        def simulate_with_budget(theta):
            self.simulation_counter += theta.shape[0]
            return self._batched_simulator(theta)

        self._simulate_with_budget = simulate_with_budget

    def __call__(
        self,
        x_o: Union[Tensor, ndarray],
        num_particles: int,
        num_initial_pop: int,
        num_simulations: int,
        epsilon_decay: float,
        distance_based_decay: bool = False,
        ess_min: float = 0.5,
        kernel_variance_scale: float = 1.0,
        use_last_pop_samples: bool = True,
        return_summary: bool = False,
    ) -> Union[Distribution, Tuple[Distribution, dict]]:
        r"""Run SMCABC.

        Args:
            x_o: Observed data.
            num_particles: Number of particles in each population.
            num_initial_pop: Number of simulations used for initial population.
            num_simulations: Total number of possible simulations.
            epsilon_decay: Factor with which the acceptance threshold $\epsilon$ decays.
            distance_based_decay: Whether the $\epsilon$ decay is constant over
                populations or calculated from the previous populations distribution of
                distances.
            ess_min: Threshold of effective sampling size for resampling weights.
            kernel_variance_scale: Factor for scaling the perturbation kernel variance.
            use_last_pop_samples: Whether to fill up the current population with
                samples from the previous population when the budget is used up. If
                False, the current population is discarded and the previous population
                is returned.
            return_summary: Whether to return a dictionary with all accepted particles, 
                weights, etc. at the end.

        Returns:
            posterior: Empirical posterior distribution defined by the accepted
                particles and their weights.
            summary (optional): A dictionary containing particles, weights, epsilons
                and distances of each population.
        """

        pop_idx = 0
        self.num_simulations = num_simulations

        # run initial population
        particles, epsilon, distances = self._set_xo_and_sample_initial_population(
            x_o, num_particles, num_initial_pop
        )
        log_weights = torch.log(1 / num_particles * ones(num_particles))

        self.logger.info(
            (
                f"population={pop_idx}, eps={epsilon}, ess={1.0}, "
                f"num_sims={num_initial_pop}"
            )
        )

        all_particles = [particles]
        all_log_weights = [log_weights]
        all_distances = [distances]
        all_epsilons = [epsilon]

        while self.simulation_counter < num_simulations:

            pop_idx += 1
            # Decay based on quantile of distances from previous pop.
            if distance_based_decay:
                epsilon = self._get_next_epsilon(
                    all_distances[pop_idx - 1], epsilon_decay
                )
            # Constant decay.
            else:
                epsilon *= epsilon_decay

            # Get kernel variance from previous pop.
            self.kernel_variance = self.get_kernel_variance(
                all_particles[pop_idx - 1],
                torch.exp(all_log_weights[pop_idx - 1]),
                num_samples=1000,
                kernel_variance_scale=kernel_variance_scale,
            )
            particles, log_weights, distances = self._sample_next_population(
                particles=all_particles[pop_idx - 1],
                log_weights=all_log_weights[pop_idx - 1],
                distances=all_distances[pop_idx - 1],
                epsilon=epsilon,
                use_last_pop_samples=use_last_pop_samples,
            )

            # Resample population if effective sampling size is too small.
            if self.algorithm_variant == "B":
                particles, log_weights = self.resample_if_ess_too_small(
                    particles, log_weights, num_particles, ess_min, pop_idx
                )

            self.logger.info(
                (
                    f"population={pop_idx} done: eps={epsilon:.6f},"
                    f" num_sims={self.simulation_counter}."
                )
            )

            # collect results
            all_particles.append(particles)
            all_log_weights.append(log_weights)
            all_distances.append(distances)
            all_epsilons.append(epsilon)

        posterior = Empirical(all_particles[-1], log_weights=all_log_weights[-1])

        if return_summary:
            return (
                posterior,
                dict(
                    particles=all_particles,
                    weights=all_log_weights,
                    epsilons=all_epsilons,
                    distances=all_distances,
                ),
            )
        else:
            return posterior

    def _set_xo_and_sample_initial_population(
        self, x_o, num_particles: int, num_initial_pop: int,
    ) -> Tuple[Tensor, float, Tensor]:
        """Return particles, epsilon and distances of initial population."""

        assert (
            num_particles <= num_initial_pop
        ), "number of initial round simulations must be greater than population size"

        theta = self.prior.sample((num_initial_pop,))
        x = self._simulate_with_budget(theta)

        # Infer x shape to test and set x_o.
        self.x_shape = x[0].unsqueeze(0).shape
        self.x_o = process_x(x_o, self.x_shape)

        distances = self.distance(self.x_o, x)
        sortidx = torch.argsort(distances)
        particles = theta[sortidx][:num_particles]
        # Take last accepted distance as epsilon.
        initial_epsilon = distances[sortidx][num_particles - 1]

        if not torch.isfinite(initial_epsilon):
            initial_epsilon = 1e8

        return particles, initial_epsilon, distances[sortidx][:num_particles]

    def _sample_next_population(
        self,
        particles: Tensor,
        log_weights: Tensor,
        distances: Tensor,
        epsilon: float,
        use_last_pop_samples: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return particles, weights and distances of new population."""

        new_particles = []
        new_log_weights = []
        new_distances = []

        num_accepted_particles = 0
        num_particles = particles.shape[0]

        while num_accepted_particles < num_particles:

            # Upperbound for batch size to not exceed simulation budget.
            num_batch = min(
                num_particles - num_accepted_particles,
                self.num_simulations - self.simulation_counter,
            )

            # Sample from previous population and perturb.
            particle_candidates = self._sample_and_perturb(
                particles, torch.exp(log_weights), num_samples=num_batch
            )
            # Simulate and select based on distance.
            x = self._simulate_with_budget(particle_candidates)
            dists = self.distance(self.x_o, x)
            is_accepted = dists <= epsilon
            num_accepted_batch = is_accepted.sum().item()

            if num_accepted_batch > 0:
                new_particles.append(particle_candidates[is_accepted])
                new_log_weights.append(
                    self._calculate_new_log_weights(
                        particle_candidates[is_accepted], particles, log_weights,
                    )
                )
                new_distances.append(dists[is_accepted])
                num_accepted_particles += num_accepted_batch

            # If simulation budget was exceeded and we still need particles, take
            # previous population or fill up with previous population.
            if (
                self.simulation_counter >= self.num_simulations
                and num_accepted_particles < num_particles
            ):
                if use_last_pop_samples:
                    num_remaining = num_particles - num_accepted_particles
                    self.logger.info(
                        f"""Simulation Budget exceeded, filling up with {num_remaining}
                        samples from last population."""
                    )
                    # Some new particles have been accepted already, therefore
                    # fill up the remaining once with old particles and weights.
                    new_particles.append(particles[:num_remaining, :])
                    # Recalculate weights with new particles.
                    new_log_weights = [
                        self._calculate_new_log_weights(
                            torch.cat(new_particles), particles, log_weights,
                        )
                    ]
                    new_distances.append(distances[:num_remaining])
                else:
                    self.logger.info(
                        "Simulation Budget exceeded, returning previous population."
                    )
                    new_particles = [particles]
                    new_log_weights = [log_weights]
                    new_distances = [distances]

                break

        # collect lists of tensors into tensors
        new_particles = torch.cat(new_particles)
        new_log_weights = torch.cat(new_log_weights)
        new_distances = torch.cat(new_distances)

        # normalize the new weights
        new_log_weights -= torch.logsumexp(new_log_weights, dim=0)

        return new_particles, new_log_weights, new_distances

    def _get_next_epsilon(self, distances: Tensor, quantile: float) -> float:
        """Return epsilon for next round based on quantile of this round's distances.

        Note: distances are made unique to avoid repeated distances from simulations
        that result in the same observation.

        Args:
            distances: The distances accepted in this round.
            quantile: Quantile in the distance distribution to determine new epsilon.

        Returns:
            epsilon: Epsilon for the next population.
        """
        # Take unique distances to skip same distances simulations (return is sorted).
        distances = torch.unique(distances)
        # Cumsum as cdf proxy.
        distances_cdf = torch.cumsum(distances, dim=0) / distances.sum()
        # Take the q quantile of distances.
        try:
            qidx = torch.where(distances_cdf >= quantile)[0][0]
        except IndexError:
            self.logger.warning(
                (
                    f"Accepted unique distances={distances} don't match "
                    f"quantile={quantile:.2f}. Selecting last distance."
                )
            )
            qidx = -1

        # The new epsilon is given by that distance.
        return distances[qidx].item()

    def _calculate_new_log_weights(
        self, new_particles: Tensor, old_particles: Tensor, old_log_weights: Tensor,
    ) -> Tensor:
        """Return new log weights following formulas in publications A,B anc C."""

        # Prior can be batched across new particles.
        prior_log_probs = self.prior.log_prob(new_particles)

        # Contstruct function to get kernel log prob for given old particle.
        # The kernel is centered on each old particle as in all three variants (A,B,C).
        def kernel_log_prob(new_particle):
            return self.get_new_kernel(old_particles).log_prob(new_particle)

        # We still have to loop over particles here because
        # the kernel log probs are already batched across old particles.
        log_weighted_sum = tensor(
            [
                torch.logsumexp(old_log_weights + kernel_log_prob(new_particle), dim=0)
                for new_particle in new_particles
            ],
            dtype=torch.float32,
        )
        # new weights are prior probs over weighted sum:
        return prior_log_probs - log_weighted_sum

    @staticmethod
    def sample_from_population_with_weights(
        particles: Tensor, weights: Tensor, num_samples: int = 1
    ) -> Tensor:
        """Return samples from particles sampled with weights."""

        # define multinomial with weights as probs
        multi = Multinomial(probs=weights)
        # sample num samples, with replacement
        samples = multi.sample(sample_shape=(num_samples,))
        # get indices of success trials
        indices = torch.where(samples)[1]
        # return those indices from trace
        return particles[indices]

    def _sample_and_perturb(
        self, particles: Tensor, weights: Tensor, num_samples: int = 1
    ) -> Tensor:
        """Sample and perturb batch of new parameters from trace.

        Reject sampled and perturbed parameters outside of prior.
        """

        num_accepted = 0
        parameters = []
        while num_accepted < num_samples:
            parms = self.sample_from_population_with_weights(
                particles, weights, num_samples=num_samples - num_accepted
            )

            # Create kernel on params and perturb.
            parms_perturbed = self.get_new_kernel(parms).sample()

            is_within_prior = torch.isfinite(self.prior.log_prob(parms_perturbed))
            num_accepted += is_within_prior.sum().item()

            if num_accepted > 0:
                parameters.append(parms_perturbed[is_within_prior])

        return torch.cat(parameters)

    def get_kernel_variance(
        self,
        particles: Tensor,
        weights: Tensor,
        num_samples: int = 1000,
        kernel_variance_scale: float = 1.0,
    ) -> Tensor:

        # get weighted samples
        samples = self.sample_from_population_with_weights(
            particles, weights, num_samples=num_samples
        )

        if self.kernel == "gaussian":
            # For variant C, Beaumont et al. 2009, the kernel variance comes from the
            # previous population.
            if self.algorithm_variant == "C":
                mean = torch.mean(samples, dim=0).unsqueeze(0)

                # take double the weighted sample cov as proposed in Beaumont 2009
                population_cov = torch.matmul(samples.T, samples) / (
                    num_samples - 1
                ) - torch.matmul(mean.T, mean)
                return kernel_variance_scale * population_cov
            # While for Toni et al. and Sisson et al. it comes from the parameter
            # ranges.
            elif self.algorithm_variant in ("A", "B"):
                # Variance spans the range of parameters for every dimension.
                parameter_ranges = tensor(
                    [
                        max(theta_column) - min(theta_column)
                        for theta_column in samples.T
                    ]
                )
                assert parameter_ranges.ndim < 2
                return kernel_variance_scale * torch.diag(parameter_ranges)
            else:
                raise ValueError(f"Variant, '{self.algorithm_variant}' not supported.")

        elif self.kernel == "uniform":
            # Variance spans the range of parameters for every dimension.
            return kernel_variance_scale * tensor(
                [max(theta_column) - min(theta_column) for theta_column in samples.T]
            )
        else:
            raise ValueError(f"Kernel, '{self.kernel}' not supported.")

    def get_new_kernel(self, thetas: Tensor) -> Distribution:
        """Return new kernel distribution for a given set of paramters."""

        if self.kernel == "gaussian":
            assert self.kernel_variance.ndim == 2
            return MultivariateNormal(
                loc=thetas, covariance_matrix=self.kernel_variance
            )

        elif self.kernel == "uniform":
            low = thetas - self.kernel_variance
            high = thetas + self.kernel_variance
            # Move batch shape to event shape to get Uniform that is multivariate in
            # parameter dimension.
            return Uniform(low=low, high=high).to_event(1)
        else:
            raise ValueError(f"Kernel, '{self.kernel}' not supported.")

    def resample_if_ess_too_small(
        self,
        particles: Tensor,
        log_weights: Tensor,
        num_particles: int,
        ess_min: float,
        pop_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        """Return resampled particles and uniform weights if effectice sampling size is
        too small.
        """

        ess = (1 / torch.sum(torch.exp(2.0 * log_weights), dim=0)) / num_particles
        # Resampling of weights for low ESS only for Sisson et al. 2007.
        if ess < ess_min:
            self.logger.info(f"ESS={ess:.2f} too low, resampling pop {pop_idx}...")
            # First resample, then set to uniform weights in in Sisson et al. 2007.
            particles = self.sample_from_population_with_weights(
                particles, torch.exp(log_weights), num_samples=num_particles
            )
            log_weights = torch.log(1 / num_particles * ones(num_particles))

        return particles, log_weights
