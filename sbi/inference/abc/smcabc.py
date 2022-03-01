from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from pyro.distributions import Uniform
from torch import Tensor, ones, tensor
from torch.distributions import Distribution, Multinomial, MultivariateNormal

from sbi.inference.abc.abc_base import ABCBASE
from sbi.types import Array
from sbi.utils import KDEWrapper, get_kde, process_x, within_support


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
        ess_min: Optional[float] = None,
        kernel_variance_scale: float = 1.0,
        use_last_pop_samples: bool = True,
        return_summary: bool = False,
        kde: bool = False,
        kde_kwargs: Dict[str, Any] = {},
        kde_sample_weights: bool = False,
        lra: bool = False,
        lra_with_weights: bool = False,
        sass: bool = False,
        sass_fraction: float = 0.25,
        sass_expansion_degree: int = 1,
    ) -> Union[Tensor, KDEWrapper, Tuple[Tensor, dict], Tuple[KDEWrapper, dict]]:
        r"""Run SMCABC and return accepted parameters or KDE object fitted on them.

        Args:
            x_o: Observed data.
            num_particles: Number of particles in each population.
            num_initial_pop: Number of simulations used for initial population.
            num_simulations: Total number of possible simulations.
            epsilon_decay: Factor with which the acceptance threshold $\epsilon$ decays.
            distance_based_decay: Whether the $\epsilon$ decay is constant over
                populations or calculated from the previous populations distribution of
                distances.
            ess_min: Threshold of effective sampling size for resampling weights. Not
                used when None (default).
            kernel_variance_scale: Factor for scaling the perturbation kernel variance.
            use_last_pop_samples: Whether to fill up the current population with
                samples from the previous population when the budget is used up. If
                False, the current population is discarded and the previous population
                is returned.
            lra: Whether to run linear regression adjustment as in Beaumont et al. 2002
            lra_with_weights: Whether to run lra as weighted linear regression with SMC
                weights
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
            kde_sample_weights: Whether perform weighted KDE with SMC weights or on raw
                particles.
            return_summary: Whether to return a dictionary with all accepted particles,
                weights, etc. at the end.

        Returns:
            theta (if kde False): accepted parameters of the last population.
            kde (if kde True): KDE object fitted on accepted parameters, from which one
                can .sample() and .log_prob().
            summary (if return_summary True): dictionary containing the accepted
                paramters (if kde True), distances and simulated data x of all
                populations.
        """

        pop_idx = 0
        self.num_simulations = num_simulations

        # Pilot run for SASS.
        if sass:
            num_pilot_simulations = int(sass_fraction * num_simulations)
            self.logger.info(
                f"Running SASS with {num_pilot_simulations} pilot samples."
            )
            sass_transform = self.run_sass_set_xo(
                num_particles, num_pilot_simulations, x_o, lra, sass_expansion_degree
            )
            # Udpate simulator and xo
            x_o = sass_transform(self.x_o)

            def sass_simulator(theta):
                self.simulation_counter += theta.shape[0]
                return sass_transform(self._batched_simulator(theta))

            self._simulate_with_budget = sass_simulator

        # run initial population
        particles, epsilon, distances, x = self._set_xo_and_sample_initial_population(
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
        all_x = [x]

        while self.simulation_counter < self.num_simulations:

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
                samples_per_dim=500,
                kernel_variance_scale=kernel_variance_scale,
            )
            particles, log_weights, distances, x = self._sample_next_population(
                particles=all_particles[pop_idx - 1],
                log_weights=all_log_weights[pop_idx - 1],
                distances=all_distances[pop_idx - 1],
                epsilon=epsilon,
                x=all_x[pop_idx - 1],
                use_last_pop_samples=use_last_pop_samples,
            )

            # Resample population if effective sampling size is too small.
            if ess_min is not None:
                particles, log_weights = self.resample_if_ess_too_small(
                    particles, log_weights, ess_min, pop_idx
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
            all_x.append(x)

        # Maybe run LRA and adjust weights.
        if lra:
            self.logger.info("Running Linear regression adjustment.")
            adjusted_particles, adjusted_weights = self.run_lra_update_weights(
                particles=all_particles[-1],
                xs=all_x[-1],
                observation=process_x(x_o),
                log_weights=all_log_weights[-1],
                lra_with_weights=lra_with_weights,
            )
            final_particles = adjusted_particles
        else:
            final_particles = all_particles[-1]

        if kde:
            self.logger.info(
                f"""KDE on {final_particles.shape[0]} samples with bandwidth option
                {kde_kwargs["bandwidth"] if "bandwidth" in kde_kwargs else "cv"}.
                Beware that KDE can give unreliable results when used with too few
                samples and in high dimensions."""
            )
            # Maybe get particles weights from last population for weighted KDE.
            if kde_sample_weights:
                kde_kwargs["sample_weights"] = all_log_weights[-1].exp()

            kde_dist = get_kde(final_particles, **kde_kwargs)

            if return_summary:
                return (
                    kde_dist,
                    dict(
                        particles=all_particles,
                        weights=all_log_weights,
                        epsilons=all_epsilons,
                        distances=all_distances,
                        xs=all_x,
                    ),
                )
            else:
                return kde_dist

        if return_summary:
            return (
                final_particles,
                dict(
                    particles=all_particles,
                    weights=all_log_weights,
                    epsilons=all_epsilons,
                    distances=all_distances,
                    xs=all_x,
                ),
            )
        else:
            return final_particles

    def _set_xo_and_sample_initial_population(
        self,
        x_o: Array,
        num_particles: int,
        num_initial_pop: int,
    ) -> Tuple[Tensor, float, Tensor, Tensor]:
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

        return (
            particles,
            initial_epsilon,
            distances[sortidx][:num_particles],
            x[sortidx][:num_particles],
        )

    def _sample_next_population(
        self,
        particles: Tensor,
        log_weights: Tensor,
        distances: Tensor,
        epsilon: float,
        x: Tensor,
        use_last_pop_samples: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return particles, weights and distances of new population."""

        new_particles = []
        new_log_weights = []
        new_distances = []
        new_x = []

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
            x_candidates = self._simulate_with_budget(particle_candidates)
            dists = self.distance(self.x_o, x_candidates)
            is_accepted = dists <= epsilon
            num_accepted_batch = is_accepted.sum().item()

            if num_accepted_batch > 0:
                new_particles.append(particle_candidates[is_accepted])
                new_log_weights.append(
                    self._calculate_new_log_weights(
                        particle_candidates[is_accepted],
                        particles,
                        log_weights,
                    )
                )
                new_distances.append(dists[is_accepted])
                new_x.append(x_candidates[is_accepted])
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
                            torch.cat(new_particles),
                            particles,
                            log_weights,
                        )
                    ]
                    new_distances.append(distances[:num_remaining])
                    new_x.append(x[:num_remaining])
                else:
                    self.logger.info(
                        "Simulation Budget exceeded, returning previous population."
                    )
                    new_particles = [particles]
                    new_log_weights = [log_weights]
                    new_distances = [distances]
                    new_x = [x]

                break

        # collect lists of tensors into tensors
        new_particles = torch.cat(new_particles)
        new_log_weights = torch.cat(new_log_weights)
        new_distances = torch.cat(new_distances)
        new_x = torch.cat(new_x)

        # normalize the new weights
        new_log_weights -= torch.logsumexp(new_log_weights, dim=0)

        # Return sorted wrt distances.
        sort_idx = torch.argsort(new_distances)

        return (
            new_particles[sort_idx],
            new_log_weights[sort_idx],
            new_distances[sort_idx],
            new_x[sort_idx],
        )

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
        self,
        new_particles: Tensor,
        old_particles: Tensor,
        old_log_weights: Tensor,
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
        samples = multi.sample(sample_shape=torch.Size((num_samples,)))
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

            is_within_prior = within_support(self.prior, parms_perturbed)
            num_accepted += int(is_within_prior.sum().item())

            if num_accepted > 0:
                parameters.append(parms_perturbed[is_within_prior])

        return torch.cat(parameters)

    def get_kernel_variance(
        self,
        particles: Tensor,
        weights: Tensor,
        samples_per_dim: int = 100,
        kernel_variance_scale: float = 1.0,
    ) -> Tensor:

        if self.kernel == "gaussian":
            # For variant C, Beaumont et al. 2009, the kernel variance comes from the
            # previous population.
            if self.algorithm_variant == "C":
                # Calculate weighted covariance of particles.
                population_cov = torch.tensor(
                    np.atleast_2d(np.cov(particles, rowvar=False, aweights=weights)),
                    dtype=torch.float32,
                )
                # Make sure variance is nonsingular.
                try:
                    torch.cholesky(kernel_variance_scale * population_cov)
                except RuntimeError:
                    self.logger.warning(
                        """"Singular particle covariance, using unit covariance."""
                    )
                    population_cov = torch.eye(particles.shape[1])
                return kernel_variance_scale * population_cov
            # While for Toni et al. and Sisson et al. it comes from the parameter
            # ranges.
            elif self.algorithm_variant in ("A", "B"):
                particle_ranges = self.get_particle_ranges(
                    particles, weights, samples_per_dim=samples_per_dim
                )
                return kernel_variance_scale * torch.diag(particle_ranges)
            else:
                raise ValueError(f"Variant, '{self.algorithm_variant}' not supported.")
        elif self.kernel == "uniform":
            # Variance spans the range of parameters for every dimension.
            return kernel_variance_scale * self.get_particle_ranges(
                particles, weights, samples_per_dim=samples_per_dim
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
        ess_min: float,
        pop_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        """Return resampled particles and uniform weights if effectice sampling size is
        too small.
        """

        num_particles = particles.shape[0]
        ess = (1 / torch.sum(torch.exp(2.0 * log_weights), dim=0)) / num_particles
        # Resampling of weights for low ESS only for Sisson et al. 2007.
        if ess < ess_min:
            self.logger.info(f"ESS={ess:.2f} too low, resampling pop {pop_idx}...")
            # First resample, then set to uniform weights as in Sisson et al. 2007.
            particles = self.sample_from_population_with_weights(
                particles, torch.exp(log_weights), num_samples=num_particles
            )
            log_weights = torch.log(1 / num_particles * ones(num_particles))

        return particles, log_weights

    def run_lra_update_weights(
        self,
        particles: Tensor,
        xs: Tensor,
        observation: Tensor,
        log_weights: Tensor,
        lra_with_weights: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Return particles and weights adjusted with LRA.

        Runs (weighted) linear regression from xs onto particles to adjust the
        particles.

        Updates the SMC weights according to the new particles.
        """

        adjusted_particels = self.run_lra(
            theta=particles,
            x=xs,
            observation=observation,
            sample_weight=log_weights.exp() if lra_with_weights else None,
        )

        # Update SMC weights with LRA adjusted weights
        adjusted_log_weights = self._calculate_new_log_weights(
            new_particles=adjusted_particels,
            old_particles=particles,
            old_log_weights=log_weights,
        )

        return adjusted_particels, adjusted_log_weights

    def run_sass_set_xo(
        self,
        num_particles: int,
        num_pilot_simulations: int,
        x_o,
        lra: bool = False,
        sass_expansion_degree: int = 1,
    ) -> Callable:
        """Return transform for semi-automatic summary statistics.

        Runs an single round of rejection abc with fixed budget and accepts
        num_particles simulations to run the regression for sass.

        Sets self.x_o once the x_shape can be derived from simulations.
        """
        (pilot_particles, _, _, pilot_xs,) = self._set_xo_and_sample_initial_population(
            x_o, num_particles, num_pilot_simulations
        )
        # Adjust with LRA.
        if lra:
            pilot_particles = self.run_lra(pilot_particles, pilot_xs, self.x_o)
        sass_transform = self.get_sass_transform(
            pilot_particles,
            pilot_xs,
            expansion_degree=sass_expansion_degree,
            sample_weight=None,
        )
        return sass_transform

    def get_particle_ranges(
        self, particles: Tensor, weights: Tensor, samples_per_dim: int = 100
    ) -> Tensor:
        """Return range of particles in each parameter dimension."""

        # get weighted samples
        samples = self.sample_from_population_with_weights(
            particles,
            weights,
            num_samples=samples_per_dim * particles.shape[1],
        )

        # Variance spans the range of particles for every dimension.
        particle_ranges = samples.max(0).values - samples.min(0).values
        assert particle_ranges.ndim < 2
        return particle_ranges
