from __future__ import annotations

from pyro.distributions.empirical import Empirical
from sbi.inference.abc.abc_base import ABCBASE
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor, ones, tensor
from torch.distributions import Distribution, Multinomial, MultivariateNormal
from pyro.distributions import Uniform
from numpy import ndarray
import logging


class SMCABC(ABCBASE):
    """ABC PMC as in Beaumont 2002 and 2009."""

    def __init__(
        self,
        simulator: Callable,
        prior: Distribution,
        x_o: Union[Tensor, ndarray],
        distance: Union[str, Callable] = "l2",
        kernel: Optional[str] = "gaussian",
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        show_progress_bars: bool = True,
    ):

        super().__init__(
            simulator=simulator,
            prior=prior,
            x_o=x_o,
            distance=distance,
            num_workers=num_workers,
            simulation_batch_size=simulation_batch_size,
            show_progress_bars=show_progress_bars,
        )

        assert kernel in ("gaussian", "uniform"), f"Kernel '{kernel}' not supported."

        self.distance_to_x0 = lambda x: self.distance(self.x_o, x)
        self.num_simulations = 0
        self.num_simulation_budget = 0
        self.kernel = kernel

        self.logger = logging.getLogger(__name__)

        # Define simulator that keeps track of budget.
        def simulate_with_budget(theta):
            self.num_simulations += theta.shape[0]
            return self._batched_simulator(theta)

        self._simulate_with_budget = simulate_with_budget

    def __call__(
        self,
        num_particles: int,
        num_initial_pop: int,
        epsilon_decay: float,
        num_simulation_budget: int,
        batch_size: int = 100,
        qt_decay: bool = False,
        ess_min: Optional[float] = None,
        kernel_variance_scale: float = 1.0,
        use_last_pop_samples: bool = True,
        return_summary: bool = False,
    ):
        pop_idx = 0
        self.num_simulation_budget = num_simulation_budget

        # run initial population
        particles, epsilon, distances = self._sample_initial_population(
            num_particles, num_initial_pop
        )
        log_weights = torch.log(1 / num_particles * ones(num_particles))

        self.logger.info(
            (
                f"population={pop_idx}, eps={epsilon}, ess={1.0}, "
                "num_sims={num_initial_pop}"
            )
        )

        all_particles = [particles]
        all_log_weights = [log_weights]
        all_distances = [distances]
        all_epsilons = [epsilon]

        while self.num_simulations < num_simulation_budget:

            pop_idx += 1
            if qt_decay:
                epsilon = self._get_next_epsilon(
                    all_distances[pop_idx - 1], epsilon_decay
                )
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

            # Resample weights if ess too low.
            ess = (1 / torch.sum(torch.exp(2.0 * log_weights), dim=0)) / num_particles
            if ess_min is not None:
                if ess < ess_min:
                    self.logger.info(
                        f"ESS={ess:.2f} too low, resampling pop {pop_idx}..."
                    )
                    particles = self.sample_from_population_with_weights(
                        particles, torch.exp(log_weights), num_samples=num_particles
                    )
                    log_weights = torch.log(1 / num_particles * ones(num_particles))

            self.logger.info(
                (
                    f"population={pop_idx} done: eps={epsilon:.6f}, ess={ess:.2f},"
                    " num_sims={self.num_simulations}, acc={acc:.4f}."
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

    def _sample_initial_population(
        self, num_particles: int, num_initial_pop: int,
    ) -> Tuple[Tensor, float, Tensor]:

        assert (
            num_particles <= num_initial_pop
        ), "number of initial round simulations must be greater than population size"

        theta = self.prior.sample((num_initial_pop,))
        x = self._simulate_with_budget(theta)
        distances = self.distance_to_x0(x)
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

        # new_particles = zeros_like(particles)
        # new_log_weights = zeros_like(log_weights)
        new_particles = []
        new_log_weights = []
        new_distances = []

        num_accepted_particles = 0
        num_particles = particles.shape[0]

        while num_accepted_particles < num_particles:

            # Upperbound for batch size to not exceed simulation budget.
            num_batch = min(
                num_particles - num_accepted_particles,
                self.num_simulation_budget - self.num_simulations,
            )

            # Sample from previous population and perturb.
            particle_candidates = self._sample_and_perturb(
                particles, torch.exp(log_weights), num_samples=num_batch
            )
            # Simulate and select based on distance.
            x = self._simulate_with_budget(particle_candidates)
            dists = self.distance_to_x0(x)
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
                self.num_simulations >= self.num_simulation_budget
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

        Arguments:
            distances  -- The distances accepted in this round.
            quantile -- quantile in the distance distribution to determine new epsilon

        Returns:
            epsilon -- epsilon for the next population.
        """
        # Make distances unique to skip simulations with same outcome.
        # NOTE: unique sorts the input already, so no sorting needed below?
        distances = torch.unique(distances)
        distances = distances[torch.argsort(distances)]
        # Cumsum as cdf proxy.
        distances_cdf = torch.cumsum(distances, dim=0) / distances.sum()
        # Take the q quantile of distances.
        try:
            qidx = torch.where(distances_cdf >= quantile)[0][0]
        except IndexError:
            self.logger.warning(
                (
                    f"Accepted unique distances={distances} dont match "
                    "quantile={quantile:.2f}. Selecting last distance."
                )
            )
            qidx = -1

        # The new epsilon is given by that distance.
        return distances[qidx].item()

    def _calculate_new_log_weights(
        self, new_particles: Tensor, old_particles: Tensor, log_weights: Tensor,
    ) -> Tensor:

        # Prior can be batched across new particles.
        prior_log_probs = self.prior.log_prob(new_particles)

        # Contstruct function to get kernel log prob for given new particle.
        # The kernel is centered on each new particle.
        def kernel_log_prob(new_particle):
            return self.get_new_kernel(new_particle).log_prob(old_particles)

        # We still have to loop over particles here because
        # the kernel log probs are already batched across old particles.
        log_weighted_sum = tensor(
            [
                torch.logsumexp(log_weights + kernel_log_prob(new_particle), dim=0)
                for new_particle in new_particles
            ],
            dtype=torch.float32,
        )
        # new weights are prior probs over weighted sum:
        return prior_log_probs - log_weighted_sum

    @staticmethod
    def sample_from_population_with_weights(
        particles: Tensor, weights: Tensor, num_samples: int = 1
    ):
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
    ):
        """Sample and perturb batch of new parameters from trace.

        Reject sampled and perturbed parameters outside of prior.

        Args:
            trace {Tensor} -- [description]
            weights {Tensor} -- [description]

        Kwargs:
            num_samples {int} -- [description] (default: {1})
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
            mean = torch.mean(samples, dim=0).unsqueeze(0)

            # take double the weighted sample cov as proposed in Beaumont 2009
            population_cov = torch.matmul(samples.T, samples) / (
                num_samples - 1
            ) - torch.matmul(mean.T, mean)

            return kernel_variance_scale * population_cov

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
