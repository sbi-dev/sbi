import logging
from typing import Callable

import pyro
import torch
from pyro import distributions as pdist


class GaussianMixture:
    def __init__(
        self,
        dim: int = 2,
        prior_bound: float = 10.0,
    ):
        """Gaussian Mixture as implemented in `sbim` [1].

        Inference of mean under uniform prior.

        Args:
            dim: Dimensionality of parameters and data.
            prior_bound: Prior is uniform in [-prior_bound, +prior_bound].

        References:
        [1]: https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/gaussian_mixture/task.py
        """
        self.dim_parameters = dim
        self.prior_params = {
            "low": -prior_bound * torch.ones((self.dim_parameters,)),
            "high": +prior_bound * torch.ones((self.dim_parameters,)),
        }

        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

        self.simulator_params = {
            "mixture_locs_factor": torch.tensor([1.0, 1.0]),
            "mixture_scales": torch.tensor([1.0, 0.1]),
            "mixture_weights": torch.tensor([0.5, 0.5]),
        }

    def get_simulator(self) -> Callable:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters):
            # Sample mixture index for each parameter in batch
            idx = pyro.sample(
                "mixture_idx",
                pdist.Categorical(
                    probs=self.simulator_params["mixture_weights"]
                ).expand_by([parameters.shape[0]]),
            ).unsqueeze(1)

            # Select loc and scales according to mixture index
            loc = self.simulator_params["mixture_locs_factor"][idx] * parameters
            scale = self.simulator_params["mixture_scales"][idx]

            return pyro.sample("data", pdist.Normal(loc=loc, scale=scale).to_event(1))

        return simulator

    def _sample_reference_posterior(
        self,
        num_samples: int,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Uses closed form solution with rejection sampling

        Args:
            num_samples: Number of samples to generate
            observation: Observation.

        Returns:
            Samples from reference posterior
        """

        log = logging.getLogger(__name__)

        reference_posterior_samples = []

        # Reject samples outside of prior bounds
        counter = 0
        while len(reference_posterior_samples) < num_samples:
            counter += 1
            idx = pyro.sample(
                "mixture_idx",
                pdist.Categorical(self.simulator_params["mixture_weights"]),
            )
            sample = pyro.sample(
                "posterior_sample",
                pdist.Normal(
                    loc=self.simulator_params["mixture_locs_factor"][idx] * observation,
                    scale=self.simulator_params["mixture_scales"][idx],
                ),
            )
            is_outside_prior = torch.isinf(self.prior_dist.log_prob(sample).sum())

            if len(reference_posterior_samples) > 0:
                is_duplicate = sample in torch.cat(reference_posterior_samples)
            else:
                is_duplicate = False

            if not is_outside_prior and not is_duplicate:
                reference_posterior_samples.append(sample)

        reference_posterior_samples = torch.cat(reference_posterior_samples)
        acceptance_rate = float(num_samples / counter)

        log.info(f"Acceptance rate: {acceptance_rate}")

        return reference_posterior_samples
