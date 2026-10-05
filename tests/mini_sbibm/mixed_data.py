# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Union

import torch
from pyro.distributions import InverseGamma
from torch import Tensor
from torch.distributions import Beta, Binomial, Distribution, Gamma

from sbi.utils.user_input_checks_utils import MultipleIndependent

from .base_task import Task


def mixed_simulator(
    theta: Tensor, stimulus_condition: Union[Tensor, float] = 2.0
) -> Tensor:
    """Simulate reaction times and binary choices for mixed observations."""
    rate, choice_probability = theta[:, :1], theta[:, 1:]
    choices = Binomial(probs=choice_probability).sample()
    reaction_times = InverseGamma(
        concentration=stimulus_condition * torch.ones_like(rate), rate=rate
    ).sample()
    return torch.cat((reaction_times, choices), dim=1)


class MixedData(Task):
    """Mixed data task with an exact conjugate reference posterior."""

    def __init__(self, num_trials: int = 10):
        """Initialize the task.

        Args:
            num_trials: Number of independent trials in each evaluation observation.
        """
        if num_trials < 1:
            raise ValueError("num_trials must be at least one.")
        self.num_trials = num_trials
        self.stimulus_condition = 2.0
        super().__init__(f"mixed_data-{num_trials}trials")

    def theta_dim(self) -> int:
        """Return the parameter dimensionality."""
        return 2

    def x_dim(self) -> int:
        """Return the single trial observation dimensionality."""
        return 2

    def get_prior(self) -> Distribution:
        """Return the Gamma and Beta prior used by the MNLE tests."""
        return MultipleIndependent(
            [
                Gamma(torch.tensor([1.0]), torch.tensor([0.5])),
                Beta(torch.tensor([2.0]), torch.tensor([2.0])),
            ],
            validate_args=False,
        )

    def get_simulator(self) -> Callable:
        """Return the mixed data simulator."""
        return mixed_simulator

    def get_true_parameters(self, idx: int) -> Tensor:
        """Generate reproducible true parameters for an observation."""
        torch.manual_seed(idx)
        return self.get_prior().sample()

    def get_observation(self, idx: int) -> Tensor:
        """Generate independent trials at fixed true parameters."""
        theta = self.get_true_parameters(idx).repeat(self.num_trials, 1)
        return mixed_simulator(theta, self.stimulus_condition)

    def get_reference_posterior_samples(self, idx: int) -> Tensor:
        """Draw samples from the exact Gamma and Beta posterior."""
        observation = self.get_observation(idx)
        rate_posterior, choice_posterior = self._get_reference_posterior(observation)
        return torch.cat(
            (rate_posterior.sample((10_000,)), choice_posterior.sample((10_000,))),
            dim=1,
        )

    def _get_reference_posterior(self, observation: Tensor) -> tuple[Gamma, Beta]:
        """Return the conjugate posterior distributions for an observation."""
        reaction_times = observation[:, :1]
        choices = observation[:, 1:]
        num_trials = observation.shape[0]
        rate_prior, choice_prior = self.get_prior().dists

        rate_posterior = Gamma(
            rate_prior.concentration + self.stimulus_condition * num_trials,
            rate_prior.rate + torch.sum(reaction_times.reciprocal(), dim=0),
        )
        choice_posterior = Beta(
            choice_prior.concentration1 + torch.sum(choices, dim=0),
            choice_prior.concentration0 + num_trials - torch.sum(choices, dim=0),
        )
        return rate_posterior, choice_posterior
