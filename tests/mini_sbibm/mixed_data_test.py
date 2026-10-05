# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from pyro.distributions import InverseGamma
from torch.distributions import Bernoulli, Beta, Gamma

from sbi.utils.user_input_checks_utils import MultipleIndependent

from .mixed_data import MixedData


@pytest.mark.parametrize("num_trials", [1, 4, 10])
@pytest.mark.parametrize("custom_prior", [False, True])
def test_reference_density_matches_simulator_likelihood(num_trials, custom_prior):
    """The normalized reference and simulator joint differ only by evidence."""
    task = MixedData(num_trials=10)
    task.stimulus_condition = 3.0
    if custom_prior:
        task.get_prior = lambda: MultipleIndependent(
            [
                Gamma(torch.tensor([2.0]), torch.tensor([0.8])),
                Beta(torch.tensor([3.0]), torch.tensor([4.0])),
            ],
            validate_args=False,
        )
    observation = task.get_observation(3)[:num_trials]
    theta = torch.tensor([[0.4, 0.2], [1.1, 0.5], [3.0, 0.8]])
    rate, choice = task._get_reference_posterior(observation)
    reference_log_prob = rate.log_prob(theta[:, 0]) + choice.log_prob(theta[:, 1])

    likelihood = InverseGamma(task.stimulus_condition, theta[:, :1]).log_prob(
        observation[:, 0]
    )
    likelihood += Bernoulli(probs=theta[:, 1:]).log_prob(observation[:, 1])
    joint_log_prob = likelihood.sum(dim=1) + task.get_prior().log_prob(theta)

    torch.testing.assert_close(
        reference_log_prob - reference_log_prob[0],
        joint_log_prob - joint_log_prob[0],
        atol=1e-4,
        rtol=1e-5,
    )
