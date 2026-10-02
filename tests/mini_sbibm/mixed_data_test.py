# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from pyro.distributions import InverseGamma
from torch.distributions import Bernoulli

from .mixed_data import MixedData


def test_mixed_data_task_shapes():
    """The task separates simulation batches from IID evaluation trials."""
    task = MixedData(num_trials=4)

    theta, x = task.get_data(5)

    assert theta.shape == (5, 2)
    assert x.shape == (5, 2)
    assert task.get_observation(1).shape == (4, 2)
    assert task.get_reference_posterior_samples(1).shape == (10_000, 2)


def test_mixed_data_reference_posterior_parameters():
    """The exact posterior uses Gamma and Beta conjugate updates."""
    task = MixedData(num_trials=2)
    observation = torch.tensor([[1.0, 1.0], [2.0, 0.0]])

    rate_posterior, choice_posterior = task._get_reference_posterior(observation)

    assert torch.equal(rate_posterior.concentration, torch.tensor([5.0]))
    assert torch.equal(rate_posterior.rate, torch.tensor([2.0]))
    assert torch.equal(choice_posterior.concentration1, torch.tensor([3.0]))
    assert torch.equal(choice_posterior.concentration0, torch.tensor([3.0]))


def test_mixed_data_rejects_nonpositive_trial_count():
    """Evaluation observations need at least one trial."""
    with pytest.raises(ValueError, match="num_trials must be at least one"):
        MixedData(num_trials=0)


@pytest.mark.parametrize("num_trials", [1, 4, 10])
def test_reference_density_matches_simulator_likelihood(num_trials):
    """The normalized reference and simulator joint differ only by evidence."""
    task = MixedData(num_trials=num_trials)
    observation = task.get_observation(3)
    theta = torch.tensor([[0.4, 0.2], [1.1, 0.5], [3.0, 0.8]])
    rate, choice = task._get_reference_posterior(observation)
    reference_log_prob = rate.log_prob(theta[:, 0]) + choice.log_prob(theta[:, 1])

    likelihood = InverseGamma(2.0, theta[:, :1]).log_prob(observation[:, 0])
    likelihood += Bernoulli(probs=theta[:, 1:]).log_prob(observation[:, 1])
    joint_log_prob = likelihood.sum(dim=1) + task.get_prior().log_prob(theta)

    torch.testing.assert_close(
        reference_log_prob - reference_log_prob[0],
        joint_log_prob - joint_log_prob[0],
        atol=1e-4,
        rtol=1e-5,
    )
