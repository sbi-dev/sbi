# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch

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
