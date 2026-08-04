# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Callable, Union

import pytest
import torch
from torch.distributions import (
    Categorical,
    MixtureSameFamily,
    MultivariateNormal,
    Normal,
)

from sbi.inference.trainers.marginal import MarginalTrainer
from sbi.neural_nets.factory import ZukoFlowType, marginal_nn
from sbi.utils.metrics import check_c2st
from sbi.utils.torchutils import process_device


@pytest.mark.parametrize(
    "dist",
    [
        MultivariateNormal(
            loc=torch.tensor([2.0, 3.0]),
            covariance_matrix=torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
        ),
        MixtureSameFamily(
            Categorical(torch.ones(2)),
            Normal(torch.randn(2), torch.rand(2)),
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("model", ["nsf", marginal_nn(model=ZukoFlowType.NSF)])
def test_marginal_estimator(
    dist: torch.distributions.Distribution, device: str, model: Union[str, Callable]
):
    """Test the marginal estimator with various distributions and devices."""
    num_training_samples = 2_000
    num_test_samples = 1_000
    device = process_device(device)

    # Generate samples from the true distribution
    x_train = dist.sample((num_training_samples,))
    if len(x_train.shape) == 1:
        x_train = x_train.unsqueeze(1)

    # Instantiate a trainer for the marginal pdf and train it
    trainer = MarginalTrainer(density_estimator=model, device=device)
    trainer.append_samples(x_train)
    est = trainer.train(max_num_epochs=3000)

    # Sample from the marginal pdf estimator
    samples = est.sample(torch.Size((num_test_samples,)))

    # Compute the C2ST score
    x_test = dist.sample(torch.Size((num_test_samples,)))
    if len(x_test.shape) == 1:
        x_test = x_test.unsqueeze(1)

    check_c2st(x_test, samples.cpu(), f'MarginalEstimator-{model}')


def test_marginal_trainer_exhausted_epoch_budget_returns_the_best_weights():
    """A run that hits `max_num_epochs` must warn and end on its best weights.

    The trainer has its own training loop, whose condition short-circuits like the
    shared one in `NeuralInference`: `_converged` never scores the final epoch.
    """
    torch.manual_seed(0)
    x = torch.randn(200, 2)

    trainer = MarginalTrainer(show_progress_bars=False)
    trainer.append_samples(x)
    with pytest.warns(UserWarning, match="max_num_epochs"):
        trainer.train(max_num_epochs=2, stop_after_epochs=50)

    assert trainer.epoch > 2, "the budget must run out for this to test anything"
    final = trainer._neural_net.state_dict()
    assert all(
        torch.equal(final[k], v) for k, v in trainer._best_model_state_dict.items()
    )
