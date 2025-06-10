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
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
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
