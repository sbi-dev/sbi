# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
from torch import Tensor, eye, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import NRE
from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets.net_builders import build_linear_classifier
from sbi.neural_nets.ratio_estimators import RatioEstimator
from sbi.utils.torchutils import BoxUniform


class EmbeddingNet(torch.nn.Module):
    def __init__(self, shape: torch.Size) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x):
        for _ in range(len(self.shape) - 1):
            x = torch.sum(x, dim=-1)
        return x


def get_embedding_net(shape: torch.Size) -> torch.nn.Module:
    if len(shape) == 3:
        return CNNEmbedding(shape[:2], shape[-1], kernel_size=2)
    else:
        return EmbeddingNet(shape)


@pytest.mark.parametrize("ratio_estimator", (RatioEstimator,))
@pytest.mark.parametrize(
    "theta_shape", ((1,), (2,), (1, 1), (2, 2), (28, 28, 1), (28, 28, 3), (28, 36, 3))
)
@pytest.mark.parametrize(
    "x_shape", ((1,), (2,), (1, 1), (2, 2), (28, 28, 1), (28, 28, 3), (28, 36, 3))
)
def test_api_ratio_estimator(ratio_estimator, theta_shape, x_shape):
    r"""Checks whether we can evaluate ratio estimators correctly.

    Args:
        ratio_estimator: RatioEstimator subclass.
        input_dim: Dimensionality of the input.
    """

    nsamples = 10

    theta_mvn = MultivariateNormal(
        loc=zeros(*theta_shape), covariance_matrix=eye(theta_shape[-1])
    )
    batch_theta = theta_mvn.sample(torch.Size((nsamples,)))
    x_mvn = MultivariateNormal(loc=zeros(*x_shape), covariance_matrix=eye(x_shape[-1]))
    batch_x = x_mvn.sample(torch.Size((nsamples,)))

    if ratio_estimator == RatioEstimator:
        estimator = build_linear_classifier(
            batch_x=batch_theta,
            batch_y=batch_x,
            embedding_net_x=get_embedding_net(theta_shape),
            embedding_net_y=get_embedding_net(x_shape),
        )
    else:
        raise NotImplementedError()

    # forward computes the unnormalized_log_ratio
    # calling all other methods in the process
    unnormalized_log_ratio = estimator(batch_theta, batch_x)
    assert unnormalized_log_ratio.shape == (
        nsamples,
    ), f"""unnormalized_log_ratio shape is not correct. It is of shape
    {unnormalized_log_ratio.shape}, but should be {(nsamples,)}"""


def test_ratio_estimator_inference_with_correct_classifier():
    """Test NRE inference with a correct classifier."""

    def build_classifier(theta, x):
        net = torch.nn.Linear(theta.shape[1] + x.shape[1], 1)
        return RatioEstimator(net=net, theta_shape=theta[0].shape, x_shape=x[0].shape)

    def simulator(theta):
        return 1.0 + theta + torch.randn(theta.shape, device=theta.device) * 0.1

    num_dim = 3
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

    inference = NRE(classifier=build_classifier)
    theta = prior.sample((300,))
    x = simulator(theta)

    inference.append_simulations(theta, x).train(max_num_epochs=1)


# Incorrect ratio estimator classifier builders
def build_classifier_missing_args():  # Missing required parameters
    pass


def build_classifier_missing_return(theta: Tensor, x: Tensor):
    # Missing return of RatioEstimator
    pass


@pytest.mark.parametrize(
    "classifier",
    [
        build_classifier_missing_args,
        build_classifier_missing_return,
    ],
)
def test_ratio_estimator_inference_with_incorrect_classifier(classifier):
    """Test NRE inference raises an error with incorrect classifiers."""

    def simulator(theta):
        return 1.0 + theta + torch.randn(theta.shape, device=theta.device) * 0.1

    num_dim = 3
    prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
    theta = prior.sample((300,))
    x = simulator(theta)

    inference = NRE(classifier=classifier)
    inference.append_simulations(theta, x)

    with pytest.raises((AttributeError, TypeError)):
        inference.train(max_num_epochs=1)
