# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.neural_nets.critic import build_linear_critic
from sbi.neural_nets.ratio_estimators import TensorRatioEstimator


class EmbeddingNet(torch.nn.Module):
    def __init__(self, shape: torch.Size) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x):
        for _ in range(len(self.shape) - 1):
            x = torch.sum(x, dim=-1)
        return x


@pytest.mark.parametrize("ratio_estimator", (TensorRatioEstimator,))
@pytest.mark.parametrize(
    "theta_shape", ((1,), (2,), (1, 1), (2, 2), (1, 1, 1), (2, 2, 2))
)
@pytest.mark.parametrize("x_shape", ((1,), (2,), (1, 1), (2, 2), (1, 1, 1), (2, 2, 2)))
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
    batch_theta = theta_mvn.sample((nsamples,))
    x_mvn = MultivariateNormal(loc=zeros(*x_shape), covariance_matrix=eye(x_shape[-1]))
    batch_x = x_mvn.sample((nsamples,))

    if ratio_estimator == TensorRatioEstimator:
        estimator = build_linear_critic(
            batch_x=batch_theta,
            batch_y=batch_x,
            embedding_net_x=EmbeddingNet(theta_shape),
            embedding_net_y=EmbeddingNet(x_shape),
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
