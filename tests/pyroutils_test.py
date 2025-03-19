# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pyro
import pytest
import torch

from sbi.inference import NLE, NPE
from sbi.utils.pyroutils import (
    ConditionedEstimatorDistribution,
    get_transforms,
)


def test_unbounded_transform():
    prior_dim = 10
    prior_params = {
        "low": -1.0 * torch.ones((prior_dim,)),
        "high": +1.0 * torch.ones((prior_dim,)),
    }
    prior_dist = pyro.distributions.Uniform(**prior_params).to_event(1)

    def prior(num_samples=1):
        return pyro.sample("theta", prior_dist.expand_by([num_samples]))

    transforms = get_transforms(prior)

    to_unbounded = transforms["theta"]
    to_bounded = transforms["theta"].inv

    assert to_unbounded(prior(1000)).max() > 1.0
    assert to_bounded(to_unbounded(prior(1000))).max() < 1.0


@pytest.mark.parametrize(
    "trainer_cls, distribution_cls",
    [
        (NLE, ConditionedEstimatorDistribution),
        (NPE, ConditionedEstimatorDistribution),
    ],
)
def test_estimator_distribution_basic_properties(
    trainer_cls,
    distribution_cls,
    num_simulations: int = 100,
    dim: int = 5,
):
    prior = torch.distributions.MultivariateNormal(
        loc=torch.zeros(dim), covariance_matrix=torch.diag(torch.ones(dim))
    )
    theta = prior.sample(torch.Size([num_simulations]))
    x = torch.distributions.Normal(theta, 1.0).sample()
    trainer = trainer_cls(prior=prior).append_simulations(theta=theta, x=x)
    density_estimator = trainer.train()

    # Test basic properties
    estimator_dist = distribution_cls(estimator=density_estimator, condition=theta)
    assert isinstance(estimator_dist, distribution_cls)
    assert estimator_dist.estimator == density_estimator
    assert estimator_dist.condition_shape == torch.Size([dim])
    assert estimator_dist.batch_shape == torch.Size([num_simulations])
    assert estimator_dist.event_shape == torch.Size([dim])
    assert estimator_dist.support == torch.distributions.constraints.real

    # Test sample method
    assert estimator_dist.sample().shape == torch.Size([num_simulations, dim])
    x_samples = estimator_dist.sample(torch.Size([3]))
    assert x_samples.shape == torch.Size([3, num_simulations, dim])

    # Test log_prob method
    assert estimator_dist.log_prob(x).shape == torch.Size([num_simulations])
    assert estimator_dist.log_prob(x_samples).shape == torch.Size([3, num_simulations])

    # Test expand method
    estimator_dist_expanded = estimator_dist.expand(torch.Size([2, num_simulations]))
    assert isinstance(estimator_dist_expanded, distribution_cls)
    assert estimator_dist_expanded.batch_shape == torch.Size([2, num_simulations])
    assert estimator_dist_expanded.event_shape == estimator_dist.event_shape
    assert estimator_dist_expanded.condition_shape == estimator_dist.condition_shape
    assert torch.equal(
        estimator_dist_expanded.condition,
        theta.expand(torch.Size([2, num_simulations, dim])),
    )
    assert estimator_dist_expanded.estimator == estimator_dist.estimator
