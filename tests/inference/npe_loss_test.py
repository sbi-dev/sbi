# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch.distributions import MultivariateNormal
from typing import Any, Optional

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.trainers.npe.npe_loss import (
    AtomicLoss,
    ImportanceWeightedLoss,
    NonAtomicGaussianLoss,
)
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.estimators.mixture_density_estimator import MixtureDensityEstimator
from sbi.neural_nets.estimators.mog import MoG
from sbi.utils.torchutils import BoxUniform


class DummyEstimator(ConditionalDensityEstimator):
    def __init__(self, is_mixture=False):
        super().__init__(net=torch.nn.Identity(), input_shape=torch.Size((2,)), condition_shape=torch.Size((2,)))
        self.is_mixture = is_mixture
        self.condition_shape = torch.Size((2,))
        self.input_shape = torch.Size((2,))

    def log_prob(self, theta: torch.Tensor, condition: torch.Tensor, **kwargs):
        # Dummy implementation returning zeros matching batch shape
        batch_size = max(theta.shape[0], condition.shape[0])
        return torch.zeros(batch_size, device=theta.device)

    def sample(self, sample_shape: torch.Size, condition: torch.Tensor, **kwargs):
        return torch.zeros(sample_shape + self.input_shape, device=condition.device)

    def loss(self, theta, condition, **kwargs):
        return torch.zeros(theta.shape[0])

    def get_uncorrected_mog(self, condition: torch.Tensor) -> MoG:
        assert self.is_mixture
        # Returns simple 1-component MoG
        batch_size = condition.shape[0]
        dim = self.input_shape[0]
        logits = torch.zeros(batch_size, 1)
        means = torch.zeros(batch_size, 1, dim)
        precisions = torch.eye(dim).view(1, 1, dim, dim).expand(batch_size, 1, dim, dim)
        precf = torch.eye(dim).view(1, 1, dim, dim).expand(batch_size, 1, dim, dim)
        return MoG(logits, means, precisions, prec_factors=precf)

    @property
    def has_input_transform(self):
        return False

# Proxy dummy specifically simulating MixtureDensityEstimator to pass isinstance
class DummyMixtureEstimator(DummyEstimator, MixtureDensityEstimator): # type: ignore
    def __init__(self):
        DummyEstimator.__init__(self, is_mixture=True)


class DummyPosterior(DirectPosterior):
    def __init__(self, estimator, prior):
        # We bypass DirectPosterior's strict checks for testing MoG extraction
        self.posterior_estimator = estimator
        self.prior = prior
        self.default_x = torch.zeros(1, 2)


@pytest.fixture
def theta():
    return torch.randn(5, 2)


@pytest.fixture
def x():
    return torch.randn(5, 2)


@pytest.fixture
def masks():
    return torch.ones(5)


@pytest.fixture
def prior():
    return MultivariateNormal(torch.zeros(2), torch.eye(2))


def test_atomic_loss_initialization_and_call(theta, x, masks, prior):
    neural_net = DummyEstimator()
    strategy = AtomicLoss(neural_net=neural_net, prior=prior, num_atoms=2)
    
    assert strategy.uses_only_latest_round is False
    
    loss = strategy(theta, x, masks, proposal=None)
    assert loss.shape == (5,)


def test_non_atomic_loss_initialization_and_call(theta, x, masks, prior):
    neural_net = DummyMixtureEstimator()
    proposal = DummyPosterior(neural_net, prior)
    
    strategy = NonAtomicGaussianLoss(
        neural_net=neural_net,
        maybe_z_scored_prior=prior,
    )
    
    assert strategy.uses_only_latest_round is True
    
    loss = strategy(theta, x, masks, proposal=proposal)
    assert loss.shape == (5,)


def test_importance_weighted_loss_initialization_and_call(theta, x, masks, prior):
    neural_net = DummyEstimator()
    proposal = DummyPosterior(neural_net, prior)
    
    theta_roundwise = [torch.randn(10, 2), torch.randn(10, 2)]
    proposal_roundwise = [prior, proposal]
    
    strategy = ImportanceWeightedLoss(
        neural_net=neural_net,
        prior=prior,
        round_idx=1,
        theta_roundwise=theta_roundwise,
        proposal_roundwise=proposal_roundwise,
    )
    
    assert strategy.uses_only_latest_round is False
    
    loss = strategy(theta, x, masks, proposal=proposal)
    assert loss.shape == (5,)
