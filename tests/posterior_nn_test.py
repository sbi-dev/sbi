import pytest
from torch import zeros, ones, eye
from torch.distributions import MultivariateNormal

from sbi.inference.snpe.snpe_c import SnpeC
from sbi.simulators.linear_gaussian import linear_gaussian


def test_log_prob_with_different_x():

    num_dim = 2

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    posterior = SnpeC(simulator=linear_gaussian, prior=prior, x_o=ones(num_dim))(
        num_rounds=1, num_simulations_per_round=1000
    )

    _ = posterior.sample(10, x=None)
    theta = posterior.sample(10, ones(num_dim))
    posterior.log_prob(theta, x=None)
    posterior.log_prob(theta, x=ones(num_dim))
    posterior.log_prob(theta, x=ones(1, num_dim))

    # Both must fail due to batch size of x > 1.
    with pytest.raises(ValueError):
        posterior.log_prob(theta, x=ones(2, num_dim))
    with pytest.raises(ValueError):
        posterior.sample(2, x=ones(2, num_dim))
