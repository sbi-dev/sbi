# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import SNPE_A, SNPE_C, prepare_for_sbi, simulate_for_sbi
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian


@pytest.mark.parametrize("snpe_method", [SNPE_A, SNPE_C])
def test_log_prob_with_different_x(snpe_method: type):

    num_dim = 2

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator, prior = prepare_for_sbi(diagonal_linear_gaussian, prior)
    inference = snpe_method(prior)
    theta, x = simulate_for_sbi(simulator, prior, 1000)
    _ = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()

    _ = posterior.sample((10,), x=ones(1, num_dim))
    theta = posterior.sample((10,), ones(1, num_dim))
    posterior.log_prob(theta, x=ones(num_dim))
    posterior.log_prob(theta, x=ones(num_dim))
    posterior.log_prob(theta, x=ones(1, num_dim))
    posterior = posterior.set_default_x(ones(1, num_dim))
    posterior.log_prob(theta, x=None)
    posterior.sample((10,), x=None)

    # Both must fail due to batch size of x > 1.
    with pytest.raises(ValueError):
        posterior.log_prob(theta, x=ones(2, num_dim))
    with pytest.raises(ValueError):
        posterior.sample(2, x=ones(2, num_dim))
