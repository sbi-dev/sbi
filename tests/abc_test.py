import pytest
import torch
from torch import eye, ones, zeros

from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)

from sbi.inference.abc.mcabc import MCABC
from torch.distributions import MultivariateNormal
from pyro.distributions import Empirical
from tests.test_utils import check_c2st


@pytest.mark.parametrize("num_dim", (1, 2))
def test_mcabc_inference_on_linear_gaussian(num_dim):
    x_o = zeros((1, num_dim))

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((10000,))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = MCABC(simulator, prior, x_o, simulation_batch_size=10000)

    phat = infer(1000000, quantile=0.001)

    check_c2st(phat.sample((10000,)), target_samples, alg="MCABC")

