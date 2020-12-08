import pytest
from torch import eye, ones, zeros

from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)

from sbi.inference import ABC, SMC
from torch.distributions import MultivariateNormal
from tests.test_utils import check_c2st


@pytest.mark.parametrize("num_dim", (1, 2))
def test_mcabc_inference_on_linear_gaussian(
    num_dim, lra=False, sass=False, sass_expansion_degree=1,
):
    x_o = zeros((1, num_dim))
    num_samples = 1000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = ABC(simulator, prior, simulation_batch_size=10000)

    phat = infer(
        x_o,
        100000,
        quantile=0.01,
        lra=lra,
        sass=sass,
        sass_expansion_degree=sass_expansion_degree,
        sass_fraction=0.33,
    )

    check_c2st(phat.sample((num_samples,)), target_samples, alg="MCABC")


@pytest.mark.slow
@pytest.mark.parametrize("lra", (True, False))
@pytest.mark.parametrize("sass_expansion_degree", (1, 2))
def test_mcabc_sass_lra(lra, sass_expansion_degree, set_seed):

    test_mcabc_inference_on_linear_gaussian(
        num_dim=2, lra=lra, sass=True, sass_expansion_degree=sass_expansion_degree,
    )


@pytest.mark.parametrize("num_dim", (1, 2))
def test_smcabc_inference_on_linear_gaussian(
    num_dim, lra=False, sass=False, sass_expansion_degree=1
):
    x_o = zeros((1, num_dim))
    num_samples = 1000
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SMC(simulator, prior, simulation_batch_size=10000, algorithm_variant="C")

    phat = infer(
        x_o,
        num_particles=1000,
        num_initial_pop=5000,
        epsilon_decay=0.5,
        num_simulations=100000,
        distance_based_decay=True,
        lra=lra,
        sass=sass,
        sass_fraction=0.5,
        sass_expansion_degree=sass_expansion_degree,
    )

    check_c2st(phat.sample((num_samples,)), target_samples, alg="SMCABC")


@pytest.mark.slow
@pytest.mark.parametrize("lra", (True, False))
@pytest.mark.parametrize("sass_expansion_degree", (1, 2))
def test_smcabc_sass_lra(lra, sass_expansion_degree, set_seed):

    test_smcabc_inference_on_linear_gaussian(
        num_dim=2, lra=lra, sass=True, sass_expansion_degree=sass_expansion_degree
    )
