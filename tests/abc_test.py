import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal, biject_to

from sbi.inference import MCABC, SMC
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils import BoxUniform
from tests.test_utils import check_c2st


@pytest.mark.parametrize("num_dim", (1, 2))
def test_mcabc_inference_on_linear_gaussian(
    num_dim,
    lra=False,
    sass=False,
    sass_expansion_degree=1,
    kde=False,
    kde_bandwidth="cv",
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

    inferer = MCABC(simulator, prior, simulation_batch_size=10000)

    phat = inferer(
        x_o,
        120000,
        quantile=0.01,
        lra=lra,
        sass=sass,
        sass_expansion_degree=sass_expansion_degree,
        sass_fraction=0.33,
        kde=kde,
        kde_kwargs=dict(bandwidth=kde_bandwidth) if kde else {},
        return_summary=False,
    )

    check_c2st(
        phat.sample((num_samples,)) if kde else phat,
        target_samples,
        alg=f"MCABC_lra{lra}_sass{sass}_kde{kde}_{kde_bandwidth}",
    )


@pytest.mark.slow
@pytest.mark.parametrize("lra", (True, False))
@pytest.mark.parametrize("sass_expansion_degree", (1, 2))
def test_mcabc_sass_lra(lra, sass_expansion_degree):

    test_mcabc_inference_on_linear_gaussian(
        num_dim=2, lra=lra, sass=True, sass_expansion_degree=sass_expansion_degree
    )


@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("prior_type", ("uniform", "gaussian"))
def test_smcabc_inference_on_linear_gaussian(
    num_dim,
    prior_type: str,
    lra=False,
    sass=False,
    sass_expansion_degree=1,
    kde=False,
    kde_bandwidth="cv",
    transform=False,
    num_simulations=20000,
):
    x_o = zeros((1, num_dim))
    num_samples = 1000
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    if prior_type == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    elif prior_type == "uniform":
        prior = BoxUniform(-ones(num_dim), ones(num_dim))
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o[0], likelihood_shift, likelihood_cov, prior, num_samples
        )
    else:
        raise ValueError("Wrong prior string.")

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SMC(simulator, prior, simulation_batch_size=10000, algorithm_variant="C")

    phat = infer(
        x_o,
        num_particles=1000,
        num_initial_pop=5000,
        epsilon_decay=0.5,
        num_simulations=num_simulations,
        distance_based_decay=True,
        return_summary=False,
        lra=lra,
        sass=sass,
        sass_fraction=0.5,
        sass_expansion_degree=sass_expansion_degree,
        kde=kde,
        kde_kwargs=dict(
            bandwidth=kde_bandwidth,
            transform=biject_to(prior.support) if transform else None,
        ),
    )

    check_c2st(
        phat.sample((num_samples,)) if kde else phat,
        target_samples,
        alg=f"SMCABC-{prior_type}prior-lra{lra}-sass{sass}-kde{kde}-{kde_bandwidth}",
    )

    if kde:
        samples = phat.sample((10,))
        phat.log_prob(samples)


@pytest.mark.slow
@pytest.mark.parametrize("lra", (True, False))
@pytest.mark.parametrize("sass_expansion_degree", (1, 2))
def test_smcabc_sass_lra(lra, sass_expansion_degree):

    test_smcabc_inference_on_linear_gaussian(
        num_dim=2,
        lra=lra,
        sass=True,
        sass_expansion_degree=sass_expansion_degree,
        prior_type="gaussian",
        num_simulations=20000,
    )


@pytest.mark.parametrize("kde_bandwidth", ("cv", "silvermann", "scott", 0.1))
def test_mcabc_kde(kde_bandwidth):
    test_mcabc_inference_on_linear_gaussian(
        num_dim=2, kde=True, kde_bandwidth=kde_bandwidth
    )


@pytest.mark.slow
@pytest.mark.parametrize("kde_bandwidth", ("cv",))
def test_smcabc_kde(kde_bandwidth):
    test_smcabc_inference_on_linear_gaussian(
        num_dim=2,
        lra=False,
        sass=False,
        prior_type="uniform",
        kde=True,
        kde_bandwidth=kde_bandwidth,
        transform=True,
    )
