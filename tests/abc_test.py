# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
from torch import eye, norm, ones, zeros
from torch.distributions import MultivariateNormal, biject_to

from sbi.inference import MCABC, SMC
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.torchutils import BoxUniform
from tests.test_utils import check_c2st


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("distance", ("l2", lambda x, xo: norm((x - xo), dim=-1)))
def test_mcabc_inference_on_linear_gaussian(
    num_dim,
    distance,
    lra=False,
    sass=False,
    sass_expansion_degree=1,
    kde=False,
    kde_bandwidth="cv",
    num_iid_samples=1,
    distance_kwargs=None,
):
    x_o = zeros((num_iid_samples, num_dim))
    num_samples = 1000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inferer = MCABC(
        simulator,
        prior,
        simulation_batch_size=10000,
        distance=distance,
        distance_kwargs=distance_kwargs,
    )

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
        num_iid_samples=num_iid_samples,
    )

    check_c2st(
        phat.sample((num_samples,)) if kde else phat,
        target_samples,
        alg=f"MCABC_lra{lra}_sass{sass}_kde{kde}_{kde_bandwidth}",
    )


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("prior_type", ("uniform", "gaussian"))
def test_smcabc_inference_on_linear_gaussian(
    num_dim,
    prior_type: str,
    distance="l2",
    lra=False,
    sass=False,
    sass_expansion_degree=1,
    kde=False,
    kde_bandwidth="cv",
    transform=False,
    num_simulations=20000,
    num_iid_samples=1,
    distance_kwargs=None,
):
    x_o = zeros((num_iid_samples, num_dim))
    num_samples = 1000
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    if prior_type == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    elif prior_type == "uniform":
        prior = BoxUniform(-ones(num_dim), ones(num_dim))
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o, likelihood_shift, likelihood_cov, prior, num_samples
        )
    else:
        raise ValueError("Wrong prior string.")

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SMC(
        simulator,
        prior,
        distance=distance,
        simulation_batch_size=10000,
        algorithm_variant="C",
        distance_kwargs=distance_kwargs,
    )

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
        num_iid_samples=num_iid_samples,
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
def test_mcabc_sass_lra(lra, sass_expansion_degree):
    test_mcabc_inference_on_linear_gaussian(
        num_dim=2,
        lra=lra,
        sass=True,
        sass_expansion_degree=sass_expansion_degree,
        distance="l2",
    )


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


@pytest.mark.slow
@pytest.mark.parametrize("kde_bandwidth", ("cv", "silvermann", "scott", 0.1))
def test_mcabc_kde(kde_bandwidth):
    test_mcabc_inference_on_linear_gaussian(
        num_dim=2, kde=True, kde_bandwidth=kde_bandwidth, distance="l2"
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "distance, num_iid_samples, distance_kwargs",
    (
        ["l2", 1, None],
        ["mmd", 10, {"scale": 1.5}],
    ),
)
def test_mc_abc_iid_inference(distance, num_iid_samples, distance_kwargs):
    test_mcabc_inference_on_linear_gaussian(
        num_dim=2,
        distance=distance,
        num_iid_samples=num_iid_samples,
        distance_kwargs=distance_kwargs,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "distance, num_iid_samples, distance_kwargs, distance_batch_size",
    (
        ["l2", 1, None, -1],
        ["mmd", 20, {"scale": 1.0}, -1],
        ["wasserstein", 10, {"epsilon": 1.0, "tol": 1e-6, "max_iter": 1000}, 1000],
    ),
)
def test_smcabc_iid_inference(
    distance, num_iid_samples, distance_kwargs, distance_batch_size
):
    test_smcabc_inference_on_linear_gaussian(
        num_dim=2,
        prior_type="gaussian",
        distance=distance,
        num_iid_samples=num_iid_samples,
        num_simulations=20000,
        distance_kwargs=distance_kwargs,
    )
