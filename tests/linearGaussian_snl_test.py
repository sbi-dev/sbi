import pytest
import torch
from torch import zeros, ones, eye
from torch.distributions import MultivariateNormal

import sbi.utils as utils
from sbi.inference.snl.snl import SNL
from sbi.simulators.linear_gaussian import (
    true_posterior_linear_gaussian_mvn_prior,
    samples_true_posterior_linear_gaussian_uniform_prior,
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    standard_linear_gaussian,
    linear_gaussian_different_dims,
    linear_gaussian,
)
from tests.test_utils import check_c2st, get_prob_outside_uniform_prior

# use cpu by default
torch.set_default_tensor_type("torch.FloatTensor")

# Seeding:
# Some tests in this module have "set_seed" as an argument. This argument points to
# tests/conftest.py to seed the test with the seed set in conftext.py.


@pytest.mark.parametrize("num_dim", (1, 3))
def test_snl_on_linearGaussian_api(num_dim: int):
    """Test API for inference on linear Gaussian model using SNL.

    Avoids expensive computations by training on few simulations and generating few
    posterior samples.

    Args:
        num_dim: parameter dimension of the gaussian model
    """
    num_samples = 10

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    infer = SNL(
        simulator=standard_linear_gaussian,
        prior=prior,
        x_o=zeros(num_dim),
        density_estimator=None,  # Use default MAF.
        simulation_batch_size=50,
        mcmc_method="slice_np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    posterior.sample(num_samples=num_samples, num_chains=1)


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("prior_str", ("uniform", "gaussian"))
def test_snl_on_linearGaussian_based_on_c2st(num_dim: int, prior_str: str, set_seed):
    """Test SNL on linear Gaussian, comparing to ground truth posterior via MMD.

    NOTE: The MMD threshold is calculated based on a number of test runs and taking the
    mean plus 2 stds.

    Args:
        num_dim: parameter dimension of the gaussian model
        prior_str: one of "gaussian" or "uniform"
        set_seed: fixture for manual seeding
    """

    x_o = zeros((1, num_dim))
    num_samples = 300

    likelihood_shift = -1.0 * ones(
        num_dim
    )  # likelihood_mean will be likelihood_shift+theta
    likelihood_cov = 0.3 * eye(num_dim)

    if prior_str == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    else:
        prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o, likelihood_shift, likelihood_cov, prior=prior, num_samples=num_samples
        )

    simulator = lambda theta: linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SNL(
        simulator=simulator,
        prior=prior,
        x_o=x_o,
        density_estimator=None,  # Use default MAF.
        mcmc_method="slice_np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    samples = posterior.sample(num_samples=num_samples)

    # Check performance based on c2st accuracy.
    check_c2st(samples, target_samples, alg=f"snl-{prior_str}-prior")

    # TODO: we do not have a test for SNL log_prob(). This is because the output
    # TODO: density is not normalized, so KLd does not make sense.
    if prior_str == "uniform":
        # Check whether the returned probability outside of the support is zero.
        posterior_prob = get_prob_outside_uniform_prior(posterior, num_dim)
        assert (
            posterior_prob == 0.0
        ), "The posterior probability outside of the prior support is not zero"


def test_snl_on_linearGaussian_different_dims_based_on_c2st(set_seed):
    """Test whether SNL infers well a simple example with available round truth.

    This example has different number of parameters theta than number of x.

    Args:
        set_seed: fixture for manual seeding
    """

    theta_dim = 3
    x_dim = 2
    discard_dims = theta_dim - x_dim

    x_o = ones(1, x_dim)
    num_samples = 500

    likelihood_shift = -1.0 * ones(
        x_dim
    )  # likelihood_mean will be likelihood_shift+theta
    likelihood_cov = 0.3 * eye(x_dim)

    prior_mean = zeros(theta_dim)
    prior_cov = eye(theta_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    target_samples = samples_true_posterior_linear_gaussian_mvn_prior_different_dims(
        x_o[0],
        likelihood_shift,
        likelihood_cov,
        prior_mean,
        prior_cov,
        num_discarded_dims=discard_dims,
        num_samples=num_samples,
    )

    simulator = lambda theta: linear_gaussian_different_dims(
        theta, likelihood_shift, likelihood_cov, num_discarded_dims=discard_dims
    )

    infer = SNL(
        simulator=simulator,
        prior=prior,
        x_o=x_o,
        density_estimator=None,  # Use default MAF.
        simulation_batch_size=50,
        mcmc_method="slice_np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=3000)  # type: ignore
    samples = posterior.sample(num_samples, thin=3)

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="snl")


@pytest.mark.slow
def test_multi_round_snl_on_linearGaussian_based_on_c2st(set_seed):
    """Test SNL on linear Gaussian, comparing to ground truth posterior via MMD.

    NOTE: The MMD threshold is calculated based on a number of test runs and taking the
    mean plus 2 stds.

    Args:
        set_seed: fixture for manual seeding
    """

    num_dim = 2
    x_o = zeros((1, num_dim))
    num_samples = 300

    likelihood_shift = -1.0 * ones(
        num_dim
    )  # likelihood_mean will be likelihood_shift+theta
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    simulator = lambda theta: linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SNL(
        simulator=simulator,
        prior=prior,
        x_o=x_o,
        density_estimator=None,  # Use default MAF.
        simulation_batch_size=50,
        mcmc_method="slice",
    )

    posterior = infer(num_rounds=2, num_simulations_per_round=1000)

    samples = posterior.sample(num_samples=num_samples)

    # Check performance based on c2st accuracy.
    check_c2st(samples, target_samples, alg="multi-round-snl")


@pytest.mark.slow
@pytest.mark.parametrize(
    "mcmc_method, prior_str",
    (
        ("slice_np", "gaussian"),
        ("slice_np", "uniform"),
        ("slice", "gaussian"),
        ("slice", "uniform"),
    ),
)
def test_snl_posterior_correction(mcmc_method: str, prior_str: str, set_seed):
    """Runs SNL on linear Gaussian and tests sampling from posterior via mcmc.

    Args:
        mcmc_method: which mcmc method to use for sampling
        prior_str: use gaussian or uniform prior
        set_seed: fixture for manual seeding
    """

    num_dim = 2
    num_samples = 30
    x_o = zeros((1, num_dim))

    if prior_str == "gaussian":
        prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    else:
        prior = utils.BoxUniform(-1.0 * ones(num_dim), ones(num_dim))

    infer = SNL(
        simulator=standard_linear_gaussian,
        prior=prior,
        x_o=x_o,
        density_estimator=None,  # Use default MAF.
        simulation_batch_size=50,
        mcmc_method="slice_np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    posterior.sample(num_samples=num_samples)
