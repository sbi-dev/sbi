import pytest
import torch
from torch import distributions, zeros, ones, eye

import sbi.utils as utils
from sbi.inference.snl.snl import SNL
from sbi.simulators.linear_gaussian import (
    get_true_posterior_samples_linear_gaussian_mvn_prior,
    get_true_posterior_samples_linear_gaussian_uniform_prior,
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

    prior = distributions.MultivariateNormal(
        loc=zeros(num_dim), covariance_matrix=eye(num_dim)
    )

    infer = SNL(
        simulator=linear_gaussian,
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

    if prior_str == "gaussian":
        prior = distributions.MultivariateNormal(
            loc=zeros(num_dim), covariance_matrix=eye(num_dim)
        )
        target_samples = get_true_posterior_samples_linear_gaussian_mvn_prior(
            x_o, num_samples=num_samples
        )
    else:
        prior = utils.BoxUniform(-1.0 * ones(num_dim), ones(num_dim))
        target_samples = get_true_posterior_samples_linear_gaussian_uniform_prior(
            x_o, num_samples=num_samples, prior=prior
        )

    infer = SNL(
        simulator=linear_gaussian,
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

    prior = distributions.MultivariateNormal(
        loc=zeros(num_dim), covariance_matrix=eye(num_dim)
    )
    target_samples = get_true_posterior_samples_linear_gaussian_mvn_prior(
        x_o, num_samples=num_samples
    )

    infer = SNL(
        simulator=linear_gaussian,
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
        prior = distributions.MultivariateNormal(
            loc=zeros(num_dim), covariance_matrix=eye(num_dim)
        )
    else:
        prior = utils.BoxUniform(-1.0 * ones(num_dim), ones(num_dim))

    infer = SNL(
        simulator=linear_gaussian,
        prior=prior,
        x_o=x_o,
        density_estimator=None,  # Use default MAF.
        simulation_batch_size=50,
        mcmc_method="slice_np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    posterior.sample(num_samples=num_samples)
