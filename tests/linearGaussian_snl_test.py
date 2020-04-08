import pytest
import torch
from torch import distributions

from sbi.inference.snl.snl import SNL
from sbi.simulators.linear_gaussian import (
    get_true_posterior_samples_linear_gaussian_mvn_prior,
    get_true_posterior_samples_linear_gaussian_uniform_prior,
    linear_gaussian,
)
import sbi.utils as utils

torch.manual_seed(0)


@pytest.mark.parametrize("num_dim", (1, 3))
def test_snl_on_linearGaussian_api(num_dim: int):
    """Test API for inference on linear Gaussian model using SNL.
    
    Avoids expensive computations by training on few simulations and generating few posterior samples.

    Args:
        num_dim: parameter dimension of the gaussian model
    """
    num_samples = 10

    prior = distributions.MultivariateNormal(
        loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
    )

    true_observation = torch.zeros(num_dim)

    neural_likelihood = utils.likelihood_nn(
        model="maf", prior=prior, context=true_observation,
    )

    infer = SNL(
        simulator=linear_gaussian,
        prior=prior,
        true_observation=true_observation,
        density_estimator=neural_likelihood,
        simulation_batch_size=50,
        mcmc_method="slice-np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    posterior.sample(num_samples=num_samples, num_chains=1)


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 3))
@pytest.mark.parametrize("prior_str", ("uniform", "gaussian"))
def test_snl_on_linearGaussian_based_on_mmd(num_dim: int, prior_str: str):
    """Test SNL on linear Gaussian, comparing to ground truth posterior via MMD.

    NOTE: The MMD threshold is calculated based on a number of test runs and taking the mean plus 2 stds. 
    
    Args:
        num_dim: parameter dimension of the gaussian model
        prior_str: one of "gaussian" or "uniform"
    """

    true_observation = torch.zeros((1, num_dim))
    num_samples = 200

    if prior_str == "gaussian":
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
        )
        target_samples = get_true_posterior_samples_linear_gaussian_mvn_prior(
            true_observation, num_samples=num_samples
        )
    else:
        prior = utils.BoxUniform(-1.0 * torch.ones(num_dim), torch.ones(num_dim))
        target_samples = get_true_posterior_samples_linear_gaussian_uniform_prior(
            true_observation, num_samples=num_samples, prior=prior
        )

    neural_likelihood = utils.likelihood_nn(
        model="maf", prior=prior, context=true_observation,
    )

    infer = SNL(
        simulator=linear_gaussian,
        prior=prior,
        true_observation=true_observation,
        density_estimator=neural_likelihood,
        mcmc_method="slice-np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    samples = posterior.sample(num_samples=num_samples)

    # compute the mmd
    mmd = utils.unbiased_mmd_squared(target_samples, samples)

    # check if mmd is larger than expected
    # NOTE: the mmd is calculated based on a number of test runs
    max_mmd = 0.02

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."


@pytest.mark.slow
def test_multi_round_snl_on_linearGaussian_based_on_mmd():
    """Test SNL on linear Gaussian, comparing to ground truth posterior via MMD.

    NOTE: The MMD threshold is calculated based on a number of test runs and taking the mean plus 2 stds. 

    """

    num_dim = 3
    true_observation = torch.zeros((1, num_dim))
    num_samples = 200

    prior = distributions.MultivariateNormal(
        loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
    )
    target_samples = get_true_posterior_samples_linear_gaussian_mvn_prior(
        true_observation, num_samples=num_samples
    )

    neural_likelihood = utils.likelihood_nn(
        model="maf", prior=prior, context=true_observation,
    )

    infer = SNL(
        simulator=linear_gaussian,
        prior=prior,
        true_observation=true_observation,
        density_estimator=neural_likelihood,
        simulation_batch_size=50,
        mcmc_method="slice",
    )

    posterior = infer(num_rounds=2, num_simulations_per_round=1000)

    samples = posterior.sample(num_samples=500)

    mmd = utils.unbiased_mmd_squared(target_samples, samples)

    # check if mmd is larger than expected
    # NOTE: the mmd is calculated based on a number of test runs
    max_mmd = 0.02

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."


@pytest.mark.slow
@pytest.mark.parametrize(
    "mcmc_method, prior_str",
    (
        ("slice-np", "gaussian"),
        ("slice-np", "uniform"),
        ("slice", "gaussian"),
        ("slice", "uniform"),
    ),
)
def test_snl_posterior_correction(mcmc_method: str, prior_str: str):
    """Test SNL on linear Gaussian, comparing to ground truth posterior via MMD.

    NOTE: The mmd threshold is calculated based on a number of test runs and taking the mean plus 2 stds.

    Args:
        mcmc_method: which mcmc method to use for sampling
        prior_str: use gaussian or uniform prior
    """

    num_dim = 2
    num_samples = 30
    true_observation = torch.zeros((1, num_dim))

    if prior_str == "gaussian":
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
        )
    else:
        prior = utils.BoxUniform(-1.0 * torch.ones(num_dim), torch.ones(num_dim))

    neural_likelihood = utils.likelihood_nn(
        model="maf", prior=prior, context=true_observation,
    )

    infer = SNL(
        simulator=linear_gaussian,
        prior=prior,
        true_observation=true_observation,
        density_estimator=neural_likelihood,
        simulation_batch_size=50,
        mcmc_method="slice-np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    posterior.sample(num_samples=num_samples)
