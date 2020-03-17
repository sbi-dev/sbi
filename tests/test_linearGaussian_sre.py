import pytest
import torch
from torch import distributions

import sbi.utils as utils
from sbi.inference.sre.sre import SRE
from sbi.simulators.linear_gaussian import (
    get_true_posterior_samples_linear_gaussian_mvn_prior,
    get_true_posterior_samples_linear_gaussian_uniform_prior,
    linear_gaussian,
)

# seed the simulations
torch.manual_seed(0)

# will be called by pytest. Then runs test_*(num_dim) for 1D and 3D
@pytest.mark.parametrize("num_dim", [1, 3])
def test_sre_on_linearGaussian_api(num_dim: int):
    """Test inference api of SRE with linear gaussian model. 

    Avoids intense computation for fast testing of API etc. 
    
    Keyword Arguments:
        num_dim {int} -- Parameter dimension of the gaussian model (default: {3})
    """
    # test api for inference on linear Gaussian model using SNL
    # avoids expensive computations for fast testing

    simulator = linear_gaussian
    prior = distributions.MultivariateNormal(
        loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
    )

    parameter_dim, observation_dim = num_dim, num_dim
    true_observation = torch.zeros(num_dim)

    # get classifier
    classifier = utils.classifier_nn("resnet", prior=prior, context=true_observation,)

    # create inference method
    inference_method = SRE(
        simulator=simulator,
        prior=prior,
        true_observation=true_observation,
        classifier=classifier,
        mcmc_method="slice-np",
    )

    # run inference
    posterior = inference_method(num_rounds=1, num_simulations_per_round=1000)

    # draw samples from posterior
    samples = posterior.sample(num_samples=100)

    log_probs = posterior.log_prob(samples)


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", [1, 3])
@pytest.mark.parametrize("prior_str", ("uniform", "gaussian"))
def test_sre_on_linearGaussian_based_on_mmd(num_dim: int, prior_str: str):
    """Test mmd accuracy of inference with SRE on linear gaussian model. 

    NOTE: The mmd threshold is calculated based on a number of test runs and taking the mean plus 2 stds. 
    
    Keyword Arguments:
        num_dim {int} -- Parameter dimension of the gaussian model (default: {3})
        prior_str {str} -- string for which prior to use: gaussian or uniform
    """

    true_observation = torch.zeros((1, num_dim))
    num_samples = 1000

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

    # get classifier
    classifier = utils.classifier_nn("resnet", prior=prior, context=true_observation,)

    # create inference method
    inference_method = SRE(
        simulator=linear_gaussian,
        prior=prior,
        true_observation=true_observation,
        classifier=classifier,
        mcmc_method="slice-np",
    )

    # run inference
    posterior = inference_method(num_rounds=1, num_simulations_per_round=1000)

    # draw samples from posterior
    samples = posterior.sample(num_samples=1000)

    # compute the mmd
    mmd = utils.unbiased_mmd_squared(target_samples, samples)

    # check if mmd is larger than expected
    max_mmd = 0.02

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."


@pytest.mark.parametrize(
    "mcmc_method, prior",
    (
        ("slice-np", "gaussian"),
        ("slice-np", "uniform"),
        # ("slice", "gaussian"),  # XXX this is broken-hard to debug error
        # ("slice", "uniform"),   # XXX might be slow? check pyro sampling
    ),
)
def test_sre_posterior_correction(mcmc_method, prior):
    """Test that leakage correction applied to sampling works, with both MCMC and rejection."""

    num_dim = 2
    simulator = linear_gaussian
    if prior == "gaussian":
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
        )
    else:
        prior = utils.BoxUniform(
            low=-1.0 * torch.ones(num_dim), high=torch.ones(num_dim)
        )

    true_observation = torch.zeros(num_dim)

    classifier = utils.classifier_nn("resnet", prior=prior, context=true_observation,)

    # create inference method
    inference_method = SRE(
        simulator=simulator,
        prior=prior,
        true_observation=true_observation,
        classifier=classifier,
        mcmc_method=mcmc_method,
    )

    # run inference
    posterior = inference_method(num_rounds=1, num_simulations_per_round=1000)

    # draw samples from posterior
    samples = posterior.sample(num_samples=1000)

    # no log prob for SRE yet - see issue #73
    # densities = posterior.log_prob(samples)
