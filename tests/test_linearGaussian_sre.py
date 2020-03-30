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
    
    Args:
        num_dim: parameter dimension of the gaussian model (default: {3})
    """
    # test api for inference on linear Gaussian model using SNL
    # avoids expensive computations for fast testing

    simulator = linear_gaussian
    prior = distributions.MultivariateNormal(
        loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
    )

    parameter_dim, observation_dim = num_dim, num_dim
    # XXX this breaks the test! (and #76 doesn't seem to fix)
    # true_observation = torch.zeros(1, num_dim)
    true_observation = torch.zeros(num_dim)

    # get classifier
    classifier = utils.classifier_nn("resnet", prior=prior, context=true_observation,)

    # create inference method
    inference_method = SRE(
        simulator=simulator,
        prior=prior,
        true_observation=true_observation,
        classifier=classifier,
        simulation_batch_size=50,
        mcmc_method="slice-np",
    )

    # run inference
    posterior = inference_method(num_rounds=1, num_simulations_per_round=1000)

    # draw samples from posterior
    samples = posterior.sample(num_samples=10, num_chains=2)

    # XXX log_prob is not implemented yet for SRE
    # log_probs = posterior.log_prob(samples)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_dim, prior_str, classifier_loss",
    (
        (3, "gaussian", "sre"),
        (1, "gaussian", "sre"),
        (3, "uniform", "sre"),
        (3, "gaussian", "aalr"),
    ),
)
def test_sre_on_linearGaussian_based_on_mmd(
    num_dim: int, prior_str: str, classifier_loss: str
):
    """Test mmd accuracy of inference with SRE on linear gaussian model. 

    NOTE: The mmd threshold is calculated based on a number of test runs and taking the mean plus 2 stds. 
    
    Args:
        num_dim: parameter dimension of the gaussian model (default: {3})
        prior_str: string for which prior to use: gaussian or uniform
    """

    true_observation = torch.zeros(num_dim)
    num_samples = 300

    if prior_str == "gaussian":
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
        )
        target_samples = get_true_posterior_samples_linear_gaussian_mvn_prior(
            true_observation[None,], num_samples=num_samples
        )
    else:
        prior = utils.BoxUniform(-1.0 * torch.ones(num_dim), torch.ones(num_dim))
        target_samples = get_true_posterior_samples_linear_gaussian_uniform_prior(
            true_observation[None,], num_samples=num_samples, prior=prior
        )

    # get classifier
    classifier = utils.classifier_nn("resnet", prior=prior, context=true_observation,)

    num_atoms = 2 if classifier_loss == "aalr" else -1

    # create inference method
    inference_method = SRE(
        simulator=linear_gaussian,
        prior=prior,
        true_observation=true_observation,
        num_atoms=num_atoms,
        classifier=classifier,
        classifier_loss=classifier_loss,
        simulation_batch_size=50,
        mcmc_method="slice-np",
    )

    # run inference
    posterior = inference_method(num_rounds=1, num_simulations_per_round=1000)

    # draw samples from posterior
    samples = posterior.sample(num_samples=num_samples)

    # compute the mmd
    mmd = utils.unbiased_mmd_squared(target_samples, samples)

    # check if mmd is larger than expected
    max_mmd = 0.045

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."

    # Checks for log_prob()
    if prior_str == "gaussian" and classifier_loss == "aalr":
        # For the Gaussian prior, we compute the D-KL between ground truth and
        # posterior. We can do this only if the classifier_loss was as described in
        # Hermans et al. 2019 ('aalr') since Durkan et al. 2019 version only allows
        # evaluation up to a constant.
        utils.test_utils.dkl_gaussian_prior(posterior, true_observation, num_dim)
    if prior_str == "uniform":
        # test whether likelihood outside prior support is zero. Prior bounds are
        # [-1, 1] in each dimension, so tensor of 2s will be outside of bounds.
        sample_outside_support = 2 * torch.ones(num_dim)
        posterior_prob = torch.exp(posterior.log_prob(sample_outside_support))
        assert (
            posterior_prob == 0.0
        ), "The posterior probability outside of the prior support is not zero"


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
def test_sre_posterior_correction(mcmc_method, prior_str):
    """Test that leakage correction applied to sampling works, with both MCMC and rejection.

    Args:
        mcmc_method: which mcmc method to use for sampling
        prior_str: use gaussian or uniform prior
    """

    num_dim = 2
    simulator = linear_gaussian
    if prior_str == "gaussian":
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
        simulation_batch_size=50,
        mcmc_method=mcmc_method,
    )

    # run inference
    posterior = inference_method(num_rounds=1, num_simulations_per_round=1000)

    # draw samples from posterior
    samples = posterior.sample(num_samples=50)

    # no log prob for SRE yet - see issue #73
    # densities = posterior.log_prob(samples)
