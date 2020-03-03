import pytest
import torch
from torch import distributions

import sbi.simulators as simulators
import sbi.utils as utils
from sbi import inference
from sbi.inference.sre.sre import SRE

# use cpu by default
torch.set_default_tensor_type("torch.FloatTensor")

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

    simulator = simulators.linear_gaussian
    prior = distributions.MultivariateNormal(
        loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
    )

    parameter_dim, observation_dim = num_dim, num_dim
    true_observation = torch.zeros(num_dim)

    # get classifier
    classifier = utils.get_classifier(
        "resnet", parameter_dim=parameter_dim, observation_dim=observation_dim,
    )

    # create inference method
    inference_method = SRE(
        simulator=simulator,
        prior=prior,
        true_observation=true_observation,
        classifier=classifier,
        mcmc_method="slice-np",
    )

    # run inference
    inference_method.run_inference(num_rounds=1, num_simulations_per_round=1000)

    # draw samples from posterior
    samples = inference_method.sample_posterior(num_samples=100)


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", [1, 3])
def test_sre_on_linearGaussian_based_on_mmd(num_dim: int):
    """Test mmd accuracy of inference with SRE on linear gaussian model. 

    NOTE: The mmd threshold is calculated based on a number of test runs and taking the mean plus 2 stds. 
    
    Keyword Arguments:
        num_dim {int} -- Parameter dimension of the gaussian model (default: {3})
    """

    simulator = simulators.linear_gaussian
    prior = distributions.MultivariateNormal(
        loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
    )

    parameter_dim, observation_dim = num_dim, num_dim
    true_observation = torch.zeros(num_dim)

    # get classifier
    classifier = utils.get_classifier(
        "resnet", parameter_dim=parameter_dim, observation_dim=observation_dim,
    )

    # create inference method
    inference_method = SRE(
        simulator=simulator,
        prior=prior,
        true_observation=true_observation,
        classifier=classifier,
        mcmc_method="slice-np",
    )

    # run inference
    inference_method.run_inference(num_rounds=1, num_simulations_per_round=1000)

    # draw samples from posterior
    samples = inference_method.sample_posterior(num_samples=1000)

    # define target distribution (analytically tractable) and sample from it
    target_samples = simulator.get_ground_truth_posterior_samples(1000)

    # compute the mmd
    mmd = utils.unbiased_mmd_squared(target_samples, samples)

    # check if mmd is larger than expected
    max_mmd = 0.02

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."
