import pytest
import torch
from torch import distributions

import sbi.utils as utils
from sbi.inference.snpe.snpe_b import SnpeB
from sbi.inference.snpe.snpe_c import SnpeC
from sbi.simulators.linear_gaussian import (
    get_true_posterior_samples_linear_gaussian_mvn_prior,
    get_true_posterior_samples_linear_gaussian_uniform_prior,
    linear_gaussian,
)

torch.manual_seed(0)
import numpy as np


def load_data():
    # load the training data
    datafile = "/Users/deismic/Documents/Studium_TUM/MasterThesis/prinzetal/results/energy/classifier/prior_simulated/samples_array_subset1.npz"
    data = np.load(datafile)

    params = data["params"]
    stats = data["stats"]

    params = torch.as_tensor(params, dtype=torch.float32)
    stats = torch.as_tensor(stats, dtype=torch.float32)

    # define observation
    x_o = np.load(
        "/Users/deismic/Documents/Studium_TUM/MasterThesis/prinzetal/thesis_results/experimental/summstats/845_082/190807_summstats_prep845_082_0044.npz"
    )["summ_stats"]
    x_o = torch.as_tensor(x_o, dtype=torch.float32).unsqueeze(0)

    x_o = (x_o - torch.mean(stats, dim=0)) / torch.std(stats, dim=0)
    params = (params - torch.mean(params, dim=0)) / torch.std(params, dim=0)
    stats = (stats - torch.mean(stats, dim=0)) / torch.std(stats, dim=0)

    prior = utils.torchutils.BoxUniform(
        low=-1.73 * torch.ones(31), high=1.73 * torch.ones(31)
    )

    num_params = 3
    num_stats = 3
    params = params[:, :num_params]
    stats = stats[:, :num_stats]
    prior = utils.torchutils.BoxUniform(
        low=-1.73 * torch.ones(num_params), high=1.73 * torch.ones(num_params)
    )
    x_o = x_o[:, :num_stats]

    return params, stats, x_o, prior


# running all combinations is excessive. The standard test is (3, "gaussian", "snpe_c"),
# and we then vary only one parameter at a time to test single-d, uniform, and snpe-b
@pytest.mark.parametrize(
    "num_dim, prior_str, algorithm_str, simulation_batch_size",
    (
        (3, "gaussian", "snpe_c", 10),
        (3, "uniform", "snpe_c", 10),
        (1, "gaussian", "snpe_c", 10),
        (3, "gaussian", "snpe_b", 10),
        (3, "gaussian", "snpe_c", 1),
    ),
)
def test_apt_on_linearGaussian_based_on_mmd(
    num_dim: int, prior_str: str, algorithm_str: str, simulation_batch_size: int
):
    """Test whether APT infers well a simple example where ground truth is available."""

    x_o = torch.zeros(num_dim)
    num_samples = 100

    prior = utils.BoxUniform(-1.73 * torch.ones(num_dim), -1.73 * torch.ones(num_dim))

    neural_net = utils.posterior_nn(model="nsf", prior=prior, context=x_o)

    dataset_params = prior.sample(torch.tensor([1000]))
    dataset_stats = linear_gaussian(dataset_params)
    print("dataset_params", dataset_params)
    print("dataset_stats", dataset_stats)
    print("x_o", x_o)

    dataset_params, dataset_stats, x_o, prior = load_data()

    print("dataset_params", dataset_params)
    print("dataset_stats", dataset_stats)
    print("x_o", x_o)

    def dummy_simulate(parameter_vals):
        return 3 * torch.ones((1, 3))

    snpe = SnpeC(
        pilot_data=(dataset_params, dataset_stats),
        simulator=dummy_simulate,
        true_observation=x_o,
        density_estimator=neural_net,
        prior=prior,
        z_score_obs=False,
        train_with_mcmc=False,
    )

    # run inference
    num_rounds, num_simulations_per_round = 1, 1000
    posterior = snpe(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round,
    )

    # draw samples from posterior
    samples = posterior.sample(1)

    print("=======sample", samples)


test_apt_on_linearGaussian_based_on_mmd(3, "uniform", "snpe_c", 10)

# test multi-round SNPE
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_str", ("snpe_b", "snpe_c"))
def test_multi_round_snpe_on_linearGaussian_based_on_mmd(algorithm_str: str):
    """Test whether APT infers well a simple example where ground truth is available."""

    num_dim = 3
    true_observation = torch.zeros((1, num_dim))
    num_samples = 100

    prior = distributions.MultivariateNormal(
        loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
    )
    target_samples = get_true_posterior_samples_linear_gaussian_mvn_prior(
        true_observation, num_samples=num_samples
    )

    neural_net = utils.posterior_nn(model="maf", prior=prior, context=true_observation,)

    if algorithm_str == "snpe_b":
        snpe = SnpeB(
            simulator=linear_gaussian,
            true_observation=true_observation,
            density_estimator=neural_net,
            prior=prior,
            z_score_obs=True,
            simulation_batch_size=10,
            use_combined_loss=False,
            retrain_from_scratch_each_round=False,
            discard_prior_samples=False,
        )
    elif algorithm_str == "snpe_c":
        snpe = SnpeC(
            simulator=linear_gaussian,
            true_observation=true_observation,
            density_estimator=neural_net,
            prior=prior,
            num_atoms=10,
            z_score_obs=True,
            simulation_batch_size=50,
            use_combined_loss=False,
            retrain_from_scratch_each_round=False,
            discard_prior_samples=False,
            train_with_mcmc=False,
        )

    # run inference
    num_rounds, num_simulations_per_round = 2, 1000
    posterior = snpe(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    # draw samples from posterior
    samples = posterior.sample(num_samples)

    # compute the mmd
    mmd = utils.unbiased_mmd_squared(target_samples, samples)

    # check if mmd is larger than expected
    max_mmd = 0.02

    print("mmd for apt is:  ", mmd)

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."


# testing rejction and mcmc sampling methods
@pytest.mark.slow
@pytest.mark.parametrize(
    "train_with_mcmc, mcmc_method, prior",
    (
        (True, "slice-np", "gaussian"),
        (True, "slice", "gaussian"),
        # (True, "slice", "uniform"), # takes very long. fix when refactoring pyro sampling
        (False, "rejection", "gaussian"),
        (False, "rejection", "uniform"),
    ),
)
def test_apt_posterior_correction(train_with_mcmc, mcmc_method, prior):
    """Test that leakage correction applied to sampling works, with both MCMC and rejection."""

    num_dim = 2

    if prior == "gaussian":
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
        )
    else:
        prior = utils.BoxUniform(
            low=-1.0 * torch.ones(num_dim), high=torch.ones(num_dim)
        )

    true_observation = torch.zeros((1, num_dim))

    neural_net = utils.posterior_nn(model="maf", prior=prior, context=true_observation,)

    apt = SnpeC(
        simulator=linear_gaussian,
        true_observation=true_observation,
        density_estimator=neural_net,
        prior=prior,
        num_atoms=-1,
        z_score_obs=True,
        simulation_batch_size=50,
        use_combined_loss=False,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        train_with_mcmc=train_with_mcmc,
        mcmc_method=mcmc_method,
    )

    # run inference
    num_rounds, num_simulations_per_round = 1, 1000
    posterior = apt(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    # draw samples from posterior (should be corrected for leakage)
    # even if just num_rounds=1
    samples = posterior.sample(10)

    # evaluate the samples to check correction factor
    lob_probs = posterior.log_prob(samples)
