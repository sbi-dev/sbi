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

torch.manual_seed(1)


# running all combinations is excessive. The standard test is (3, "gaussian", "snpe_c"),
# and we then vary only one parameter at a time to test single-d, uniform, and snpe-b
@pytest.mark.parametrize(
    "num_dim, prior_str, algorithm_str",
    (
        (3, "gaussian", "snpe_c"),
        (1, "gaussian", "snpe_c"),
        (3, "uniform",  "snpe_c"),
        (3, "gaussian", "snpe_b"),
    ),
)
def test_apt_on_linearGaussian_based_on_mmd(num_dim: int, prior_str: str, algorithm_str: str):
    """Test whether APT infers well a simple example where ground truth is available."""

    true_observation = torch.zeros(num_dim)
    num_samples = 100

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

    neural_net = utils.posterior_nn(model="maf", prior=prior, context=true_observation,)

    if algorithm_str == "snpe_b":
        snpe = SnpeB(
            simulator=linear_gaussian,
            true_observation=true_observation,
            density_estimator=neural_net,
            prior=prior,
            z_score_obs=True,
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
            num_atoms=-1,
            z_score_obs=True,
            simulation_batch_size=10,
            use_combined_loss=False,
            retrain_from_scratch_each_round=False,
            discard_prior_samples=False,
            train_with_mcmc=False,
        )

    # run inference
    num_rounds, num_simulations_per_round = 1, 1000
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
