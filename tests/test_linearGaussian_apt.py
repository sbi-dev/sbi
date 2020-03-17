import pytest
import torch
from torch import distributions

import sbi.utils as utils
from sbi.inference.snpe.snpe_c import SnpeC
from sbi.simulators.linear_gaussian import (
    get_ground_truth_posterior_samples_linear_gaussian,
    linear_gaussian,
)

torch.set_default_tensor_type("torch.FloatTensor")


# seed the simulations
torch.manual_seed(0)


@pytest.mark.parametrize("num_dim", [1, 3])
def test_apt_on_linearGaussian_based_on_mmd(num_dim):
    """Test whether APT infers well a simple example where ground truth is available."""

    prior = distributions.MultivariateNormal(
        loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
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
        train_with_mcmc=False,
    )

    # run inference
    num_rounds, num_simulations_per_round = 1, 1000
    posterior = apt(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    # draw samples from posterior
    samples = posterior.sample(100)

    # define target distribution (analytically tractable) and sample from it
    target_samples = get_ground_truth_posterior_samples_linear_gaussian(
        true_observation
    )

    # compute the mmd
    mmd = utils.unbiased_mmd_squared(target_samples, samples)

    # check if mmd is larger than expected
    max_mmd = 0.02

    print("mmd for apt is:  ", mmd)

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."


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
