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
import tests.utils_for_testing.linearGaussian_logprob as test_utils

torch.manual_seed(0)

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

    neural_net = utils.posterior_nn(model="maf", prior=prior, context=true_observation)

    if algorithm_str == "snpe_b":
        snpe = SnpeB(
            simulator=linear_gaussian,
            true_observation=true_observation,
            density_estimator=neural_net,
            prior=prior,
            z_score_obs=True,
            simulation_batch_size=simulation_batch_size,
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
            simulation_batch_size=simulation_batch_size,
            use_combined_loss=False,
            retrain_from_scratch_each_round=False,
            discard_prior_samples=False,
            sample_with_mcmc=False,
        )

    # run inference
    num_rounds, num_simulations_per_round = 1, 1000
    posterior = snpe(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round,
    )

    # draw samples from posterior
    samples = posterior.sample(num_samples)

    # compute the mmd
    mmd = utils.unbiased_mmd_squared(target_samples, samples)

    # check if mmd is larger than expected
    max_mmd = 0.03

    print("mmd for apt is:  ", mmd)

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."

    # Checks for log_prob()
    if prior_str == "gaussian":
        # For the Gaussian prior, we compute the D-KL between ground truth and posterior
        dkl = test_utils.get_dkl_gaussian_prior(posterior, true_observation, num_dim)

        max_dkl = 0.05 if num_dim == 1 else 0.8

        assert (
            dkl < max_dkl
        ), f"D-KL={dkl} is more than 2 stds above the average performance."

    elif prior_str == "uniform":
        # Check whether the returned probability outside of the support is zero
        posterior_prob = test_utils.get_prob_outside_uniform_prior(posterior, num_dim)
        assert (
            posterior_prob == 0.0
        ), "The posterior probability outside of the prior support is not zero"

        # Check whether normalization (i.e. scaling up the density due
        # to leakage into regions without prior support) scales up the density by the
        # correct factor.
        (
            posterior_likelihood_unnorm,
            posterior_likelihood_norm,
            acceptance_prob,
        ) = test_utils.get_normalization_uniform_prior(
            posterior, prior, true_observation
        )
        # The acceptance probability should be *exactly* the ratio of the unnormalized
        # and the normalized likelihood. However, we allow for an error margin of 1%,
        # since the estimation of the acceptance probability is random (based on
        # rejection sampling).
        assert (
            acceptance_prob * 0.99
            < posterior_likelihood_unnorm / posterior_likelihood_norm
            < acceptance_prob * 1.01
        ), "Normalizing the posterior density using the acceptance probability failed."


# test multi-round SNPE
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
            sample_with_mcmc=False,
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
    "sample_with_mcmc, mcmc_method, prior",
    (
        (True, "slice-np", "gaussian"),
        (True, "slice", "gaussian"),
        # (True, "slice", "uniform"), # takes very long. fix when refactoring pyro sampling
        (False, "rejection", "gaussian"),
        (False, "rejection", "uniform"),
    ),
)
def test_apt_posterior_correction(sample_with_mcmc, mcmc_method, prior):
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
        sample_with_mcmc=sample_with_mcmc,
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
