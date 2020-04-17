import pytest
import torch
from torch import distributions

import sbi.utils as utils
import tests.utils_for_testing.linearGaussian_logprob as test_utils
from sbi.inference.snpe.snpe_b import SnpeB
from sbi.inference.snpe.snpe_c import SnpeC
from sbi.simulators.linear_gaussian import (
    get_true_posterior_samples_linear_gaussian_mvn_prior,
    get_true_posterior_samples_linear_gaussian_uniform_prior,
    linear_gaussian,
)
from sbi.simulators.simutils import prepare_sbi_problem


# Running all combinations is excessive. The standard test is (3, "gaussian", "snpe_c"),
# and we then vary only one parameter at a time to test single-D, uniform, and snpe-b.
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
    num_dim: int,
    prior_str: str,
    algorithm_str: str,
    simulation_batch_size: int,
    set_seed,
):
    """Test whether APT infers well a simple example where ground truth is available.

    This test is seeded using the set_seed fixture defined in tests/conftest.py.

    Args:
        set_seed: fixture for manual seeding, see tests/conftest.py
    """

    x_o = torch.zeros(1, num_dim)
    num_samples = 100

    if prior_str == "gaussian":
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
        )
        target_samples = get_true_posterior_samples_linear_gaussian_mvn_prior(
            x_o, num_samples=num_samples
        )
    else:
        prior = utils.BoxUniform(-1.0 * torch.ones(num_dim), torch.ones(num_dim))
        target_samples = get_true_posterior_samples_linear_gaussian_uniform_prior(
            x_o, num_samples=num_samples, prior=prior
        )

    simulator, prior, x_o = prepare_sbi_problem(linear_gaussian, prior, x_o)

    neural_net = utils.posterior_nn(model="maf", prior=prior, x_o=x_o)

    snpe_common_args = dict(
        simulator=simulator,
        x_o=x_o,
        density_estimator=neural_net,
        prior=prior,
        z_score_x=True,
        simulation_batch_size=simulation_batch_size,
        use_combined_loss=False,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
    )

    if algorithm_str == "snpe_b":
        infer = SnpeB(**snpe_common_args)
    elif algorithm_str == "snpe_c":
        infer = SnpeC(num_atoms=-1, sample_with_mcmc=False, **snpe_common_args)

    # Run inference.
    num_rounds, num_simulations_per_round = 1, 1000
    posterior = infer(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round,
    )

    # Draw from posterior.
    samples = posterior.sample(num_samples)

    # Compute the mmd, and check if larger than expected
    mmd = utils.unbiased_mmd_squared(target_samples, samples)
    max_mmd = 0.03

    print(f"mmd for {algorithm_str} is {mmd}.")

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."

    # Checks for log_prob()
    if prior_str == "gaussian":
        # For the Gaussian prior, we compute the KLd between ground truth and posterior.
        dkl = test_utils.get_dkl_gaussian_prior(posterior, x_o, num_dim)

        max_dkl = 0.05 if num_dim == 1 else 0.8

        assert (
            dkl < max_dkl
        ), f"D-KL={dkl} is more than 2 stds above the average performance."

    elif prior_str == "uniform":
        # Check whether the returned probability outside of the support is zero.
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
        ) = test_utils.get_normalization_uniform_prior(posterior, prior, x_o)
        # The acceptance probability should be *exactly* the ratio of the unnormalized
        # and the normalized likelihood. However, we allow for an error margin of 1%,
        # since the estimation of the acceptance probability is random (based on
        # rejection sampling).
        assert (
            acceptance_prob * 0.99
            < posterior_likelihood_unnorm / posterior_likelihood_norm
            < acceptance_prob * 1.01
        ), "Normalizing the posterior density using the acceptance probability failed."


# Test multi-round SNPE.
@pytest.mark.parametrize("algorithm_str", ("snpe_b", "snpe_c"))
def test_multi_round_snpe_on_linearGaussian_based_on_mmd(algorithm_str: str, set_seed):
    """Test whether APT infers well a simple example where ground truth is available.

    This test is seeded using the set_seed fixture defined in tests/conftest.py.

    Args:
        set_seed: fixture for manual seeding, see tests/conftest.py.
    """

    num_dim = 3
    true_observation = torch.zeros((1, num_dim))
    num_samples = 100

    prior = distributions.MultivariateNormal(
        loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
    )
    target_samples = get_true_posterior_samples_linear_gaussian_mvn_prior(
        true_observation, num_samples=num_samples
    )

    simulator, prior, x_o = prepare_sbi_problem(
        linear_gaussian, prior, true_observation
    )

    neural_net = utils.posterior_nn(model="maf", prior=prior, x_o=true_observation,)

    snpe_common_args = dict(
        simulator=simulator,
        x_o=true_observation,
        density_estimator=neural_net,
        prior=prior,
        z_score_x=True,
        use_combined_loss=False,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
    )

    if algorithm_str == "snpe_b":
        infer = SnpeB(simulation_batch_size=10, **snpe_common_args)
    elif algorithm_str == "snpe_c":
        infer = SnpeC(
            num_atoms=10,
            simulation_batch_size=50,
            sample_with_mcmc=False,
            **snpe_common_args,
        )

    # Run inference.
    num_rounds, num_simulations_per_round = 2, 1000
    posterior = infer(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    # Draw from posterior.
    samples = posterior.sample(num_samples)

    # Compute the mmd, and check if larger than expected.
    mmd = utils.unbiased_mmd_squared(target_samples, samples)
    max_mmd = 0.02

    print(f"mmd for {algorithm_str} is {mmd}.")

    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."
    samples = posterior.sample(num_samples)


# Testing rejection and mcmc sampling methods.
@pytest.mark.slow
@pytest.mark.parametrize(
    "sample_with_mcmc, mcmc_method, prior",
    (
        (True, "slice-np", "gaussian"),
        (True, "slice", "gaussian"),
        # XXX (True, "slice", "uniform"),
        # XXX takes very long. fix when refactoring pyro sampling
        (False, "rejection", "gaussian"),
        (False, "rejection", "uniform"),
    ),
)
def test_apt_posterior_correction(sample_with_mcmc, mcmc_method, prior, set_seed):
    """Test that leakage correction applied to sampling works, with both MCMC and
    rejection.

    This test is seeded using the set_seed fixture defined in tests/conftest.py.

    Args:
        set_seed: fixture for manual seeding, see tests/conftest.py.
    """

    num_dim = 2

    if prior == "gaussian":
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
        )
    else:
        prior = utils.BoxUniform(
            low=-1.0 * torch.ones(num_dim), high=torch.ones(num_dim)
        )

    simulator, prior, x_o = prepare_sbi_problem(
        linear_gaussian, prior, torch.zeros(num_dim)
    )

    neural_net = utils.posterior_nn(model="maf", prior=prior, x_o=x_o,)

    infer = SnpeC(
        simulator=simulator,
        x_o=x_o,
        density_estimator=neural_net,
        prior=prior,
        num_atoms=-1,
        z_score_x=True,
        simulation_batch_size=50,
        use_combined_loss=False,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        sample_with_mcmc=sample_with_mcmc,
        mcmc_method=mcmc_method,
    )

    # Run inference.
    num_rounds, num_simulations_per_round = 1, 1000
    posterior = infer(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    # Draw samples from posterior (should be corrected for leakage)
    # even if just num_rounds=1.
    samples = posterior.sample(10)

    # Evaluate the samples to check correction factor.
    posterior.log_prob(samples)
