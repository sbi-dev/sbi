import pytest
import torch
from torch import distributions, eye, zeros, ones

import sbi.utils as utils
import tests.utils_for_testing.linearGaussian_logprob as test_utils
from sbi.inference.sre.sre import SRE
from sbi.simulators.linear_gaussian import (
    get_true_posterior_samples_linear_gaussian_mvn_prior,
    get_true_posterior_samples_linear_gaussian_uniform_prior,
    linear_gaussian,
)
from sbi.user_input.user_input_checks import prepare_sbi_problem

# use cpu by default
torch.set_default_tensor_type("torch.FloatTensor")


@pytest.mark.parametrize("num_dim", (1, 3))
def test_sre_on_linearGaussian_api(num_dim: int):
    """Test inference API of SRE with linear Gaussian model.

    Avoids intense computation for fast testing of API etc.

    Args:
        num_dim: parameter dimension of the Gaussian model
    """

    x_o = zeros(num_dim)
    prior = distributions.MultivariateNormal(
        loc=zeros(num_dim), covariance_matrix=eye(num_dim)
    )

    # XXX this breaks the test! (and #76 doesn't seem to fix)

    simulator, prior, x_o = prepare_sbi_problem(linear_gaussian, prior, x_o)

    infer = SRE(
        simulator=simulator,
        prior=prior,
        x_o=x_o,
        classifier=None,  # Use default RESNET.
        simulation_batch_size=50,
        mcmc_method="slice-np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    posterior.sample(num_samples=10, num_chains=2)

    # XXX log_prob is not implemented yet for SRE
    # posterior.log_prob(samples)


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
    num_dim: int, prior_str: str, classifier_loss: str, set_seed,
):
    """Test MMD accuracy of inference with SRE on linear Gaussian model.

    NOTE: The mmd threshold is calculated based on a number of test runs and taking the
    mean plus 2 stds.

    This test is seeded using the set_seed fixture defined in tests/conftest.py.

    Args:
        num_dim: parameter dimension of the gaussian model
        prior_str: one of "gaussian" or "uniform"
        set_seed: fixture for manual seeding, see tests/conftest.py
    """

    x_o = zeros(1, num_dim)
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

    simulator, prior, x_o = prepare_sbi_problem(linear_gaussian, prior, x_o)

    num_atoms = 2 if classifier_loss == "aalr" else -1

    infer = SRE(
        simulator=simulator,
        prior=prior,
        x_o=x_o,
        num_atoms=num_atoms,
        classifier=None,  # Use default RESNET.
        classifier_loss=classifier_loss,
        simulation_batch_size=50,
        mcmc_method="slice-np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    samples = posterior.sample(num_samples=num_samples)

    # Check if mmd is larger than expected.
    mmd = utils.unbiased_mmd_squared(target_samples, samples)
    max_mmd = 0.045
    assert (
        mmd < max_mmd
    ), f"MMD={mmd} is more than 2 stds above the average performance."

    # Checks for log_prob()
    if prior_str == "gaussian" and classifier_loss == "aalr":
        # For the Gaussian prior, we compute the KLd between ground truth and
        # posterior. We can do this only if the classifier_loss was as described in
        # Hermans et al. 2019 ('aalr') since Durkan et al. 2019 version only allows
        # evaluation up to a constant.
        # For the Gaussian prior, we compute the KLd between ground truth and posterior
        dkl = test_utils.get_dkl_gaussian_prior(posterior, x_o, num_dim)

        max_dkl = 0.05 if num_dim == 1 else 0.8

        assert (
            dkl < max_dkl
        ), f"KLd={dkl} is more than 2 stds above the average performance."
    if prior_str == "uniform":
        # Check whether the returned probability outside of the support is zero.
        posterior_prob = test_utils.get_prob_outside_uniform_prior(posterior, num_dim)
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
def test_sre_posterior_correction(mcmc_method: str, prior_str: str, set_seed):
    """Test leakage correction both for MCMC and rejection sampling.

    This test is seeded using the set_seed fixture defined in tests/conftest.py.

    Args:
        mcmc_method: which mcmc method to use for sampling
        prior_str: one of "gaussian" or "uniform"
        set_seed: fixture for manual seeding, see tests/conftest.py
    """

    num_dim = 2
    x_o = zeros(num_dim)
    if prior_str == "gaussian":
        prior = distributions.MultivariateNormal(
            loc=zeros(num_dim), covariance_matrix=eye(num_dim)
        )
    else:
        prior = utils.BoxUniform(low=-1.0 * ones(num_dim), high=ones(num_dim))

    simulator, prior, x_o = prepare_sbi_problem(linear_gaussian, prior, x_o)

    infer = SRE(
        simulator=simulator,
        prior=prior,
        x_o=x_o,
        classifier=None,  # Use default RESNET.
        simulation_batch_size=50,
        mcmc_method=mcmc_method,
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    _ = posterior.sample(num_samples=50)

    # TODO No log prob for SRE yet - see #73.
    # posterior.log_prob(samples)
