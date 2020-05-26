import pytest
import torch
from torch import distributions, eye, zeros, ones

import sbi.utils as utils
from tests.test_utils import (
    check_c2st,
    get_dkl_gaussian_prior,
    get_prob_outside_uniform_prior,
)
from sbi.inference.sre.sre import SRE
from sbi.simulators.linear_gaussian import (
    get_true_posterior_samples_linear_gaussian_mvn_prior,
    get_true_posterior_samples_linear_gaussian_uniform_prior,
    linear_gaussian,
)

# use cpu by default
torch.set_default_tensor_type("torch.FloatTensor")
# Seeding:
# Some tests in this module have "set_seed" as an argument. This argument points to
# tests/conftest.py to seed the test with the seed set in conftext.py.


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

    infer = SRE(
        simulator=linear_gaussian,
        prior=prior,
        x_o=x_o,
        classifier=None,  # Use default RESNET.
        simulation_batch_size=50,
        mcmc_method="slice_np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    posterior.sample(num_samples=10, num_chains=2)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_dim, prior_str, classifier_loss",
    (
        (2, "gaussian", "sre"),
        (1, "gaussian", "sre"),
        (2, "uniform", "sre"),
        (2, "gaussian", "aalr"),
    ),
)
def test_sre_on_linearGaussian_based_on_c2st(
    num_dim: int, prior_str: str, classifier_loss: str, set_seed,
):
    """Test c2st accuracy of inference with SRE on linear Gaussian model.

    Args:
        num_dim: parameter dimension of the gaussian model
        prior_str: one of "gaussian" or "uniform"
        set_seed: fixture for manual seeding
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

    num_atoms = 2 if classifier_loss == "aalr" else None

    infer = SRE(
        simulator=linear_gaussian,
        prior=prior,
        x_o=x_o,
        num_atoms=num_atoms,
        classifier=None,  # Use default RESNET.
        classifier_loss=classifier_loss,
        simulation_batch_size=50,
        mcmc_method="slice_np",
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    samples = posterior.sample(num_samples=num_samples)

    # Check performance based on c2st accuracy.
    check_c2st(samples, target_samples, alg=f"sre-{prior_str}-{classifier_loss}")

    # Checks for log_prob()
    if prior_str == "gaussian" and classifier_loss == "aalr":
        # For the Gaussian prior, we compute the KLd between ground truth and
        # posterior. We can do this only if the classifier_loss was as described in
        # Hermans et al. 2019 ('aalr') since Durkan et al. 2019 version only allows
        # evaluation up to a constant.
        # For the Gaussian prior, we compute the KLd between ground truth and posterior
        dkl = get_dkl_gaussian_prior(posterior, x_o, num_dim)

        max_dkl = 0.05 if num_dim == 1 else 0.8

        assert (
            dkl < max_dkl
        ), f"KLd={dkl} is more than 2 stds above the average performance."
    if prior_str == "uniform":
        # Check whether the returned probability outside of the support is zero.
        posterior_prob = get_prob_outside_uniform_prior(posterior, num_dim)
        assert (
            posterior_prob == 0.0
        ), "The posterior probability outside of the prior support is not zero"


@pytest.mark.slow
@pytest.mark.parametrize(
    "mcmc_method, prior_str",
    (
        ("slice_np", "gaussian"),
        ("slice_np", "uniform"),
        ("slice", "gaussian"),
        ("slice", "uniform"),
    ),
)
def test_sre_posterior_correction(mcmc_method: str, prior_str: str, set_seed):
    """Test leakage correction both for MCMC and rejection sampling.

    Args:
        mcmc_method: which mcmc method to use for sampling
        prior_str: one of "gaussian" or "uniform"
        set_seed: fixture for manual seeding
    """

    num_dim = 2
    x_o = zeros(num_dim)
    if prior_str == "gaussian":
        prior = distributions.MultivariateNormal(
            loc=zeros(num_dim), covariance_matrix=eye(num_dim)
        )
    else:
        prior = utils.BoxUniform(low=-1.0 * ones(num_dim), high=ones(num_dim))

    infer = SRE(
        simulator=linear_gaussian,
        prior=prior,
        x_o=x_o,
        classifier=None,  # Use default RESNET.
        simulation_batch_size=50,
        mcmc_method=mcmc_method,
    )

    posterior = infer(num_rounds=1, num_simulations_per_round=1000)

    posterior.sample(num_samples=30)
