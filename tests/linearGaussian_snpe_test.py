# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import SNPE_B, SNPE_C, prepare_for_sbi
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.torchutils import configure_default_device
from tests.test_utils import (
    check_c2st,
    get_dkl_gaussian_prior,
    get_normalization_uniform_prior,
    get_prob_outside_uniform_prior,
)


@pytest.mark.parametrize(
    "num_dim, prior_str", ((2, "gaussian"), (2, "uniform"), (1, "gaussian"),),
)
def test_c2st_snpe_on_linearGaussian(
    num_dim: int, prior_str: str, set_seed,
):
    """Test whether SNPE C infers well a simple example with available ground truth.

    Args:
        set_seed: fixture for manual seeding
    """

    device = "cpu"
    configure_default_device(device)
    x_o = zeros(1, num_dim)
    num_samples = 1000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    if prior_str == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    else:
        prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o, likelihood_shift, likelihood_cov, prior=prior, num_samples=num_samples
        )

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SNPE_C(
        *prepare_for_sbi(simulator, prior),
        simulation_batch_size=1000,
        show_progress_bars=False,
        sample_with_mcmc=False,
        device=device,
    )

    posterior = infer(num_simulations=2000, training_batch_size=100).set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="snpe_c")

    # Checks for log_prob()
    if prior_str == "gaussian":
        # For the Gaussian prior, we compute the KLd between ground truth and posterior.
        dkl = get_dkl_gaussian_prior(
            posterior, x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )

        max_dkl = 0.15

        assert (
            dkl < max_dkl
        ), f"D-KL={dkl} is more than 2 stds above the average performance."

    elif prior_str == "uniform":
        # Check whether the returned probability outside of the support is zero.
        posterior_prob = get_prob_outside_uniform_prior(posterior, num_dim)
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
        ) = get_normalization_uniform_prior(posterior, prior, x_o)
        # The acceptance probability should be *exactly* the ratio of the unnormalized
        # and the normalized likelihood. However, we allow for an error margin of 1%,
        # since the estimation of the acceptance probability is random (based on
        # rejection sampling).
        assert (
            acceptance_prob * 0.99
            < posterior_likelihood_unnorm / posterior_likelihood_norm
            < acceptance_prob * 1.01
        ), "Normalizing the posterior density using the acceptance probability failed."


def test_c2st_snpe_on_linearGaussian_different_dims(set_seed):
    """Test whether SNPE B/C infer well a simple example with available ground truth.

    This example has different number of parameters theta than number of x. Also
    this implicitly tests simulation_batch_size=1.

    Args:
        set_seed: fixture for manual seeding
    """

    theta_dim = 3
    x_dim = 2
    discard_dims = theta_dim - x_dim

    x_o = zeros(1, x_dim)
    num_samples = 1000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(x_dim)
    likelihood_cov = 0.3 * eye(x_dim)

    prior_mean = zeros(theta_dim)
    prior_cov = eye(theta_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    target_samples = samples_true_posterior_linear_gaussian_mvn_prior_different_dims(
        x_o[0],
        likelihood_shift,
        likelihood_cov,
        prior_mean,
        prior_cov,
        num_discarded_dims=discard_dims,
        num_samples=num_samples,
    )

    def simulator(theta):
        return linear_gaussian(
            theta, likelihood_shift, likelihood_cov, num_discarded_dims=discard_dims
        )

    infer = SNPE_C(
        *prepare_for_sbi(simulator, prior),
        density_estimator="maf",
        simulation_batch_size=1,
        show_progress_bars=False,
        sample_with_mcmc=False,
    )

    posterior = infer(num_simulations=2000)  # type: ignore
    samples = posterior.sample((num_samples,), x=x_o)

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="snpe_c")


# Test multi-round SNPE.
@pytest.mark.slow
@pytest.mark.parametrize(
    "method_str",
    (
        pytest.param(
            "snpe_b",
            marks=pytest.mark.xfail(
                raises=NotImplementedError, reason="""SNPE-B not implemented""",
            ),
        ),
        "snpe_c",
        "snpe_c_non_atomic",
    ),
)
def test_c2st_multi_round_snpe_on_linearGaussian(method_str: str, set_seed):
    """Test whether SNPE B/C infer well a simple example with available ground truth.

    Args:
        set_seed: fixture for manual seeding.
    """

    num_dim = 2
    x_o = zeros((1, num_dim))
    num_samples = 1000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    if method_str == "snpe_c_non_atomic":
        density_estimator = utils.posterior_nn("mdn", num_components=5)
        method_str = "snpe_c"
    else:
        density_estimator = "maf"

    simulator, prior = prepare_for_sbi(simulator, prior)
    creation_args = dict(
        simulator=simulator,
        prior=prior,
        density_estimator=density_estimator,
        show_progress_bars=False,
    )

    if method_str == "snpe_b":
        infer = SNPE_B(simulation_batch_size=10, **creation_args)
        posterior1 = infer(num_simulations=1000)
        posterior1.set_default_x(x_o)
        posterior = infer(num_simulations=1000, proposal=posterior1)
    elif method_str == "snpe_c":
        infer = SNPE_C(
            simulation_batch_size=50, sample_with_mcmc=False, **creation_args
        )
        posterior = infer(num_simulations=500, num_atoms=10).set_default_x(x_o)
        posterior = infer(
            num_simulations=1000, num_atoms=10, proposal=posterior
        ).set_default_x(x_o)

    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=method_str)


@pytest.mark.slow
def test_c2st_snpe_external_data_on_linearGaussian(set_seed):
    """Test whether SNPE C infers well a simple example with available ground truth.

    Args:
        set_seed: fixture for manual seeding
    """

    num_dim = 2

    device = "cpu"
    configure_default_device(device)
    x_o = zeros(1, num_dim)
    num_samples = 1000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SNPE_C(
        *prepare_for_sbi(simulator, prior),
        simulation_batch_size=1000,
        density_estimator="nsf",
        show_progress_bars=False,
        sample_with_mcmc=False,
        device=device,
    )

    external_theta = prior.sample((500,))
    external_x = simulator(external_theta)

    infer.provide_presimulated(external_theta, external_x)

    posterior = infer(num_simulations=500, training_batch_size=50).set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="snpe_c")


# Testing rejection and mcmc sampling methods.
@pytest.mark.slow
@pytest.mark.parametrize(
    "sample_with_mcmc, mcmc_method, prior_str",
    (
        (True, "slice_np", "gaussian"),
        (True, "slice", "gaussian"),
        # XXX (True, "slice", "uniform"),
        # XXX takes very long. fix when refactoring pyro sampling
        (False, "rejection", "uniform"),
    ),
)
def test_api_snpe_c_posterior_correction(
    sample_with_mcmc, mcmc_method, prior_str, set_seed
):
    """Test that leakage correction applied to sampling works, with both MCMC and
    rejection.

    Args:
        set_seed: fixture for manual seeding
    """

    num_dim = 2
    x_o = zeros(1, num_dim)

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    if prior_str == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    else:
        prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SNPE_C(
        *prepare_for_sbi(simulator, prior),
        density_estimator="maf",
        simulation_batch_size=50,
        sample_with_mcmc=sample_with_mcmc,
        mcmc_method=mcmc_method,
        show_progress_bars=False,
    )

    posterior = infer(num_simulations=1000, max_num_epochs=5)

    # Posterior should be corrected for leakage even if num_rounds just 1.
    samples = posterior.sample((10,), x=x_o)

    # Evaluate the samples to check correction factor.
    posterior.log_prob(samples, x=x_o)


def example_posterior():
    """Return an inferred `NeuralPosterior` for interactive examination."""
    num_dim = 2
    x_o = zeros(1, num_dim)

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SNPE_C(
        *prepare_for_sbi(simulator, prior),
        simulation_batch_size=10,
        num_workers=6,
        show_progress_bars=False,
        sample_with_mcmc=False,
    )

    return infer(num_simulations=1000).set_default_x(x_o)
