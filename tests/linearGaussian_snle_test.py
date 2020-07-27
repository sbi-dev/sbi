# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import SNL, prepare_for_sbi
from sbi.simulators.linear_gaussian import (
    diagonal_linear_gaussian,
    linear_gaussian,
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.torchutils import configure_default_device
from tests.test_utils import check_c2st, get_prob_outside_uniform_prior


@pytest.mark.parametrize("num_dim", (1, 3))
def test_api_snl_on_linearGaussian(num_dim: int, set_seed):
    """Test API for inference on linear Gaussian model using SNL.

    Avoids expensive computations by training on few simulations and generating few
    posterior samples.

    Args:
        num_dim: parameter dimension of the gaussian model
    """
    num_samples = 10
    x_o = zeros(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    infer = SNL(
        *prepare_for_sbi(diagonal_linear_gaussian, prior),
        simulation_batch_size=50,
        mcmc_method="slice_np",
        show_progress_bars=False,
    )

    posterior = infer(num_simulations=1000, max_num_epochs=5)

    posterior.sample(sample_shape=(num_samples,), x=x_o, mcmc_parameters={"thin": 3})


def test_c2st_snl_on_linearGaussian_different_dims(set_seed):
    """Test whether SNL infers well a simple example with available ground truth.

    This example has different number of parameters theta than number of x. This test
    also acts as the only functional test for SNL not marked as slow.

    Args:
        set_seed: fixture for manual seeding
    """

    device = "cpu"
    configure_default_device(device)
    theta_dim = 3
    x_dim = 2
    discard_dims = theta_dim - x_dim

    x_o = ones(1, x_dim)
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

    simulator = lambda theta: linear_gaussian(
        theta, likelihood_shift, likelihood_cov, num_discarded_dims=discard_dims
    )

    infer = SNL(
        *prepare_for_sbi(simulator, prior),
        simulation_batch_size=50,
        mcmc_method="slice_np",
        show_progress_bars=False,
        device=device,
    )

    posterior = infer(num_simulations=5000)  # type: ignore
    samples = posterior.sample((num_samples,), x=x_o, mcmc_parameters={"thin": 3})

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="snle_a")


@pytest.mark.slow
def test_c2st_snle_external_data_on_linearGaussian(set_seed):
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

    infer = SNL(
        *prepare_for_sbi(simulator, prior),
        simulation_batch_size=1000,
        show_progress_bars=False,
        device=device,
    )

    external_theta = prior.sample((1000,))
    external_x = simulator(external_theta)

    infer.provide_presimulated(external_theta, external_x)

    posterior = infer(num_simulations=1000, training_batch_size=100).set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="snpe_c")


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("prior_str", ("uniform", "gaussian"))
def test_c2st_snl_on_linearGaussian(num_dim: int, prior_str: str, set_seed):
    """Test SNL on linear Gaussian, comparing to ground truth posterior via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        prior_str: one of "gaussian" or "uniform"
        set_seed: fixture for manual seeding
    """

    x_o = zeros((1, num_dim))
    num_samples = 500

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

    simulator = lambda theta: linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SNL(
        *prepare_for_sbi(simulator, prior),
        mcmc_method="slice_np",
        show_progress_bars=False,
    )

    posterior = infer(num_simulations=1000).set_default_x(x_o)

    samples = posterior.sample(sample_shape=(num_samples,), mcmc_parameters={"thin": 3})

    # Check performance based on c2st accuracy.
    check_c2st(samples, target_samples, alg=f"snle_a-{prior_str}-prior")

    # TODO: we do not have a test for SNL log_prob(). This is because the output
    # TODO: density is not normalized, so KLd does not make sense.
    if prior_str == "uniform":
        # Check whether the returned probability outside of the support is zero.
        posterior_prob = get_prob_outside_uniform_prior(posterior, num_dim)
        assert (
            posterior_prob == 0.0
        ), "The posterior probability outside of the prior support is not zero"


@pytest.mark.slow
def test_c2st_multi_round_snl_on_linearGaussian(set_seed):
    """Test SNL on linear Gaussian, comparing to ground truth posterior via c2st.

    Args:
        set_seed: fixture for manual seeding
    """

    num_dim = 2
    x_o = zeros((1, num_dim))
    num_samples = 500

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

    simulator = lambda theta: linear_gaussian(theta, likelihood_shift, likelihood_cov)

    infer = SNL(
        *prepare_for_sbi(simulator, prior),
        simulation_batch_size=50,
        mcmc_method="slice",
        show_progress_bars=False,
    )

    posterior1 = infer(num_simulations=500).set_default_x(x_o)
    posterior = infer(num_simulations=500, proposal=posterior1).set_default_x(x_o)

    samples = posterior.sample(sample_shape=(num_samples,), mcmc_parameters={"thin": 3})

    # Check performance based on c2st accuracy.
    check_c2st(samples, target_samples, alg="multi-round-snl")


@pytest.mark.slow
@pytest.mark.parametrize(
    "mcmc_method, prior_str", (("slice", "gaussian"), ("slice", "uniform"),),
)
def test_api_snl_sampling_methods(mcmc_method: str, prior_str: str, set_seed):
    """Runs SNL on linear Gaussian and tests sampling from posterior via mcmc.

    Args:
        mcmc_method: which mcmc method to use for sampling
        prior_str: use gaussian or uniform prior
        set_seed: fixture for manual seeding
    """

    num_dim = 2
    num_samples = 10
    x_o = zeros((1, num_dim))

    if prior_str == "gaussian":
        prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    else:
        prior = utils.BoxUniform(-1.0 * ones(num_dim), ones(num_dim))

    infer = SNL(
        *prepare_for_sbi(diagonal_linear_gaussian, prior),
        simulation_batch_size=50,
        mcmc_method="slice_np",
        show_progress_bars=False,
    )

    posterior = infer(num_simulations=200, max_num_epochs=5)

    posterior.sample(sample_shape=(num_samples,), x=x_o, mcmc_parameters={"thin": 3})
