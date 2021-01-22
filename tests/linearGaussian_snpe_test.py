# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import gaussian_kde
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import SNPE_B, SNPE_C, prepare_for_sbi, simulate_for_sbi
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from tests.test_utils import (
    check_c2st,
    get_dkl_gaussian_prior,
    get_normalization_uniform_prior,
    get_prob_outside_uniform_prior,
)


@pytest.mark.parametrize(
    "num_dim, prior_str",
    (
        (2, "gaussian"),
        (2, "uniform"),
        (1, "gaussian"),
    ),
)
def test_c2st_snpe_on_linearGaussian(
    num_dim: int,
    prior_str: str,
    set_seed,
):
    """Test whether SNPE C infers well a simple example with available ground truth.

    Args:
        set_seed: fixture for manual seeding
    """

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

    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE_C(
        prior,
        show_progress_bars=False,
    )

    theta, x = simulate_for_sbi(simulator, prior, 2000, simulation_batch_size=1000)
    _ = inference.append_simulations(theta, x).train(training_batch_size=100)
    posterior = inference.build_posterior().set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="snpe_c")

    map_ = posterior.map(num_init_samples=1_000)

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

        assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5

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

        assert ((map_ - ones(num_dim)) ** 2).sum() < 0.5


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

    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE_C(
        prior,
        density_estimator="maf",
        show_progress_bars=False,
    )

    theta, x = simulate_for_sbi(simulator, prior, 2000, simulation_batch_size=1)  # type: ignore
    _ = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()
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
                raises=NotImplementedError,
                reason="""SNPE-B not implemented""",
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
        inference = SNPE_B(**creation_args)
        theta, x = simulate_for_sbi(simulator, prior, 500, simulation_batch_size=10)
        _ = inference.append_simulations(theta, x).train()
        posterior1 = inference.build_posterior().set_default_x(x_o)
        theta, x = simulate_for_sbi(
            simulator, posterior1, 1000, simulation_batch_size=10
        )
        _ = inference.append_simulations(theta, x, proposal=posterior1).train()
        posterior = inference.build_posterior().set_default_x(x_o)
    elif method_str == "snpe_c":
        inference = SNPE_C(**creation_args)
        theta, x = simulate_for_sbi(simulator, prior, 500, simulation_batch_size=50)
        _ = inference.append_simulations(theta, x).train()
        posterior1 = inference.build_posterior().set_default_x(x_o)
        theta, x = simulate_for_sbi(
            simulator, posterior1, 1000, simulation_batch_size=50
        )
        _ = inference.append_simulations(theta, x, proposal=posterior1).train()
        posterior = inference.build_posterior().set_default_x(x_o)

    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=method_str)


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

    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE_C(
        prior,
        density_estimator="maf",
        simulation_batch_size=50,
        sample_with_mcmc=sample_with_mcmc,
        mcmc_method=mcmc_method,
        show_progress_bars=False,
    )

    theta, x = simulate_for_sbi(simulator, prior, 1000)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=5)
    posterior = inference.build_posterior()
    posterior = posterior.set_sample_with_mcmc(sample_with_mcmc).set_mcmc_method(
        mcmc_method
    )

    # Posterior should be corrected for leakage even if num_rounds just 1.
    samples = posterior.sample((10,), x=x_o)

    # Evaluate the samples to check correction factor.
    posterior.log_prob(samples, x=x_o)


@pytest.mark.slow
def test_sample_conditional(set_seed):
    """
    Test whether sampling from the conditional gives the same results as evaluating.

    This compares samples that get smoothed with a Gaussian kde to evaluating the
    conditional log-probability with `eval_conditional_density`.

    `eval_conditional_density` is itself tested in `sbiutils_test.py`. Here, we use
    a bimodal posterior to test the conditional.
    """

    num_dim = 3
    dim_to_sample_1 = 0
    dim_to_sample_2 = 2

    x_o = zeros(1, num_dim)

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.1 * eye(num_dim)

    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    def simulator(theta):
        if torch.rand(1) > 0.5:
            return linear_gaussian(theta, likelihood_shift, likelihood_cov)
        else:
            return linear_gaussian(theta, -likelihood_shift, likelihood_cov)

    net = utils.posterior_nn("maf", hidden_features=20)

    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE_C(
        prior,
        density_estimator=net,
        show_progress_bars=False,
    )

    # We need a pretty big dataset to properly model the bimodality.
    theta, x = simulate_for_sbi(simulator, prior, 10000)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=50)
    posterior = inference.build_posterior().set_default_x(x_o)
    samples = posterior.sample((50,))

    # Evaluate the conditional density be drawing samples and smoothing with a Gaussian
    # kde.
    cond_samples = posterior.sample_conditional(
        (500,), condition=samples[0], dims_to_sample=[dim_to_sample_1, dim_to_sample_2]
    )
    _ = analysis.pairplot(
        cond_samples,
        limits=[[-2, 2], [-2, 2], [-2, 2]],
        fig_size=(2, 2),
        diag="kde",
        upper="kde",
    )

    limits = [[-2, 2], [-2, 2], [-2, 2]]

    density = gaussian_kde(cond_samples.numpy().T, bw_method="scott")

    X, Y = np.meshgrid(
        np.linspace(
            limits[0][0],
            limits[0][1],
            50,
        ),
        np.linspace(
            limits[1][0],
            limits[1][1],
            50,
        ),
    )
    positions = np.vstack([X.ravel(), Y.ravel()])
    sample_kde_grid = np.reshape(density(positions).T, X.shape)

    # Evaluate the conditional with eval_conditional_density.
    eval_grid = analysis.eval_conditional_density(
        posterior,
        condition=samples[0],
        dim1=dim_to_sample_1,
        dim2=dim_to_sample_2,
        limits=torch.tensor([[-2, 2], [-2, 2], [-2, 2]]),
    )

    # Compare the two densities.
    sample_kde_grid = sample_kde_grid / np.sum(sample_kde_grid)
    eval_grid = eval_grid / torch.sum(eval_grid)

    error = np.abs(sample_kde_grid - eval_grid.numpy())

    max_err = np.max(error)
    assert max_err < 0.0025


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

    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE_C(
        prior,
        show_progress_bars=False,
    )
    theta, x = simulate_for_sbi(
        simulator, prior, 1000, simulation_batch_size=10, num_workers=6
    )
    _ = inference.append_simulations(theta, x).train()
    return inference.build_posterior().set_default_x(x_o)
