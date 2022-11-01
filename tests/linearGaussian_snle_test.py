# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import (
    SNLE,
    ImportanceSamplingPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
    likelihood_estimator_based_potential,
    prepare_for_sbi,
    simulate_for_sbi,
)
from sbi.simulators.linear_gaussian import (
    diagonal_linear_gaussian,
    linear_gaussian,
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils import likelihood_nn
from tests.test_utils import check_c2st, get_prob_outside_uniform_prior


@pytest.mark.parametrize("num_dim", (1, 3))
def test_api_snl_on_linearGaussian(num_dim: int):
    """Test API for inference on linear Gaussian model using SNL.

    Avoids expensive computations by training on few simulations and generating few
    posterior samples.

    Args:
        num_dim: parameter dimension of the gaussian model
    """
    num_samples = 10

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    simulator, prior = prepare_for_sbi(diagonal_linear_gaussian, prior)
    density_estimator = likelihood_nn("maf", num_transforms=3)
    inference = SNLE(density_estimator=density_estimator, show_progress_bars=False)

    theta, x = simulate_for_sbi(simulator, prior, 1000, simulation_batch_size=50)
    likelihood_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=100
    )

    for num_trials in [1, 2]:
        x_o = zeros((num_trials, num_dim))
        potential_fn, theta_transform = likelihood_estimator_based_potential(
            prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
        )
        posterior = MCMCPosterior(
            proposal=prior,
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            thin=3,
        )
        posterior.sample(sample_shape=(num_samples,))


def test_c2st_snl_on_linearGaussian():
    """Test whether SNL infers well a simple example with available ground truth.

    This example has different number of parameters theta than number of x. This test
    also acts as the only functional test for SNL not marked as slow.

    """

    theta_dim = 3
    x_dim = 2
    discard_dims = theta_dim - x_dim

    x_o = zeros(1, x_dim)
    num_samples = 1000
    num_simulations = 3000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(x_dim)
    likelihood_cov = 0.3 * eye(x_dim)

    prior_mean = zeros(theta_dim)
    prior_cov = eye(theta_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    target_samples = samples_true_posterior_linear_gaussian_mvn_prior_different_dims(
        x_o,
        likelihood_shift,
        likelihood_cov,
        prior_mean,
        prior_cov,
        num_discarded_dims=discard_dims,
        num_samples=num_samples,
    )
    simulator, prior = prepare_for_sbi(
        lambda theta: linear_gaussian(
            theta, likelihood_shift, likelihood_cov, num_discarded_dims=discard_dims
        ),
        prior,
    )
    density_estimator = likelihood_nn("maf", num_transforms=3)
    inference = SNLE(density_estimator=density_estimator, show_progress_bars=False)

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=50
    )
    likelihood_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = likelihood_estimator_based_potential(
        prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
    )
    posterior = MCMCPosterior(
        proposal=prior,
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        method="slice_np_vectorized",
        num_chains=5,
        thin=10,
    )
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="snle_a")


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("prior_str", ("uniform", "gaussian"))
def test_c2st_and_map_snl_on_linearGaussian_different(num_dim: int, prior_str: str):
    """Test SNL on linear Gaussian, comparing to ground truth posterior via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        prior_str: one of "gaussian" or "uniform"

    """
    num_samples = 500
    num_simulations = 4500
    trials_to_test = [1]

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    # Use increased cov to avoid too small posterior cov for many trials.
    likelihood_cov = 0.8 * eye(num_dim)

    if prior_str == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    else:
        prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    simulator, prior = prepare_for_sbi(
        lambda theta: linear_gaussian(theta, likelihood_shift, likelihood_cov), prior
    )
    density_estimator = likelihood_nn("maf", num_transforms=3)
    inference = SNLE(density_estimator=density_estimator, show_progress_bars=False)

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=10000
    )
    likelihood_estimator = inference.append_simulations(theta, x).train()

    # Test inference amortized over trials.
    for num_trials in trials_to_test:
        x_o = zeros((num_trials, num_dim))
        if prior_str == "gaussian":
            gt_posterior = true_posterior_linear_gaussian_mvn_prior(
                x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
            )
            target_samples = gt_posterior.sample((num_samples,))
        elif prior_str == "uniform":
            target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
                x_o,
                likelihood_shift,
                likelihood_cov,
                prior=prior,
                num_samples=num_samples,
            )
        else:
            raise ValueError(f"Wrong prior_str: '{prior_str}'.")

        potential_fn, theta_transform = likelihood_estimator_based_potential(
            prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
        )
        posterior = MCMCPosterior(
            proposal=prior,
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            method="slice_np_vectorized",
            thin=5,
            num_chains=5,
        )

        samples = posterior.sample(sample_shape=(num_samples,))

        # Check performance based on c2st accuracy.
        check_c2st(
            samples, target_samples, alg=f"snle_a-{prior_str}-prior-{num_trials}-trials"
        )

        map_ = posterior.map(
            num_init_samples=1_000, init_method="proposal", show_progress_bars=False
        )

        # TODO: we do not have a test for SNL log_prob(). This is because the output
        # TODO: density is not normalized, so KLd does not make sense.
        if prior_str == "uniform":
            # Check whether the returned probability outside of the support is zero.
            posterior_prob = get_prob_outside_uniform_prior(posterior, prior, num_dim)
            assert (
                posterior_prob == 0.0
            ), "The posterior probability outside of the prior support is not zero"

            assert ((map_ - ones(num_dim)) ** 2).sum() < 0.5
        else:
            assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5


@pytest.mark.slow
@pytest.mark.parametrize("num_trials", (1, 3))
def test_c2st_multi_round_snl_on_linearGaussian(num_trials: int):
    """Test SNL on linear Gaussian, comparing to ground truth posterior via c2st."""

    num_dim = 2
    x_o = zeros((num_trials, num_dim))
    num_samples = 500
    num_simulations_per_round = 600 * num_trials

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    simulator, prior = prepare_for_sbi(
        lambda theta: linear_gaussian(theta, likelihood_shift, likelihood_cov), prior
    )
    inference = SNLE(show_progress_bars=False)

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations_per_round, simulation_batch_size=50
    )
    likelihood_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = likelihood_estimator_based_potential(
        prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
    )
    posterior1 = MCMCPosterior(
        proposal=prior,
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        thin=5,
        num_chains=20,
    )

    theta, x = simulate_for_sbi(
        simulator, posterior1, num_simulations_per_round, simulation_batch_size=50
    )
    likelihood_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = likelihood_estimator_based_potential(
        prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
    )
    posterior = MCMCPosterior(
        proposal=prior,
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        thin=5,
        num_chains=20,
    )

    samples = posterior.sample(sample_shape=(num_samples,))

    # Check performance based on c2st accuracy.
    check_c2st(samples, target_samples, alg="multi-round-snl")


@pytest.mark.slow
@pytest.mark.parametrize("num_trials", (1, 3))
def test_c2st_multi_round_snl_on_linearGaussian_vi(num_trials: int):
    """Test SNL on linear Gaussian, comparing to ground truth posterior via c2st."""

    num_dim = 2
    x_o = zeros((num_trials, num_dim))
    num_samples = 500
    num_simulations_per_round = 500 * num_trials

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    simulator, prior = prepare_for_sbi(
        lambda theta: linear_gaussian(theta, likelihood_shift, likelihood_cov), prior
    )
    inference = SNLE(show_progress_bars=False)

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations_per_round, simulation_batch_size=50
    )
    likelihood_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = likelihood_estimator_based_potential(
        prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
    )
    posterior1 = VIPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
    )
    posterior1.train()

    theta, x = simulate_for_sbi(
        simulator, posterior1, num_simulations_per_round, simulation_batch_size=50
    )
    likelihood_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = likelihood_estimator_based_potential(
        prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
    )
    posterior = VIPosterior(
        potential_fn=potential_fn, theta_transform=theta_transform, q=posterior1
    )
    posterior.train(eps=1e-7)

    samples = posterior.sample(sample_shape=(num_samples,))

    # Check performance based on c2st accuracy.
    check_c2st(samples, target_samples, alg="multi-round-snl")


@pytest.mark.slow
@pytest.mark.parametrize(
    "sampling_method, prior_str",
    (
        ("slice_np", "gaussian"),
        ("slice_np", "uniform"),
        ("slice_np_vectorized", "gaussian"),
        ("slice_np_vectorized", "uniform"),
        ("slice", "gaussian"),
        ("slice", "uniform"),
        ("nuts", "gaussian"),
        ("nuts", "uniform"),
        ("hmc", "gaussian"),
        ("rejection", "uniform"),
        ("rejection", "gaussian"),
        ("rKL", "uniform"),
        ("fKL", "uniform"),
        ("IW", "uniform"),
        ("alpha", "uniform"),
        ("rKL", "gaussian"),
        ("fKL", "gaussian"),
        ("IW", "gaussian"),
        ("alpha", "gaussian"),
        ("importance", "uniform"),
        ("importance", "gaussian"),
    ),
)
@pytest.mark.parametrize("init_strategy", ("proposal", "resample", "sir"))
def test_api_snl_sampling_methods(
    sampling_method: str, prior_str: str, init_strategy: str
):
    """Runs SNL on linear Gaussian and tests sampling from posterior via mcmc.

    Args:
        mcmc_method: which mcmc method to use for sampling
        prior_str: use gaussian or uniform prior

    """

    num_dim = 2
    num_samples = 10
    num_trials = 2
    num_simulations = 1000
    x_o = zeros((num_trials, num_dim))
    # Test for multiple chains is cheap when vectorized.
    num_chains = 3 if sampling_method == "slice_np_vectorized" else 1
    if sampling_method == "rejection":
        sample_with = "rejection"
    elif (
        "slice" in sampling_method
        or "nuts" in sampling_method
        or "hmc" in sampling_method
    ):
        sample_with = "mcmc"
    elif sampling_method == "importance":
        sample_with = "importance"
    else:
        sample_with = "vi"

    if prior_str == "gaussian":
        prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    else:
        prior = utils.BoxUniform(-1.0 * ones(num_dim), ones(num_dim))

    # Why do we have this if-case? Only the `MCMCPosterior` uses the `init_strategy`.
    # Thus, we would not like to run, e.g., VI with all init_strategies, but only once
    # (namely with `init_strategy=proposal`).
    if sample_with == "mcmc" or init_strategy == "proposal":
        simulator, prior = prepare_for_sbi(diagonal_linear_gaussian, prior)
        inference = SNLE(show_progress_bars=False)

        theta, x = simulate_for_sbi(
            simulator, prior, num_simulations, simulation_batch_size=1000
        )
        likelihood_estimator = inference.append_simulations(theta, x).train(
            max_num_epochs=5
        )
        potential_fn, theta_transform = likelihood_estimator_based_potential(
            prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
        )
        if sample_with == "rejection":
            posterior = RejectionPosterior(potential_fn=potential_fn, proposal=prior)
        elif (
            "slice" in sampling_method
            or "nuts" in sampling_method
            or "hmc" in sampling_method
        ):
            posterior = MCMCPosterior(
                potential_fn,
                proposal=prior,
                theta_transform=theta_transform,
                method=sampling_method,
                thin=3,
                num_chains=num_chains,
                init_strategy=init_strategy,
            )
        elif sample_with == "importance":
            posterior = ImportanceSamplingPosterior(
                potential_fn,
                proposal=prior,
                theta_transform=theta_transform,
            )
        else:
            posterior = VIPosterior(
                potential_fn, theta_transform=theta_transform, vi_method=sampling_method
            )
            posterior.train(max_num_iters=10)

        posterior.sample(sample_shape=(num_samples,))
