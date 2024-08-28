# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import HalfNormal, MultivariateNormal

from sbi.inference import (
    NLE,
    ImportanceSamplingPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
    likelihood_estimator_based_potential,
)
from sbi.neural_nets import likelihood_nn
from sbi.simulators.linear_gaussian import (
    diagonal_linear_gaussian,
    linear_gaussian,
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior

from .test_utils import check_c2st, get_prob_outside_uniform_prior


@pytest.mark.parametrize("num_dim", (1,))  # dim 3 is tested below.
@pytest.mark.parametrize("prior_str", ("uniform", "gaussian"))
def test_api_nle_multiple_trials_and_rounds_map(
    num_dim: int, prior_str: str, mcmc_params_fast: dict
):
    """Test NLE API with 2 rounds, different priors num trials and MAP."""
    num_rounds = 2
    num_samples = 1
    num_simulations_per_round = 100

    if prior_str == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    else:
        prior = BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    simulator = diagonal_linear_gaussian
    inference = NLE(prior=prior, density_estimator="mdn", show_progress_bars=False)

    proposals = [prior]
    for _ in range(num_rounds):
        theta = proposals[-1].sample((num_simulations_per_round,))
        x = simulator(theta)
        inference.append_simulations(theta, x).train(
            training_batch_size=100, max_num_epochs=2
        )
        for num_trials in [1, 3]:
            x_o = zeros((num_trials, num_dim))
            posterior = inference.build_posterior(
                mcmc_method="slice_np_vectorized",
                mcmc_parameters=mcmc_params_fast,
            ).set_default_x(x_o)
            posterior.sample(sample_shape=(num_samples,))
        proposals.append(posterior)
        posterior.map(num_iter=1)


def test_c2st_nle_on_linear_gaussian_different_dims(
    mcmc_params_accurate: dict, model_str="maf"
):
    """Test NLE on linear Gaussian task with different theta and x dims."""

    theta_dim = 3
    x_dim = 2
    discard_dims = theta_dim - x_dim

    x_o = zeros(1, x_dim)
    num_samples = 1000
    num_simulations = 2000

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

    def simulator(theta):
        return linear_gaussian(
            theta,
            likelihood_shift,
            likelihood_cov,
            num_discarded_dims=discard_dims,
        )

    density_estimator = likelihood_nn(model=model_str, num_transforms=3)
    inference = NLE(density_estimator=density_estimator, show_progress_bars=False)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    likelihood_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = likelihood_estimator_based_potential(
        prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
    )
    posterior = MCMCPosterior(
        proposal=prior,
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        method="slice_np_vectorized",
        **mcmc_params_accurate,
    )
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"nle_a-{model_str}")


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("prior_str", ("uniform", "gaussian"))
@pytest.mark.parametrize("model_str", ("maf", "zuko_maf"))
def test_c2st_and_map_nle_on_linearGaussian_different(
    num_dim: int, prior_str: str, model_str: str, mcmc_params_accurate: dict
):
    """Test SNL on linear Gaussian, comparing to ground truth posterior via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        prior_str: one of "gaussian" or "uniform"

    """
    num_samples = 500
    num_simulations = 3000
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
        prior = BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    density_estimator = likelihood_nn(model_str, num_transforms=3)
    inference = NLE(density_estimator=density_estimator, show_progress_bars=False)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

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
            **mcmc_params_accurate,
        )

        samples = posterior.sample(sample_shape=(num_samples,))

        # Check performance based on c2st accuracy.
        check_c2st(
            samples,
            target_samples,
            alg=f"nle_a-{prior_str}-prior-{model_str}-{num_trials}-trials",
        )

        map_ = posterior.map(
            num_init_samples=1_000,
            init_method="proposal",
            show_progress_bars=False,
        )

        if prior_str == "uniform":
            # Check whether the returned probability outside of the support is zero.
            posterior_prob = get_prob_outside_uniform_prior(posterior, prior, num_dim)
            assert (
                posterior_prob == 0.0
            ), "The posterior probability outside of the prior support is not zero"

            assert ((map_ - ones(num_dim)) ** 2).sum() < 0.5
        else:
            assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5


@pytest.mark.parametrize("use_transform", (True, False))
def test_map_with_multiple_independent_prior(use_transform):
    """Test whether map works with multiple independent priors, see issue #841, #650."""

    dim = 2
    prior, *_ = process_prior([
        BoxUniform(low=-ones(dim), high=ones(dim)),
        HalfNormal(scale=ones(1) * 2),
    ])

    def simulator(theta):
        return theta[:, 2:] * torch.randn_like(theta[:, :2]) + theta[:, :2]

    num_simulations = 1000
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
    x_o = zeros((1, dim))

    trainer = NLE(prior).append_simulations(theta, x)
    likelihood_estimator = trainer.train(max_num_epochs=5)

    potential_fn, parameter_transform = likelihood_estimator_based_potential(
        likelihood_estimator,
        prior,
        x_o=x_o,
    )
    posterior = MCMCPosterior(
        potential_fn,
        proposal=prior,
        theta_transform=parameter_transform if use_transform else None,
    )
    posterior.map()
    posterior.set_default_x(x_o).map(num_iter=10)


@pytest.mark.slow
@pytest.mark.parametrize("num_trials", (1, 3))
def test_c2st_multi_round_nle_on_linearGaussian(
    num_trials: int, mcmc_params_accurate: dict
):
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

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = NLE(show_progress_bars=False)

    theta = prior.sample((num_simulations_per_round,))
    x = simulator(theta)

    likelihood_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = likelihood_estimator_based_potential(
        prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
    )
    posterior1 = MCMCPosterior(
        proposal=prior,
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        method="slice_np_vectorized",
        **mcmc_params_accurate,
    )

    theta = posterior1.sample((num_simulations_per_round,))
    x = simulator(theta)

    likelihood_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = likelihood_estimator_based_potential(
        prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
    )
    posterior = MCMCPosterior(
        proposal=prior,
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        method="slice_np_vectorized",
        **mcmc_params_accurate,
    )

    samples = posterior.sample(sample_shape=(num_samples,))

    # Check performance based on c2st accuracy.
    check_c2st(samples, target_samples, alg="multi-round-snl")


@pytest.mark.slow
@pytest.mark.parametrize("num_trials", (1, 3))
def test_c2st_multi_round_nle_on_linearGaussian_vi(num_trials: int):
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

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = NLE(show_progress_bars=False)

    theta = prior.sample((num_simulations_per_round,))
    x = simulator(theta)

    likelihood_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = likelihood_estimator_based_potential(
        prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
    )
    posterior1 = VIPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
    )
    posterior1.train()

    theta = posterior1.sample((num_simulations_per_round,))
    x = simulator(theta)

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
        pytest.param("slice_np", "gaussian", marks=pytest.mark.mcmc),
        pytest.param("slice_np", "uniform", marks=pytest.mark.mcmc),
        pytest.param("slice_np_vectorized", "gaussian", marks=pytest.mark.mcmc),
        pytest.param("slice_np_vectorized", "uniform", marks=pytest.mark.mcmc),
        pytest.param("nuts_pymc", "gaussian", marks=pytest.mark.mcmc),
        pytest.param("nuts_pyro", "uniform", marks=pytest.mark.mcmc),
        pytest.param("hmc_pymc", "gaussian", marks=pytest.mark.mcmc),
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
def test_api_nle_sampling_methods(
    sampling_method: str, prior_str: str, init_strategy: str, mcmc_params_fast: dict
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
        prior = BoxUniform(-1.0 * ones(num_dim), ones(num_dim))

    simulator = diagonal_linear_gaussian

    inference = NLE(show_progress_bars=False)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)
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
            init_strategy=init_strategy,
            **mcmc_params_fast,
        )
    elif sample_with == "importance":
        posterior = ImportanceSamplingPosterior(
            potential_fn,
            proposal=prior,
            theta_transform=theta_transform,
        )
    else:
        posterior = VIPosterior(
            potential_fn,
            theta_transform=theta_transform,
            vi_method=sampling_method,
        )
        posterior.train(max_num_iters=10)

    posterior.sample(sample_shape=(num_samples,), show_progress_bars=False)
