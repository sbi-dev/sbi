# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import (
    NPE_A,
    NPE_C,
    SNL,
    SRE,
    DirectPosterior,
    simulate_for_sbi,
)
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    samples_true_posterior_linear_gaussian_uniform_prior,
)
from sbi.utils import BoxUniform, RestrictionEstimator
from sbi.utils.sbiutils import handle_invalid_x
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from .test_utils import check_c2st


@pytest.mark.parametrize(
    "x_shape",
    (
        torch.Size((10, 1)),
        torch.Size((10, 10)),
    ),
)
def test_handle_invalid_x(x_shape):
    x = torch.rand(x_shape)
    x[x < 0.1] = float("nan")
    x[x > 0.9] = float("inf")
    x[-1, :] = 1.0  # make sure there is one row of valid entries.

    x_is_valid, *_ = handle_invalid_x(x, exclude_invalid_x=True)

    assert torch.isfinite(x[x_is_valid]).all()


@pytest.mark.parametrize("snpe_method", [NPE_A, NPE_C])
def test_z_scoring_warning(snpe_method: type):
    # Create data with large variance.
    num_dim = 2
    theta = torch.ones(100, num_dim)
    x = torch.rand(100, num_dim)
    x[:50] += 1e7

    # Make sure a warning is raised because z-scoring will map these data to duplicate
    # data points.
    with pytest.warns(UserWarning, match="Z-scoring these simulation outputs"):
        snpe_method(utils.BoxUniform(zeros(num_dim), ones(num_dim))).append_simulations(
            theta, x
        ).train(max_num_epochs=1)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("method", "percent_nans"),
    (
        (NPE_C, 0.05),
        pytest.param(SNL, 0.05, marks=pytest.mark.xfail),
        pytest.param(SRE, 0.05, marks=pytest.mark.xfail),
    ),
)
def test_inference_with_nan_simulator(method: type, percent_nans: float):
    # likelihood_mean will be likelihood_shift+theta
    num_dim = 3
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    x_o = zeros(1, num_dim)
    num_samples = 500
    num_simulations = 5000

    def linear_gaussian_nan(
        theta, likelihood_shift=likelihood_shift, likelihood_cov=likelihood_cov
    ):
        x = linear_gaussian(theta, likelihood_shift, likelihood_cov)
        # Set nan randomly.
        x[torch.rand(x.shape) < (percent_nans * 1.0 / x.shape[1])] = float("nan")

        return x

    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
    target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
        x_o,
        likelihood_shift=likelihood_shift,
        likelihood_cov=likelihood_cov,
        num_samples=num_samples,
        prior=prior,
    )

    simulator = process_simulator(linear_gaussian_nan, prior, False)
    check_sbi_inputs(simulator, prior)
    inference = method(prior=prior)

    theta, x = simulate_for_sbi(simulator, prior, num_simulations)
    _ = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()

    samples = posterior.sample((num_samples,), x=x_o)

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"{method}")


@pytest.mark.slow
def test_inference_with_restriction_estimator():
    # likelihood_mean will be likelihood_shift+theta
    num_dim = 3
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    x_o = zeros(1, num_dim)
    num_samples = 1000
    num_simulations = 1000

    def linear_gaussian_nan(
        theta, likelihood_shift=likelihood_shift, likelihood_cov=likelihood_cov
    ):
        condition = theta[:, 0] < 0.0
        x = linear_gaussian(theta, likelihood_shift, likelihood_cov)
        x[condition] = float("nan")

        return x

    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
    target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
        x_o,
        likelihood_shift=likelihood_shift,
        likelihood_cov=likelihood_cov,
        num_samples=num_samples,
        prior=prior,
    )

    simulator = process_simulator(linear_gaussian_nan, prior, False)
    check_sbi_inputs(simulator, prior)
    restriction_estimator = RestrictionEstimator(prior=prior)
    proposal = prior
    num_rounds = 2

    for r in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations)
        restriction_estimator.append_simulations(theta, x)
        if r < num_rounds - 1:
            _ = restriction_estimator.train()
        proposal = restriction_estimator.restrict_prior()

    all_theta, all_x, _ = restriction_estimator.get_simulations()

    # Any method can be used in combination with the `RejectionEstimator`.
    inference = NPE_C(prior=prior)
    posterior_estimator = inference.append_simulations(all_theta, all_x).train()

    # Build posterior.
    posterior = DirectPosterior(
        prior=prior, posterior_estimator=posterior_estimator
    ).set_default_x(x_o)

    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"{NPE_C}")


@pytest.mark.parametrize("prior", ("uniform", "gaussian"))
def test_restricted_prior_log_prob(prior):
    """Test whether the log-prob method of the restricted prior works appropriately."""

    def simulator(theta):
        perturbed_theta = theta + 0.5 * torch.randn(2)
        perturbed_theta[theta[:, 0] < 0.8] = torch.as_tensor([
            float("nan"),
            float("nan"),
        ])
        return perturbed_theta

    if prior == "uniform":
        prior = utils.BoxUniform(-2 * torch.ones(2), 2 * torch.ones(2))
    else:
        prior = MultivariateNormal(torch.zeros(2), torch.eye(2))

    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
    theta, x = simulate_for_sbi(simulator, prior, 1000)

    restriction_estimator = RestrictionEstimator(prior=prior)
    restriction_estimator.append_simulations(theta, x)
    _ = restriction_estimator.train(max_num_epochs=40)
    restricted_prior = restriction_estimator.restrict_prior()

    def integrate_grid(distribution):
        resolution = 500
        range_ = 4
        x = torch.linspace(-range_, range_, resolution)
        y = torch.linspace(-range_, range_, resolution)
        X, Y = torch.meshgrid(x, y)
        xy = torch.stack([X, Y])
        xy = torch.reshape(xy, (2, resolution**2)).T
        dist_on_grid = torch.exp(distribution.log_prob(xy))
        integral = torch.sum(dist_on_grid) / resolution**2 * (2 * range_) ** 2
        return integral

    integal_restricted = integrate_grid(restricted_prior)
    error = torch.abs(integal_restricted - torch.as_tensor(1.0))
    assert error < 0.015, "The restricted prior does not integrate to one."

    theta = prior.sample((10_000,))
    restricted_prior_probs = torch.exp(restricted_prior.log_prob(theta))

    valid_thetas = restricted_prior._accept_reject_fn(theta).bool()
    assert torch.all(
        restricted_prior_probs[valid_thetas] > 0.0
    ), "Accepted theta have zero probability."
    assert torch.all(
        restricted_prior_probs[torch.logical_not(valid_thetas)] == 0.0
    ), "Rejected theta has non-zero probablity."


@pytest.mark.parametrize(
    "num_simulations, simulation_batch_size, num_workers, use_process_simulator",
    [
        (0, None, 1, True),
        (10, None, 1, True),
        (100, 10, 1, True),
        (100, None, 2, True),
        (1000, 50, 4, True),
        (100, 10, 2, False),
    ],
)
def test_simulate_for_sbi(
    num_simulations, simulation_batch_size, num_workers, use_process_simulator
):
    """Test the simulate_for_sbi function with various configurations."""
    num_dim = 3
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior = BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    if use_process_simulator:
        simulator = process_simulator(simulator, prior, False)

    theta, x = simulate_for_sbi(
        simulator=simulator,
        proposal=prior,
        num_simulations=num_simulations,
        simulation_batch_size=simulation_batch_size,
        num_workers=num_workers,
    )

    if num_simulations == 0:
        assert (
            theta.numel() == 0
        ), "Theta should be an empty tensor when num_simulations=0"
        assert x.numel() == 0, "x should be an empty tensor when num_simulations=0"
    else:
        assert (
            theta.shape[0] == num_simulations
        ), "Theta should have num_simulations rows"
        assert x.shape[0] == num_simulations, "x should have num_simulations rows"
        assert theta.shape[1] == num_dim, "Theta should have num_dim columns"
        assert x.shape[1] == num_dim, "x should have num_dim columns"

        assert torch.all(torch.isfinite(theta)), "Theta contains non-finite values"
        assert torch.all(torch.isfinite(x)), "x contains non-finite values"

        assert torch.all(
            theta >= prior.base_dist.low
        ), "Theta contains values below the prior lower bound"
        assert torch.all(
            theta <= prior.base_dist.high
        ), "Theta contains values above the prior upper bound"

    if not use_process_simulator and num_workers > 1:
        assert (
            theta.shape[0] == num_simulations
        ), "Simulation should work even without process_simulator"
