# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import logging
import warnings

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.diagnostics import run_sbc
from sbi.inference import (
    MNPE,
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
from sbi.utils import RestrictionEstimator
from sbi.utils.metrics import check_c2st
from sbi.utils.sbiutils import handle_invalid_x
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)


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
    # Create data with extreme outlier.
    num_dim = 2
    theta = torch.ones(100, num_dim)
    x = torch.rand(100, num_dim)
    x[0, 0] = 1e7  # Single extreme outlier

    # Make sure a warning is raised because of extreme outliers that may cause
    # precision loss during z-scoring.
    with pytest.warns(UserWarning, match="extreme outliers"):
        snpe_method(utils.BoxUniform(zeros(num_dim), ones(num_dim))).append_simulations(
            theta, x
        ).train(max_num_epochs=1)


def _round_one_data_with_nan(trainer_class: type = NPE_C, num_dim: int = 2):
    """Build a trainer plus round-one data containing a single NaN row.

    Passing a proposal that is not the prior makes `append_simulations()` treat the very
    first batch as round one, which is where the strict atomic check applies. Nothing is
    trained, so these tests are fast.
    """
    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
    theta = prior.sample((20,))
    x = theta + 0.1 * torch.randn(20, num_dim)
    x[0, 0] = float("nan")
    proposal = utils.BoxUniform(-1.0 * ones(num_dim), 1.0 * ones(num_dim))

    return trainer_class(prior=prior, show_progress_bars=False), theta, x, proposal


def test_multiround_npe_c_raises_on_invalid_x():
    """Multi-round atomic NPE-C must reject invalid simulations by default.

    `exclude_invalid_x` defaults to False for rounds > 0, and discarding invalid
    simulations biases the atomic NPE-C loss, so the default must raise rather than
    train on NaNs. This check was dead between v0.23.0 and v0.27.0 because it compared
    the class name against the pre-rename `SNPE_C`.
    """
    inference, theta, x, proposal = _round_one_data_with_nan()

    with pytest.raises(ValueError, match="does not allow invalid simulations"):
        inference.append_simulations(theta, x, proposal=proposal)


def test_multiround_npe_c_warns_when_excluding_invalid_x(caplog):
    """With an explicit `exclude_invalid_x=True`, NPE-C warns instead of raising."""
    inference, theta, x, proposal = _round_one_data_with_nan()

    with caplog.at_level(logging.WARNING), warnings.catch_warnings():
        # The proposal is not a NeuralPosterior, which warns separately.
        warnings.simplefilter("ignore", UserWarning)
        inference.append_simulations(
            theta, x, proposal=proposal, exclude_invalid_x=True
        )

    assert "Multiround NPE-C (atomic)" in caplog.text
    assert "systematically wrong results" in caplog.text
    # The single invalid row was discarded.
    assert inference.get_simulations()[0].shape[0] == theta.shape[0] - 1


def test_multiround_mnpe_raises_on_invalid_x():
    """MNPE inherits the strict check, because it is always atomic.

    MNPE subclasses NPE_C, and its `MixedDensityEstimator` is never a
    `MixtureDensityEstimator`, so `use_non_atomic_loss` can never become True. Every
    multi-round MNPE run therefore uses the atomic loss and has the same exposure to
    invalid simulations as NPE-C.
    """
    inference, theta, x, proposal = _round_one_data_with_nan(trainer_class=MNPE)

    with pytest.raises(ValueError, match="does not allow invalid simulations"):
        inference.append_simulations(theta, x, proposal=proposal)


@pytest.mark.parametrize("trainer_class", (NPE_C, MNPE))
def test_single_round_tolerates_invalid_x(trainer_class, caplog):
    """Round 0 uses the plain log-prob loss, so discarding invalid rows is safe.

    Filtering the joint on the validity of x leaves p(theta|x) unchanged for valid x, so
    the strict check must not apply without a proposal.
    """
    prior = utils.BoxUniform(-2.0 * ones(2), 2.0 * ones(2))
    theta = prior.sample((20,))
    x = theta + 0.1 * torch.randn(20, 2)
    x[0, 0] = float("nan")

    inference = trainer_class(prior=prior, show_progress_bars=False)
    with caplog.at_level(logging.WARNING):
        inference.append_simulations(theta, x)  # must not raise

    assert "Found 1 NaN simulations" in caplog.text
    assert "does not allow invalid simulations" not in caplog.text


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

    # run sbc
    num_sbc_samples = 100
    thetas = prior.sample((num_sbc_samples,))
    xs = simulator(thetas)
    ranks, daps = run_sbc(thetas, xs, posterior, num_posterior_samples=1000)
    assert torch.isfinite(ranks).all()


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

    # test passing restricted prior to inference and using process_prior, see #790.
    restricted_prior = restriction_estimator.restrict_prior()
    prior = process_prior(restricted_prior)[0]

    # Any method can be used in combination with the `RejectionEstimator`.
    inference = NPE_C(prior=prior)
    posterior_estimator = inference.append_simulations(all_theta, all_x).train()

    # Build posterior.
    posterior = DirectPosterior(
        prior=prior,
        posterior_estimator=posterior_estimator,
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

    # test restricted prior log_prob
    restricted_prior, _, _ = process_prior(restricted_prior)

    def integrate_grid(distribution):
        resolution = 500
        range_ = 4
        x = torch.linspace(-range_, range_, resolution)
        y = torch.linspace(-range_, range_, resolution)
        X, Y = torch.meshgrid(x, y, indexing="ij")
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
    assert torch.all(restricted_prior_probs[valid_thetas] > 0.0), (
        "Accepted theta have zero probability."
    )
    assert torch.all(restricted_prior_probs[torch.logical_not(valid_thetas)] == 0.0), (
        "Rejected theta has non-zero probablity."
    )
