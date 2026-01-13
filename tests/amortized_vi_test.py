# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""
Tests for AmortizedVIPosterior.

These tests validate the amortized variational inference implementation where
a conditional flow q(θ|x) is trained by optimizing ELBO against a potential
function from NLE.
"""

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import NLE, AmortizedVIPosterior
from sbi.inference.potentials.likelihood_based_potential import (
    likelihood_estimator_based_potential,
)
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.metrics import c2st


@pytest.fixture
def linear_gaussian_setup():
    """Setup for linear Gaussian test problem with trained NLE."""
    torch.manual_seed(42)

    num_dim = 2
    num_simulations = 5000
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.25 * eye(num_dim)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    # Generate simulation data
    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    # Train NLE (use NSF for better gradient properties)
    trainer = NLE(prior=prior, show_progress_bars=False, density_estimator="nsf")
    trainer.append_simulations(theta, x)
    likelihood_estimator = trainer.train(max_num_epochs=200)

    # Create potential function
    potential_fn, _ = likelihood_estimator_based_potential(
        likelihood_estimator=likelihood_estimator,
        prior=prior,
        x_o=None,
    )

    return {
        "prior": prior,
        "potential_fn": potential_fn,
        "theta": theta,
        "x": x,
        "likelihood_shift": likelihood_shift,
        "likelihood_cov": likelihood_cov,
        "num_dim": num_dim,
    }


@pytest.mark.slow
def test_amortized_vi_posterior_training(linear_gaussian_setup):
    """Test that AmortizedVIPosterior trains successfully."""
    setup = linear_gaussian_setup

    posterior = AmortizedVIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
        q="nsf",
        num_transforms=2,
        hidden_features=32,
    )

    # Train
    posterior.train(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
    )

    # Should be marked as trained
    assert posterior._trained


@pytest.mark.slow
def test_amortized_vi_posterior_accuracy(linear_gaussian_setup):
    """Test that AmortizedVIPosterior produces accurate posteriors."""
    setup = linear_gaussian_setup

    posterior = AmortizedVIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
        q="nsf",
        num_transforms=2,
        hidden_features=32,
    )

    posterior.train(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
    )

    # Test on multiple observations
    test_x_os = [
        zeros(1, setup["num_dim"]),
        ones(1, setup["num_dim"]),
        -ones(1, setup["num_dim"]),
    ]

    for x_o in test_x_os:
        # Get ground truth
        true_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o.squeeze(0),
            setup["likelihood_shift"],
            setup["likelihood_cov"],
            zeros(setup["num_dim"]),
            eye(setup["num_dim"]),
        )
        true_samples = true_posterior.sample((1000,))

        # Sample from amortized posterior
        vi_samples = posterior.sample((1000,), x=x_o)

        # Check accuracy with C2ST
        c2st_score = c2st(true_samples, vi_samples).item()
        assert c2st_score < 0.6, (
            f"C2ST too high for x_o={x_o.squeeze().tolist()}: {c2st_score:.3f}"
        )


@pytest.mark.slow
def test_amortized_vi_posterior_batched_sampling(linear_gaussian_setup):
    """Test batched sampling from AmortizedVIPosterior."""
    setup = linear_gaussian_setup

    posterior = AmortizedVIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
        q="nsf",
        num_transforms=2,
        hidden_features=32,
    )

    posterior.train(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
    )

    # Test batched sampling
    num_obs = 10
    num_samples = 100

    x_batch = torch.randn(num_obs, setup["num_dim"])
    samples = posterior.sample_batched((num_samples,), x=x_batch)

    # Check shape
    assert samples.shape == (num_samples, num_obs, setup["num_dim"])


@pytest.mark.slow
def test_amortized_vi_posterior_log_prob(linear_gaussian_setup):
    """Test log_prob evaluation."""
    setup = linear_gaussian_setup

    posterior = AmortizedVIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
        q="nsf",
        num_transforms=2,
        hidden_features=32,
    )

    posterior.train(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
    )

    # Evaluate log prob
    x_o = zeros(1, setup["num_dim"])
    theta_test = torch.randn(10, setup["num_dim"])

    log_probs = posterior.log_prob(theta_test, x=x_o)

    # Check shape
    assert log_probs.shape == (10,)

    # Check that log probs are finite
    assert torch.isfinite(log_probs).all()


@pytest.mark.slow
def test_amortized_vi_requires_x_for_sampling(linear_gaussian_setup):
    """Test that sampling without x raises an error."""
    setup = linear_gaussian_setup

    posterior = AmortizedVIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
        q="nsf",
    )

    posterior.train(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=100,
        show_progress_bar=False,
    )

    # Should raise ValueError when x is not provided
    with pytest.raises(ValueError, match="requires observation x"):
        posterior.sample((100,))


@pytest.mark.slow
def test_amortized_vi_requires_training_before_sampling(linear_gaussian_setup):
    """Test that sampling before training raises an error."""
    setup = linear_gaussian_setup

    posterior = AmortizedVIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
        q="nsf",
    )

    # Should raise RuntimeError when not trained
    with pytest.raises(RuntimeError, match="must be trained"):
        posterior.sample((100,), x=zeros(1, setup["num_dim"]))


@pytest.mark.slow
def test_amortized_vi_vs_single_x_vi(linear_gaussian_setup):
    """Compare AmortizedVIPosterior against standard VIPosterior."""
    from sbi.inference.posteriors.vi_posterior import VIPosterior

    setup = linear_gaussian_setup

    # Train amortized VI
    amortized_posterior = AmortizedVIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
        q="nsf",
        num_transforms=2,
        hidden_features=32,
    )

    amortized_posterior.train(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
    )

    # Train single-x VI for a specific observation
    x_o = torch.tensor([[0.5, 0.5]])

    potential_fn_single, _ = likelihood_estimator_based_potential(
        likelihood_estimator=setup["potential_fn"].likelihood_estimator,
        prior=setup["prior"],
        x_o=x_o,
    )

    single_x_posterior = VIPosterior(
        potential_fn=potential_fn_single,
        prior=setup["prior"],
        q="maf",
    )
    single_x_posterior.set_default_x(x_o)
    single_x_posterior.train(max_num_iters=500, show_progress_bar=False)

    # Get ground truth
    true_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o.squeeze(0),
        setup["likelihood_shift"],
        setup["likelihood_cov"],
        zeros(setup["num_dim"]),
        eye(setup["num_dim"]),
    )
    true_samples = true_posterior.sample((1000,))

    # Compare
    amortized_samples = amortized_posterior.sample((1000,), x=x_o)
    single_x_samples = single_x_posterior.sample((1000,))

    c2st_amortized = c2st(true_samples, amortized_samples).item()
    _ = c2st(true_samples, single_x_samples).item()  # unused, kept for comparison

    # Amortized should be reasonably accurate
    assert c2st_amortized < 0.6, f"Amortized VI C2ST too high: {c2st_amortized:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
