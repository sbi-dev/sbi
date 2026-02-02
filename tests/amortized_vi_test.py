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
from sbi.inference.posteriors import ZukoFlowType
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
        flow_type=ZukoFlowType.NSF,
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
        flow_type=ZukoFlowType.NSF,
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
        flow_type=ZukoFlowType.NSF,
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
        flow_type=ZukoFlowType.NSF,
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
        flow_type=ZukoFlowType.NSF,
    )

    posterior.train(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=100,
        show_progress_bar=False,
    )

    posterior.set_default_x(zeros(1, setup["num_dim"]))
    samples = posterior.sample((100,))
    assert samples.shape == (100, setup["num_dim"])


@pytest.mark.slow
def test_amortized_vi_requires_training_before_sampling(linear_gaussian_setup):
    """Test that sampling before training raises an error."""
    setup = linear_gaussian_setup

    posterior = AmortizedVIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
        flow_type=ZukoFlowType.NSF,
    )

    posterior.set_default_x(zeros(1, setup["num_dim"]))
    # Should raise RuntimeError when not trained
    with pytest.raises(RuntimeError, match="must be trained"):
        posterior.sample((100,))


@pytest.mark.slow
def test_amortized_vi_vs_single_x_vi(linear_gaussian_setup):
    """Compare AmortizedVIPosterior against standard VIPosterior."""
    from sbi.inference.posteriors.vi_posterior import VIPosterior

    setup = linear_gaussian_setup

    # Train amortized VI
    amortized_posterior = AmortizedVIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
        flow_type=ZukoFlowType.NSF,
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
    c2st_single_x = c2st(true_samples, single_x_samples).item()

    # Amortized should be reasonably accurate
    assert c2st_amortized < 0.6, f"Amortized VI C2ST too high: {c2st_amortized:.3f}"

    # Amortized should not be dramatically worse than single-x
    assert abs(c2st_amortized - c2st_single_x) < 0.15, (
        f"Amortized VI ({c2st_amortized:.3f}) much worse than "
        f"single-x VI ({c2st_single_x:.3f})"
    )


@pytest.mark.slow
def test_gradient_flow_through_elbo(linear_gaussian_setup):
    """Verify gradients flow through ELBO to flow parameters."""
    from torch.optim import Adam

    setup = linear_gaussian_setup

    posterior = AmortizedVIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
        flow_type=ZukoFlowType.NSF,
        num_transforms=2,
        hidden_features=32,
    )

    # Build the flow manually (normally done in train())
    theta = setup["theta"][:500].to("cpu")
    x = setup["x"][:500].to("cpu")
    posterior._variational_distribution = posterior._build_variational_distribution(
        theta[:100], x[:100]
    )
    posterior._variational_distribution.to("cpu")

    # Store initial parameters
    initial_params = [
        p.clone() for p in posterior._variational_distribution.parameters()
    ]

    # Compute ELBO loss
    optimizer = Adam(posterior._variational_distribution.parameters(), lr=1e-3)
    optimizer.zero_grad()

    # Use small batch for quick test
    x_batch = x[:8]
    loss = posterior._compute_elbo_loss(x_batch, n_particles=16)

    # Backward pass
    loss.backward()

    # Verify gradients exist and are non-zero for all parameters
    for i, p in enumerate(posterior._variational_distribution.parameters()):
        assert p.grad is not None, f"Gradient is None for parameter {i}"
        assert torch.isfinite(p.grad).all(), f"Gradient has NaN/Inf for parameter {i}"
        # At least some gradients should be non-trivial
        assert p.grad.abs().max() > 1e-10, f"Gradient is zero for parameter {i}"

    # Take optimization step
    optimizer.step()

    # Verify parameters actually changed
    changed_count = 0
    params = posterior._variational_distribution.parameters()
    for p_init, p_new in zip(initial_params, params, strict=True):
        if not torch.allclose(p_init, p_new.detach(), atol=1e-8):
            changed_count += 1

    assert changed_count > 0, "No parameters changed after optimization step"


@pytest.mark.slow
def test_amortized_vi_map(linear_gaussian_setup):
    """Test that MAP estimation returns high-density region."""
    setup = linear_gaussian_setup

    posterior = AmortizedVIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
        flow_type=ZukoFlowType.NSF,
        num_transforms=2,
        hidden_features=32,
    )

    posterior.train(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
    )

    # Test MAP for a specific observation
    x_o = zeros(1, setup["num_dim"])

    # Get MAP estimate
    posterior.set_default_x(x_o)
    map_estimate = posterior.map(num_iter=500, num_to_optimize=50)

    # For linear Gaussian, the MAP equals the posterior mean
    true_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o.squeeze(0),
        setup["likelihood_shift"],
        setup["likelihood_cov"],
        zeros(setup["num_dim"]),
        eye(setup["num_dim"]),
    )
    true_mean = true_posterior.mean

    map_estimate_flat = map_estimate.squeeze(0)

    # MAP should be close to true posterior mean (allowing some tolerance)
    assert torch.allclose(map_estimate_flat, true_mean, atol=0.3), (
        f"MAP {map_estimate_flat.tolist()} not close to true mean {true_mean.tolist()}"
    )

    # MAP should have higher potential than random samples
    posterior.set_default_x(x_o)
    map_log_prob = posterior.potential(map_estimate)
    random_samples = posterior.sample((100,), x=x_o)
    random_log_probs = posterior.potential(random_samples)

    assert map_log_prob > random_log_probs.median(), (
        f"MAP log_prob {map_log_prob.item():.3f} not better than "
        f"median random {random_log_probs.median().item():.3f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
