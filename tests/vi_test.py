# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""
Tests for Variational Inference (VI) using VIPosterior.

This module tests both:
- Single-x VI mode: train() - trains q(θ) for a specific observation x_o
- Amortized VI mode: train_amortized() - trains q(θ|x) across observations
"""

from __future__ import annotations

import tempfile
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.distributions.transforms as torch_tf
from torch import eye, ones, zeros
from torch.distributions import Beta, Binomial, Gamma, MultivariateNormal

from sbi.inference import NLE, likelihood_estimator_based_potential
from sbi.inference.posteriors import VIPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.factory import ZukoFlowType
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils import MultipleIndependent
from sbi.utils.metrics import c2st, check_c2st

# Supported variational families for VI
FLOWS = ["maf", "nsf", "naf", "unaf", "nice", "sospf", "gaussian", "gaussian_diag"]


# =============================================================================
# Shared Test Utilities
# =============================================================================


class FakePotential(BasePotential):
    """A potential that returns the prior log probability.

    This makes the posterior equal to the prior, which is a trivial but
    well-defined posterior that allows proper testing of VI machinery.
    """

    def __call__(self, theta, **kwargs):
        return self.prior.log_prob(theta)

    def allow_iid_x(self) -> bool:
        return True


def make_tractable_potential(target_distribution, prior):
    """Create a potential function from a known target distribution."""

    class TractablePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return target_distribution.log_prob(
                torch.as_tensor(theta, dtype=torch.float32)
            )

        def allow_iid_x(self) -> bool:
            return True

    return TractablePotential(prior=prior)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def linear_gaussian_setup():
    """Setup for linear Gaussian test problem with trained NLE.

    Returns a dict with:
    - prior: MultivariateNormal prior
    - potential_fn: Trained NLE-based potential function
    - theta, x: Simulation data
    - likelihood_shift, likelihood_cov: Likelihood parameters
    - num_dim: Dimensionality
    """
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

    # Train NLE
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


# =============================================================================
# Single-x VI Tests: train() method
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("vi_method", ("rKL", "fKL", "IW", "alpha"))
@pytest.mark.parametrize("sampling_method", ("naive", "sir"))
def test_c2st_vi_on_Gaussian(num_dim: int, vi_method: str, sampling_method: str):
    """Test single-x VI on Gaussian, comparing to ground truth via c2st."""
    if sampling_method == "naive" and vi_method == "IW":
        return  # This combination is not meant to perform well

    num_samples = 2000
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)

    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    prior = MultivariateNormal(prior_mean, prior_cov)
    potential_fn = make_tractable_potential(target_distribution, prior)
    theta_transform = torch_tf.identity_transform

    # Use 'gaussian' for 1D (normalizing flows are unstable in 1D with Zuko)
    q = "gaussian" if num_dim == 1 else "nsf"
    posterior = VIPosterior(potential_fn, prior, theta_transform=theta_transform, q=q)
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))
    posterior.vi_method = vi_method
    posterior.train()
    samples = posterior.sample((num_samples,), method=sampling_method)
    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("q", FLOWS)
def test_c2st_vi_flows_on_Gaussian(num_dim: int, q: str):
    """Test different flow types on Gaussian via c2st."""
    # Normalizing flows (except gaussian families) are unstable in 1D with Zuko
    if num_dim == 1 and q not in ["gaussian", "gaussian_diag"]:
        return

    num_samples = 2000
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)

    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    prior = MultivariateNormal(prior_mean, prior_cov)
    potential_fn = make_tractable_potential(target_distribution, prior)
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(potential_fn, prior, theta_transform=theta_transform, q=q)
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))
    posterior.train(n_particles=1000, eps=1e-8)
    samples = posterior.sample((num_samples,))
    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
def test_c2st_vi_external_distributions_on_Gaussian(num_dim: int):
    """Test VI with user-provided external distribution."""
    num_samples = 2000
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)

    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    prior = MultivariateNormal(prior_mean, prior_cov)
    potential_fn = make_tractable_potential(target_distribution, prior)
    theta_transform = torch_tf.identity_transform

    mu = zeros(num_dim, requires_grad=True)
    scale = ones(num_dim, requires_grad=True)
    q = torch.distributions.Independent(torch.distributions.Normal(mu, scale), 1)
    posterior = VIPosterior(
        potential_fn,
        prior,
        theta_transform=theta_transform,
        q=q,
        vi_method="rKL",
        parameters=[mu, scale],
    )
    posterior.set_default_x(x_o)
    posterior.train(check_for_convergence=False)
    samples = posterior.sample((num_samples,))
    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.parametrize("q", FLOWS)
def test_deepcopy_support(q: str):
    """Test that VIPosterior supports deepcopy for all flow types."""
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    potential_fn = FakePotential(prior=prior)
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(potential_fn, prior, theta_transform=theta_transform, q=q)
    posterior_copy = deepcopy(posterior)
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))
    assert posterior._x != posterior_copy._x, "Deepcopy should create independent copy"

    posterior_copy = deepcopy(posterior)
    assert (posterior._x == posterior_copy._x).all(), "Deepcopy should preserve values"

    # Verify samples are reproducible
    torch.manual_seed(0)
    if hasattr(posterior._q, "rsample"):
        s1 = posterior._q.rsample()
    else:
        s1 = posterior._q.sample((1,)).squeeze(0)
    torch.manual_seed(0)
    if hasattr(posterior_copy._q, "rsample"):
        s2 = posterior_copy._q.rsample()
    else:
        s2 = posterior_copy._q.sample((1,)).squeeze(0)
    assert torch.allclose(s1, s2), "Samples should match after deepcopy"

    # Test deepcopy after sampling (can produce nonleaf tensors in cache)
    if hasattr(posterior.q, "rsample"):
        posterior.q.rsample()
    else:
        posterior.q.sample((1,))
    deepcopy(posterior)  # Should not raise


@pytest.mark.parametrize("q", FLOWS)
def test_pickle_support(q: str):
    """Test that VIPosterior can be saved and loaded via pickle."""
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    potential_fn = FakePotential(prior=prior)
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(potential_fn, prior, theta_transform=theta_transform, q=q)
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(posterior, f.name)
        posterior_loaded = torch.load(f.name, weights_only=False)
        assert (posterior._x == posterior_loaded._x).all()

    # Verify samples are reproducible
    torch.manual_seed(0)
    if hasattr(posterior._q, "rsample"):
        s1 = posterior._q.rsample()
    else:
        s1 = posterior._q.sample((1,)).squeeze(0)
    torch.manual_seed(0)
    if hasattr(posterior_loaded._q, "rsample"):
        s2 = posterior_loaded._q.rsample()
    else:
        s2 = posterior_loaded._q.sample((1,)).squeeze(0)

    assert torch.allclose(s1, s2), "Samples should match after unpickling"


def test_vi_posterior_interface():
    """Test VIPosterior interface: hyperparameters, training, evaluation."""
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    potential_fn = FakePotential(prior=prior)
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(potential_fn, theta_transform=theta_transform)
    posterior.set_default_x(torch.zeros((1, num_dim)))

    posterior2 = VIPosterior(potential_fn)

    # Check support compatibility (if available)
    if hasattr(posterior.q, "support") and hasattr(posterior2.q, "support"):
        assert isinstance(posterior.q.support, type(posterior2.q.support))

    # Should raise if not trained
    with pytest.raises(Exception) as execinfo:
        posterior.sample()
    assert "The variational posterior was not fit" in execinfo.value.args[0]

    with pytest.raises(Exception) as execinfo:
        posterior.log_prob(prior.sample())
    assert "The variational posterior was not fit" in execinfo.value.args[0]

    # Test training hyperparameters
    posterior.train(max_num_iters=20)

    posterior.train(max_num_iters=20, optimizer=torch.optim.SGD)
    assert isinstance(posterior._optimizer._optimizer, torch.optim.SGD)

    posterior.train(max_num_iters=20, stick_the_landing=True)
    assert posterior._optimizer.stick_the_landing

    posterior.vi_method = "alpha"
    posterior.train(max_num_iters=20)
    posterior.train(max_num_iters=20, alpha=0.9)
    assert posterior._optimizer.alpha == 0.9

    posterior.vi_method = "IW"
    posterior.train(max_num_iters=20)
    posterior.train(max_num_iters=20, K=32)
    assert posterior._optimizer.K == 32

    # Test sampling hyperparameters
    posterior.sample()
    posterior.sample(method="sir")
    posterior.sample(method="sir", K=128)

    # Test evaluation
    posterior.evaluate()
    posterior.evaluate("prop")
    posterior.evaluate("prop_prior")

    # Test log_prob and potential
    posterior.log_prob(posterior.sample())
    posterior.potential(posterior.sample())


def test_vi_with_multiple_independent_prior():
    """Test VI with MultipleIndependent prior (mixed distributions)."""
    prior = MultipleIndependent(
        [
            Gamma(torch.tensor([1.0]), torch.tensor([0.5])),
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),
        ],
        validate_args=False,
    )

    def simulator(theta):
        return Binomial(probs=theta[:, 1]).sample().reshape(-1, 1)

    num_simulations = 100
    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    trainer = NLE(prior)
    nle = trainer.append_simulations(theta, x).train()
    potential, transform = likelihood_estimator_based_potential(nle, prior, x[0])
    posterior = VIPosterior(
        potential,
        prior=prior,
        theta_transform=transform,  # type: ignore
    )
    posterior.set_default_x(x[0])
    posterior.train()

    posterior.sample(sample_shape=(10,), show_progress_bars=False)


@pytest.mark.parametrize("num_dim", (1, 2, 3, 4, 5, 10, 25, 33))
@pytest.mark.parametrize("q_type", FLOWS)
def test_vi_flow_builders(num_dim: int, q_type: str):
    """Test variational families are built correctly with sampling and log_prob."""
    # Normalizing flows (except gaussian families) need >= 2 dimensions for Zuko
    if num_dim == 1 and q_type not in ("gaussian", "gaussian_diag"):
        return

    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    potential_fn = FakePotential(prior=prior)
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(
        potential_fn, prior, theta_transform=theta_transform, q=q_type
    )

    q = posterior.q

    # Test sampling without sample_shape
    sample = q.sample(())
    assert sample.shape == (num_dim,), f"Shape mismatch: {sample.shape}"
    log_prob = q.log_prob(sample.unsqueeze(0))
    assert log_prob.shape == (1,), f"Log_prob shape mismatch: {log_prob.shape}"

    # Test sampling with sample_shape
    sample_batch = q.sample((10,))
    expected_shape = (10, num_dim)
    assert sample_batch.shape == expected_shape, f"Shape mismatch: {sample_batch.shape}"
    log_prob_batch = q.log_prob(sample_batch)
    assert log_prob_batch.shape == (10,), f"Shape mismatch: {log_prob_batch.shape}"


# =============================================================================
# Amortized VI Tests: train_amortized() method
# =============================================================================


@pytest.mark.slow
def test_amortized_vi_training(linear_gaussian_setup):
    """Test that VIPosterior.train_amortized() trains successfully."""
    setup = linear_gaussian_setup

    posterior = VIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
    )

    posterior.train_amortized(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
        flow_type=ZukoFlowType.NSF,
        num_transforms=2,
        hidden_features=32,
    )

    assert posterior._mode == "amortized"


@pytest.mark.slow
def test_amortized_vi_accuracy(linear_gaussian_setup):
    """Test that amortized VI produces accurate posteriors across observations."""
    setup = linear_gaussian_setup

    posterior = VIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
    )

    posterior.train_amortized(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
        flow_type=ZukoFlowType.NSF,
        num_transforms=2,
        hidden_features=32,
    )

    # Test on multiple observations
    test_x_os = [
        zeros(1, setup["num_dim"]),
        ones(1, setup["num_dim"]),
        -ones(1, setup["num_dim"]),
    ]

    for x_o in test_x_os:
        true_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o.squeeze(0),
            setup["likelihood_shift"],
            setup["likelihood_cov"],
            zeros(setup["num_dim"]),
            eye(setup["num_dim"]),
        )
        true_samples = true_posterior.sample((1000,))
        vi_samples = posterior.sample((1000,), x=x_o)

        c2st_score = c2st(true_samples, vi_samples).item()
        assert c2st_score < 0.6, f"C2ST too high for x_o={x_o.squeeze().tolist()}"


@pytest.mark.slow
def test_amortized_vi_batched_sampling(linear_gaussian_setup):
    """Test batched sampling from amortized VIPosterior."""
    setup = linear_gaussian_setup

    posterior = VIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
    )

    posterior.train_amortized(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
        flow_type=ZukoFlowType.NSF,
        num_transforms=2,
        hidden_features=32,
    )

    num_obs = 10
    num_samples = 100
    x_batch = torch.randn(num_obs, setup["num_dim"])
    samples = posterior.sample_batched((num_samples,), x=x_batch)

    assert samples.shape == (num_samples, num_obs, setup["num_dim"])


@pytest.mark.slow
def test_amortized_vi_log_prob(linear_gaussian_setup):
    """Test log_prob evaluation in amortized mode."""
    setup = linear_gaussian_setup

    posterior = VIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
    )

    posterior.train_amortized(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
        flow_type=ZukoFlowType.NSF,
        num_transforms=2,
        hidden_features=32,
    )

    x_o = zeros(1, setup["num_dim"])
    theta_test = torch.randn(10, setup["num_dim"])

    log_probs = posterior.log_prob(theta_test, x=x_o)

    assert log_probs.shape == (10,)
    assert torch.isfinite(log_probs).all()


@pytest.mark.slow
def test_amortized_vi_default_x(linear_gaussian_setup):
    """Test that amortized mode uses default_x when x not provided."""
    setup = linear_gaussian_setup

    posterior = VIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
    )

    posterior.train_amortized(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=100,
        show_progress_bar=False,
        flow_type=ZukoFlowType.NSF,
    )

    posterior.set_default_x(zeros(1, setup["num_dim"]))
    samples = posterior.sample((100,))
    assert samples.shape == (100, setup["num_dim"])


@pytest.mark.slow
def test_amortized_vi_requires_training(linear_gaussian_setup):
    """Test that sampling before training raises an error."""
    setup = linear_gaussian_setup

    posterior = VIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
    )

    posterior.set_default_x(zeros(1, setup["num_dim"]))
    with pytest.raises(ValueError):
        posterior.sample((100,))


@pytest.mark.slow
def test_amortized_vs_single_x_vi(linear_gaussian_setup):
    """Compare amortized VI against single-x VI for the same observation."""
    setup = linear_gaussian_setup

    # Train amortized VI
    amortized_posterior = VIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
    )
    amortized_posterior.train_amortized(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
        flow_type=ZukoFlowType.NSF,
        num_transforms=2,
        hidden_features=32,
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

    assert c2st_amortized < 0.6, f"Amortized VI C2ST too high: {c2st_amortized:.3f}"
    assert abs(c2st_amortized - c2st_single_x) < 0.15, (
        f"Amortized ({c2st_amortized:.3f}) much worse than "
        f"single-x ({c2st_single_x:.3f})"
    )


@pytest.mark.slow
def test_amortized_vi_gradient_flow(linear_gaussian_setup):
    """Verify gradients flow through ELBO to flow parameters."""
    from torch.optim import Adam

    setup = linear_gaussian_setup

    posterior = VIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
    )

    # Build the flow manually
    theta = setup["theta"][:500].to("cpu")
    x = setup["x"][:500].to("cpu")
    posterior._amortized_q = posterior._build_conditional_flow(
        theta[:100],
        x[:100],
        flow_type=ZukoFlowType.NSF,
        num_transforms=2,
        hidden_features=32,
    )
    posterior._amortized_q.to("cpu")
    posterior._mode = "amortized"

    initial_params = [p.clone() for p in posterior._amortized_q.parameters()]

    # Compute ELBO loss and backprop
    optimizer = Adam(posterior._amortized_q.parameters(), lr=1e-3)
    optimizer.zero_grad()

    x_batch = x[:8]
    loss = posterior._compute_amortized_elbo_loss(x_batch, n_particles=16)
    loss.backward()

    # Verify gradients exist and are non-zero
    for i, p in enumerate(posterior._amortized_q.parameters()):
        assert p.grad is not None, f"Gradient is None for parameter {i}"
        assert torch.isfinite(p.grad).all(), f"Gradient has NaN/Inf for param {i}"
        assert p.grad.abs().max() > 1e-10, f"Gradient is zero for parameter {i}"

    # Verify parameters change after optimization step
    optimizer.step()
    changed_count = sum(
        1
        for p_init, p_new in zip(
            initial_params, posterior._amortized_q.parameters(), strict=True
        )
        if not torch.allclose(p_init, p_new.detach(), atol=1e-8)
    )
    assert changed_count > 0, "No parameters changed after optimization step"


@pytest.mark.slow
def test_amortized_vi_map(linear_gaussian_setup):
    """Test that MAP estimation returns high-density region."""
    setup = linear_gaussian_setup

    posterior = VIPosterior(
        potential_fn=setup["potential_fn"],
        prior=setup["prior"],
    )

    posterior.train_amortized(
        theta=setup["theta"],
        x=setup["x"],
        max_num_iters=500,
        show_progress_bar=False,
        flow_type=ZukoFlowType.NSF,
        num_transforms=2,
        hidden_features=32,
    )

    x_o = zeros(1, setup["num_dim"])
    posterior.set_default_x(x_o)
    map_estimate = posterior.map(num_iter=500, num_to_optimize=50)

    # For linear Gaussian, MAP equals posterior mean
    true_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o.squeeze(0),
        setup["likelihood_shift"],
        setup["likelihood_cov"],
        zeros(setup["num_dim"]),
        eye(setup["num_dim"]),
    )
    true_mean = true_posterior.mean
    map_estimate_flat = map_estimate.squeeze(0)

    assert torch.allclose(map_estimate_flat, true_mean, atol=0.3), (
        f"MAP {map_estimate_flat.tolist()} not close to true mean {true_mean.tolist()}"
    )

    # MAP should have higher potential than random samples
    map_log_prob = posterior.potential(map_estimate)
    random_samples = posterior.sample((100,), x=x_o)
    random_log_probs = posterior.potential(random_samples)

    assert map_log_prob > random_log_probs.median(), (
        f"MAP log_prob {map_log_prob.item():.3f} not better than "
        f"median random {random_log_probs.median().item():.3f}"
    )
