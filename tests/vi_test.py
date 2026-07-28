# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""
Tests for Variational Inference (VI) using VIPosterior.

This module tests both:
- Single-x VI mode: train() - trains q(θ) for a specific observation x_o
- Amortized VI mode: train_amortized() - trains q(θ|x) across observations
"""

from __future__ import annotations

import pickle
import tempfile
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.distributions.transforms as torch_tf
from torch import eye, ones, zeros
from torch.distributions import Beta, Binomial, Gamma, MultivariateNormal

from sbi.inference import NLE, NRE, likelihood_estimator_based_potential
from sbi.inference.posteriors import VIPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference.potentials.ratio_based_potential import (
    ratio_estimator_based_potential,
)
from sbi.neural_nets.factory import ZukoFlowType
from sbi.samplers.vi.vi_utils import AdaptedVariationalDistribution, LearnableGaussian
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


def _build_linear_gaussian_setup(trainer_type: str = "nle"):
    """Helper to build linear Gaussian setup with specified trainer type.

    Args:
        trainer_type: Either "nle" or "nre".

    Returns a dict with:
    - prior: MultivariateNormal prior
    - potential_fn: Trained potential function (NLE or NRE based)
    - theta, x: Simulation data
    - likelihood_shift, likelihood_cov: Likelihood parameters
    - num_dim: Dimensionality
    - trainer_type: The trainer type used
    """
    torch.manual_seed(42)

    num_dim = 2
    num_simulations = 7000
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.25 * eye(num_dim)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    # Generate simulation data
    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    # Train estimator and create potential based on trainer type
    if trainer_type == "nle":
        trainer = NLE(prior=prior, show_progress_bars=False, density_estimator="nsf")
        trainer.append_simulations(theta, x)
        estimator = trainer.train()
        potential_fn, _ = likelihood_estimator_based_potential(
            likelihood_estimator=estimator,
            prior=prior,
            x_o=None,
        )
    elif trainer_type == "nre":
        trainer = NRE(prior=prior, show_progress_bars=False, classifier="mlp")
        trainer.append_simulations(theta, x)
        estimator = trainer.train()
        potential_fn, _ = ratio_estimator_based_potential(
            ratio_estimator=estimator,
            prior=prior,
            x_o=None,
        )
    else:
        raise ValueError(f"Unknown trainer_type: {trainer_type}")

    return {
        "prior": prior,
        "potential_fn": potential_fn,
        "theta": theta,
        "x": x,
        "likelihood_shift": likelihood_shift,
        "likelihood_cov": likelihood_cov,
        "num_dim": num_dim,
        "trainer_type": trainer_type,
    }


@pytest.fixture
def linear_gaussian_setup():
    """Setup for linear Gaussian test problem with trained NLE."""
    return _build_linear_gaussian_setup("nle")


@pytest.fixture(params=["nle", "nre"])
def linear_gaussian_setup_trainers(request):
    """Parametrized setup for linear Gaussian with NLE or NRE."""
    return _build_linear_gaussian_setup(request.param)


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
    samples = posterior.sample((num_samples,))
    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("q", ["maf", "nsf", "nice", "gaussian", "gaussian_diag"])
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

    posterior = VIPosterior(
        potential_fn,
        prior,
        theta_transform=theta_transform,
        q=q,
        num_transforms=2,
        hidden_features=32,
    )
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))
    posterior.train(n_particles=1000, eps=1e-8)
    samples = posterior.sample((num_samples,))
    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg=f"vi with {q}")


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


# =============================================================================
# Serialization
# =============================================================================


def _build_diag_gaussian_q(event_shape, link_transform, device="cpu"):
    """A user-style `q` builder. Module-level, since pickling cannot capture a local."""
    return LearnableGaussian(
        dim=event_shape[0],
        full_covariance=False,
        link_transform=link_transform,
        device=device,
    )


class _ParamNormal(torch.distributions.Independent):
    """A base distribution that exposes its own trainable tensors."""

    def __init__(self, loc, scale):
        base = torch.distributions.Normal(loc, scale, validate_args=False)
        super().__init__(base, 1, validate_args=False)
        self._params = [loc, scale]

    def parameters(self):
        return iter(self._params)


def _make_posterior(kind: str, num_dim: int = 2, prior=None):
    """Build an untrained VIPosterior for each way of specifying `q`.

    Args:
        kind: One of `Q_KINDS`.
        num_dim: Dimensionality of the problem.
        prior: Overrides the default unbounded prior. When given, the link transform is
            derived from it rather than forced to the identity, which is what makes a
            bounded prior meaningful.
    """
    extra = (
        {} if prior is not None else {"theta_transform": torch_tf.identity_transform}
    )
    prior = (
        prior if prior is not None else MultivariateNormal(zeros(num_dim), eye(num_dim))
    )

    if kind in ("distribution", "transformed"):
        loc = zeros(num_dim, requires_grad=True)
        scale = ones(num_dim, requires_grad=True)
        if kind == "distribution":
            q = torch.distributions.Independent(
                torch.distributions.Normal(loc, scale), 1
            )
            extra["parameters"] = [loc, scale]
        else:
            # Trainable tensors sit on the base, so `parameters=` stays optional.
            q = torch.distributions.TransformedDistribution(
                _ParamNormal(loc, scale),
                [torch_tf.AffineTransform(zeros(num_dim), ones(num_dim))],
            )
    elif kind == "callable":
        q = _build_diag_gaussian_q
    else:
        q = "maf" if kind == "flow" else "gaussian"

    posterior = VIPosterior(FakePotential(prior=prior), prior, q=q, **extra)
    posterior.set_default_x(zeros(1, num_dim))
    return posterior


TRAIN_KWARGS = dict(
    max_num_iters=10, check_for_convergence=False, quality_control=False
)
Q_KINDS = ["flow", "gaussian", "callable", "distribution", "transformed"]


@pytest.mark.parametrize("kind", Q_KINDS)
def test_trained_posterior_survives_roundtrip(kind: str):
    """Pickling preserves training and leaves the copy fully usable."""
    num_dim = 2
    posterior = _make_posterior(kind, num_dim)
    posterior.train(**TRAIN_KWARGS)

    torch.manual_seed(0)
    expected = posterior.sample((5,))
    reloaded = pickle.loads(pickle.dumps(posterior))

    assert reloaded._trained_on is not None
    torch.manual_seed(0)
    assert torch.allclose(reloaded.sample((5,)), expected)
    reloaded.train(**TRAIN_KWARGS, retrain_from_scratch=True)
    assert reloaded.sample((5,)).shape == (5, num_dim)


def test_pickling_does_not_mutate_the_original():
    """`pickle.dumps` must leave the posterior it serializes untouched."""
    posterior = _make_posterior("gaussian")
    posterior.train(**TRAIN_KWARGS)
    trained_on, optimizer = posterior._trained_on, posterior._optimizer

    pickle.dumps(posterior)  # Discard the bytes; the original is what matters.

    assert posterior._trained_on is trained_on
    assert posterior._optimizer is optimizer
    posterior.train(**TRAIN_KWARGS, retrain_from_scratch=True)


def test_deepcopy_of_trained_posterior_rewires_optimizer():
    """A copy's optimizer must step the copy's own `q`, not the original's."""
    posterior = _make_posterior("gaussian")
    posterior.train(**TRAIN_KWARGS)

    copied = deepcopy(posterior)
    copied.train(**TRAIN_KWARGS)

    assert copied._optimizer.q is copied._q
    assert copied._optimizer.q is not posterior._q


def test_multi_round_q_is_an_independent_warm_start():
    """Passing a trained posterior as `q` copies its flow rather than aliasing it."""
    first = _make_posterior("flow")
    first.train(**TRAIN_KWARGS)

    prior = MultivariateNormal(zeros(2), eye(2))
    second = VIPosterior(FakePotential(prior=prior), prior, q=first)

    assert second._q is not first._q
    assert all(
        torch.allclose(a, b)
        for a, b in zip(second._q.parameters(), first._q.parameters(), strict=True)
    )


def test_amortized_posterior_survives_roundtrip():
    """Amortized mode had no serialization coverage."""
    num_dim = 2
    posterior = _make_posterior("flow", num_dim)
    theta = posterior._prior.sample((128,))
    posterior.train_amortized(
        theta, torch.randn(128, num_dim), max_num_iters=3, batch_size=16
    )

    x_o = zeros(1, num_dim)
    for candidate in (pickle.loads(pickle.dumps(posterior)), deepcopy(posterior)):
        assert candidate._mode == "amortized"
        assert candidate.sample((5,), x=x_o).shape == (5, num_dim)


def test_retrain_from_scratch_with_bounded_prior():
    """Retraining an already-wrapped `q` must not apply the link transform twice.

    Needs a bounded prior: with an unbounded one the link transform is the identity
    and applying it twice is unobservable.
    """
    num_dim = 2
    prior = torch.distributions.Independent(Gamma(2 * ones(num_dim), ones(num_dim)), 1)
    posterior = _make_posterior("distribution", num_dim, prior=prior)
    posterior.train(**TRAIN_KWARGS)

    posterior.train(**TRAIN_KWARGS, retrain_from_scratch=True)

    samples = posterior.sample((20,))
    assert samples.shape == (20, num_dim)
    assert torch.isfinite(posterior.log_prob(samples)).all()


@pytest.mark.parametrize("nested", [False, True])
def test_parameters_outside_q_and_its_base_must_be_passed(nested: bool):
    """sbi only looks on `q` and its base, so anything else must fail loudly.

    Torch keeps a `ComposeTransform` as one entry in `q.transforms`, so a transform
    nested inside one has to be found too.
    """
    num_dim = 2
    transform = torch_tf.AffineTransform(zeros(num_dim), ones(num_dim))
    transform.parameters = lambda: iter(())  # A transform owning trainable tensors.
    q = torch.distributions.TransformedDistribution(
        _ParamNormal(zeros(num_dim, requires_grad=True), ones(num_dim)),
        [torch_tf.ComposeTransform([transform]) if nested else transform],
    )

    with pytest.raises(ValueError, match="transforms"):
        AdaptedVariationalDistribution(
            q,
            MultivariateNormal(zeros(num_dim), eye(num_dim)),
            torch_tf.identity_transform,
        )


def test_retrain_fails_after_setting_an_already_built_q():
    """An already-built `q` cannot be rebuilt, so retraining must fail loudly."""
    posterior = _make_posterior("flow")
    replacement = LearnableGaussian(
        dim=2, full_covariance=False, link_transform=posterior.link_transform
    )

    posterior.set_q(replacement)

    assert posterior.q is replacement
    with pytest.raises(ValueError, match="Cannot rebuild"):
        posterior.train(**TRAIN_KWARGS, retrain_from_scratch=True)


# One representative per side of the asymmetry; the round-trip test above covers
# every kind, this only pins fresh-weights against restored-snapshot.
@pytest.mark.parametrize("kind", ["flow", "distribution"])
def test_retrain_from_scratch_semantics_per_kind(kind: str):
    """Pin the documented per-kind asymmetry of `retrain_from_scratch`."""
    posterior = _make_posterior(kind)
    initial = [p.detach().clone() for p in posterior._q.parameters()]
    posterior.train(**TRAIN_KWARGS)
    trained = [p.detach().clone() for p in posterior._q.parameters()]

    # Not via `train(retrain_from_scratch=True)`: it would train over the rebuild.
    rebuilt = [p.detach().clone() for p in posterior._build_q_from_spec().parameters()]

    if kind == "distribution":
        assert all(torch.allclose(a, b) for a, b in zip(initial, rebuilt, strict=True))
    else:
        assert any(
            not torch.allclose(a, b) for a, b in zip(trained, rebuilt, strict=True)
        )


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
    max_num_iters = 5
    posterior.train(max_num_iters=max_num_iters)

    posterior.train(max_num_iters=max_num_iters, optimizer=torch.optim.SGD)
    assert isinstance(posterior._optimizer._optimizer, torch.optim.SGD)

    posterior.train(max_num_iters=max_num_iters, stick_the_landing=True)
    assert posterior._optimizer.stick_the_landing

    posterior.vi_method = "alpha"
    posterior.train(max_num_iters=max_num_iters)
    posterior.train(max_num_iters=max_num_iters, alpha=0.9)
    assert posterior._optimizer.alpha == 0.9

    posterior.vi_method = "IW"
    posterior.train(max_num_iters=max_num_iters)
    posterior.train(max_num_iters=max_num_iters, K=32)
    assert posterior._optimizer.K == 32

    # Test sampling from trained posterior
    posterior.sample()

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


@pytest.mark.parametrize("num_dim", (1, 2, 5))
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
def test_amortized_vi_accuracy(linear_gaussian_setup_trainers):
    """Test that amortized VI produces accurate posteriors (NLE and NRE)."""
    setup = linear_gaussian_setup_trainers

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

    # Verify training completed successfully
    assert posterior._mode == "amortized"

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
        assert c2st_score < 0.65, (
            f"C2ST too high for {setup['trainer_type']}, "
            f"x_o={x_o.squeeze().tolist()}: {c2st_score:.3f}"
        )


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


def test_amortized_vi_with_fake_potential():
    """Fast test for amortized VI using FakePotential (no NLE training required).

    This test runs in CI (not marked slow) to ensure amortized VI coverage.
    Uses FakePotential where the posterior equals the prior.
    """
    torch.manual_seed(42)

    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    potential_fn = FakePotential(prior=prior)

    # Generate mock simulation data (not actually used for training potential)
    theta = prior.sample((500,))
    x = theta + 0.1 * torch.randn_like(theta)  # Noisy observations

    posterior = VIPosterior(
        potential_fn=potential_fn,
        prior=prior,
    )

    # Train amortized VI
    posterior.train_amortized(
        theta=theta,
        x=x,
        max_num_iters=2,  # Fewer iterations for speed
        show_progress_bar=False,
        flow_type=ZukoFlowType.NSF,
        num_transforms=2,
        hidden_features=16,  # Smaller network for speed
    )

    # Verify training completed
    assert posterior._mode == "amortized"

    # Test sampling works
    x_test = torch.randn(1, num_dim)
    samples = posterior.sample((100,), x=x_test)
    assert samples.shape == (100, num_dim)

    # Test log_prob works
    log_probs = posterior.log_prob(samples, x=x_test)
    assert log_probs.shape == (100,)
    assert torch.isfinite(log_probs).all()
