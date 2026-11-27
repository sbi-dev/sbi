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
import warnings
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributions.transforms as torch_tf
from torch import eye, ones, zeros
from torch.distributions import Beta, Binomial, Gamma, Independent, MultivariateNormal

from sbi.inference import (
    NLE,
    NPE,
    NRE,
    EnsemblePosterior,
    VectorFieldPosterior,
    likelihood_estimator_based_potential,
)
from sbi.inference.posteriors import VIPosterior, VIPosteriorParameters
from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference.potentials.ratio_based_potential import (
    ratio_estimator_based_potential,
)
from sbi.neural_nets import posterior_flow_nn
from sbi.neural_nets.factory import ZukoFlowType
from sbi.samplers.vi.vi_divergence_optimizers import (
    ElboOptimizer,
    ForwardKLOptimizer,
    IWElboOptimizer,
    RenyiDivergenceOptimizer,
)
from sbi.samplers.vi.vi_quality_control import (
    _METRIC_MESSAGE,
    _QUALITY_METRIC,
    QualityControlError,
)
from sbi.samplers.vi.vi_utils import AdaptedVariationalDistribution, LearnableGaussian
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils import BoxUniform, MultipleIndependent, mcmc_transform
from sbi.utils.metrics import c2st, check_c2st
from sbi.utils.sbiutils import (
    CallableTransform,
    _apply_to_transform,
    within_support,
)
from sbi.utils.user_input_checks import process_x
from tests.test_utils import zuko_hidden_layer_widths

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

    def bind(self, x_o: torch.Tensor, x_is_iid: bool = True) -> "FakePotential":
        """Create new potential with x bound, without mutable state."""

        bound = FakePotential(prior=self.prior, device=self.device)
        x_o = process_x(x_o).to(self.device)
        bound._x_o = x_o
        bound._x_is_iid = x_is_iid
        return bound


def make_tractable_potential(target_distribution, prior):
    """Create a potential function from a known target distribution."""

    class TractablePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return target_distribution.log_prob(
                torch.as_tensor(theta, dtype=torch.float32)
            )

        def bind(
            self, x_o: torch.Tensor, x_is_iid: bool = True
        ) -> "TractablePotential":
            """Create new potential with x bound, without mutable state."""

            bound = TractablePotential(prior=self.prior, device=self.device)
            x_o = process_x(x_o).to(self.device)
            bound._x_o = x_o
            bound._x_is_iid = x_is_iid
            return bound

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


class _ModuleNormal(torch.nn.Module, torch.distributions.Distribution):
    """A trainable distribution built as a module, a natural way to write one."""

    support = torch.distributions.constraints.real_vector
    has_rsample = True
    arg_constraints: dict = {}

    def __init__(self, num_dim: int):
        torch.nn.Module.__init__(self)
        self.loc = torch.nn.Parameter(zeros(num_dim))
        self.log_scale = torch.nn.Parameter(zeros(num_dim))
        torch.distributions.Distribution.__init__(
            self, torch.Size([]), torch.Size([num_dim]), validate_args=False
        )

    def _dist(self):
        return torch.distributions.Independent(
            torch.distributions.Normal(self.loc, self.log_scale.exp()), 1
        )

    def rsample(self, sample_shape=torch.Size()):
        return self._dist().rsample(sample_shape)

    def log_prob(self, value):
        return self._dist().log_prob(value)


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
    """Pickling preserves training, leaves the original untouched, and the copy
    stays fully usable."""
    num_dim = 2
    posterior = _make_posterior(kind, num_dim)
    posterior.train(**TRAIN_KWARGS)
    trained_on, optimizer = posterior._trained_on, posterior._optimizer

    torch.manual_seed(0)
    expected = posterior.sample((5,))
    reloaded = pickle.loads(pickle.dumps(posterior))

    assert posterior._trained_on is trained_on
    assert posterior._optimizer is optimizer
    assert reloaded._trained_on is not None
    torch.manual_seed(0)
    assert torch.allclose(reloaded.sample((5,)), expected)
    reloaded.train(**TRAIN_KWARGS, retrain_from_scratch=True)
    assert reloaded.sample((5,)).shape == (5, num_dim)


# Deepcopy shares the pickle path, so one seeded compare per family suffices.
@pytest.mark.parametrize("q", FLOWS)
def test_every_flow_family_pickles(q: str):
    """Each family builds different modules, which all must survive pickling."""
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    posterior = VIPosterior(
        FakePotential(prior=prior),
        prior,
        q=q,
        theta_transform=torch_tf.identity_transform,
    )

    reloaded = pickle.loads(pickle.dumps(posterior))

    torch.manual_seed(0)
    expected = posterior._q.sample((3,))
    torch.manual_seed(0)
    assert torch.allclose(reloaded._q.sample((3,)), expected)


@pytest.mark.parametrize(
    "hidden_features, widths", [(24, [24, 24]), ([24] * 3, [24] * 3)]
)
def test_vi_hidden_features_set_the_hidden_layers(hidden_features, widths):
    """An int gives two hidden layers; a list, as old checkpoints need, sets them."""
    prior = MultivariateNormal(zeros(2), eye(2))
    params = VIPosteriorParameters(
        q="maf", num_transforms=3, hidden_features=hidden_features
    )
    posterior = VIPosterior(FakePotential(prior=prior), prior, **asdict(params))
    posterior.train_amortized(
        prior.sample((32,)), torch.randn(32, 2), max_num_iters=2, batch_size=16
    )

    assert zuko_hidden_layer_widths(posterior._q) == [widths] * 3
    assert zuko_hidden_layer_widths(posterior._amortized_q) == [widths] * 3


def test_vi_family_pickled_with_plain_link_attribute():
    """Families pickled by sbi 0.26 and 0.27 held the link as `_link_transform`."""
    link = mcmc_transform(BoxUniform(zeros(2), ones(2))).inv
    q = LearnableGaussian(2, link_transform=link)
    old = deepcopy(q)
    del old._link
    old._link_transform = link

    theta = torch.tensor([[0.3, 0.7]])
    loaded = pickle.loads(pickle.dumps(old))
    assert torch.equal(loaded.log_prob(theta), q.log_prob(theta))
    assert isinstance(loaded._link, CallableTransform)
    assert not hasattr(loaded, "_link_transform")


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
    """Pickling, deepcopy and `q=` cloning keep the trained amortized flow."""
    num_dim = 2
    posterior = _make_posterior("flow", num_dim)
    theta = posterior._prior.sample((128,))
    posterior.train_amortized(
        theta, torch.randn(128, num_dim), max_num_iters=3, batch_size=16
    )

    x_o = zeros(1, num_dim)
    clone = VIPosterior(posterior.potential_fn, posterior._prior, q=posterior)
    for candidate in (
        pickle.loads(pickle.dumps(posterior)),
        deepcopy(posterior),
        clone,
    ):
        assert candidate._mode == "amortized"
        assert candidate.sample((5,), x=x_o).shape == (5, num_dim)
    assert clone._amortized_q is not posterior._amortized_q


@pytest.mark.parametrize("mode", ["single_x", "amortized"])
def test_copied_posterior_keeps_trained_link(mode):
    """A `q=` copy keeps its trained link: a rebuilt one would re-estimate the
    moments of this prior from samples and shift the density."""
    num_dim = 2
    prior = torch.distributions.TransformedDistribution(
        MultivariateNormal(zeros(num_dim), eye(num_dim)),
        [torch_tf.AffineTransform(ones(num_dim), 2 * ones(num_dim))],
    )
    posterior = _make_posterior("flow", num_dim, prior=prior)
    x_o = zeros(1, num_dim)
    if mode == "amortized":
        posterior.train_amortized(
            prior.sample((128,)), torch.randn(128, num_dim), max_num_iters=3
        )
    else:
        posterior.set_default_x(x_o)
        posterior.train(**TRAIN_KWARGS)

    clone = VIPosterior(posterior.potential_fn, prior, q=posterior).to("cpu")

    theta = prior.sample((10,))
    assert torch.allclose(
        clone.log_prob(theta, x=x_o), posterior.log_prob(theta, x=x_o)
    )


def test_link_transform_moves_with_its_distribution():
    """A CDF link moves the tensors of the distribution it holds."""
    normal = torch.distributions.Normal(zeros(2), ones(2))
    link = torch_tf.CumulativeDistributionTransform(normal)

    _apply_to_transform(link, lambda t: t.to("meta"))

    assert normal.loc.device.type == "meta"
    assert normal.scale.device.type == "meta"


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


# The three wrapper shapes: parameters passed explicitly, auto-detected from the
# base, and exposed by a module.
@pytest.mark.parametrize("kind", ["distribution", "transformed", "module"])
def test_adapted_q_moves_across_devices(kind: str):
    """`VIPosterior.to` only moves a `q` that offers `to`, so the wrapper must.

    A module moves to the meta device, observable without a GPU. Plain tensors move
    by `.data` reassignment, which meta refuses, so those kinds need real hardware.
    """
    num_dim = 2
    if kind == "module":
        device = "meta"
        prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
        posterior = VIPosterior(
            FakePotential(prior=prior),
            prior,
            q=_ModuleNormal(num_dim),
            theta_transform=torch_tf.identity_transform,
        )
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            pytest.skip("Moving plain tensors needs a real second device.")
        posterior = _make_posterior(kind, num_dim)

    posterior.to(device)

    assert all(p.device.type == device for p in posterior.q.parameters())


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


def _count_train_calls(monkeypatch) -> list:
    """Register the metrics `"always_fails"` and `"fails_once"` and record `train`
    calls."""

    def always_fails(posterior, N):
        raise QualityControlError("Broken on purpose.")

    checks = []

    def fails_once(posterior, N):
        checks.append(None)
        if len(checks) == 1:
            raise QualityControlError("Broken on purpose.")
        return 0.0

    for name, metric in [("always_fails", always_fails), ("fails_once", fails_once)]:
        monkeypatch.setitem(_QUALITY_METRIC, name, metric)
        monkeypatch.setitem(_METRIC_MESSAGE, name, "")
    calls = []
    train = VIPosterior.train

    def counting_train(self, *args, **kwargs):
        calls.append(kwargs)
        return train(self, *args, **kwargs)

    monkeypatch.setattr(VIPosterior, "train", counting_train)
    return calls


def test_quality_control_checks_the_trained_observation():
    """The checks sampled at the default x, which differs from an `x` passed to
    `train`, and the failure retrained `q` on the default x instead."""
    posterior = _make_posterior("flow")
    x = 3 * ones(1, 2)

    posterior.train(x=x, max_num_iters=10, show_progress_bar=False)

    assert torch.equal(posterior._trained_on, x)


def test_failed_quality_control_retrains_once_with_the_callers_settings(monkeypatch):
    """A check that kept failing retrained until the recursion limit, and each retry
    dropped the caller's `x` and training settings."""
    calls = _count_train_calls(monkeypatch)
    posterior = _make_posterior("flow")
    x = 3 * ones(1, 2)

    with pytest.warns(UserWarning, match="may be unreliable"):
        posterior.train(
            x=x,
            max_num_iters=10,
            check_for_convergence=False,
            show_progress_bar=False,
            quality_control_metric="always_fails",
        )

    assert len(calls) == 2
    assert calls[1]["max_num_iters"] == 10
    assert calls[1]["learning_rate"] == pytest.approx(1e-4)
    assert torch.equal(posterior._trained_on, x)


def test_failed_quality_control_accepts_a_retrained_q_that_passes(monkeypatch):
    """A check that fails once retrains once and then accepts the retrained `q`."""
    calls = _count_train_calls(monkeypatch)
    posterior = _make_posterior("flow")
    x = 3 * ones(1, 2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        posterior.train(
            x=x,
            max_num_iters=10,
            check_for_convergence=False,
            show_progress_bar=False,
            quality_control_metric="fails_once",
        )

    messages = [str(w.message) for w in caught]
    assert any("Retraining the variational posterior" in m for m in messages)
    assert not any("may be unreliable" in m for m in messages)
    assert len(calls) == 2
    assert torch.equal(posterior._trained_on, x)


def test_failed_quality_control_does_not_retrain_an_already_built_q(monkeypatch):
    """An already-built `q` cannot be rebuilt, so a failed check only warns."""
    posterior = _make_posterior("flow")
    posterior.set_q(
        LearnableGaussian(
            dim=2, full_covariance=False, link_transform=posterior.link_transform
        )
    )
    calls = _count_train_calls(monkeypatch)

    with pytest.warns(UserWarning, match="may be unreliable"):
        posterior.train(
            max_num_iters=10,
            check_for_convergence=False,
            show_progress_bar=False,
            quality_control_metric="always_fails",
        )

    assert len(calls) == 1


@pytest.mark.parametrize("dreg", [False, True])
@pytest.mark.parametrize("offset", [0.0, 100.0, -100.0])
def test_iwelbo_optimizer_uses_importance_weighted_loss(
    monkeypatch, dreg: bool, offset: float
):
    """IW uses IWELBO (or DReG) gradient weights and a stable displayed loss."""
    prior = MultivariateNormal(zeros(1), eye(1))
    optimizer = IWElboOptimizer(
        potential_fn=FakePotential(prior=prior),
        q=LearnableGaussian(dim=1, full_covariance=False),
        prior=prior,
        n_particles=2,
        K=3,
        dreg=dreg,
        lr=1e-3,
        gamma=0.999,
    )
    particles = (
        torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]) + offset
    ).requires_grad_()
    monkeypatch.setattr(
        optimizer, "generate_elbo_particles", lambda x_o, num_samples=None: particles
    )
    surrogate_loss, loss = optimizer.loss(zeros(1))

    # IWELBO gradient weights (Burda et al. 2016), squared for DReG (Tucker et al.
    # 2018), averaged over the n_particles groups of K samples.
    weights = torch.softmax(particles.detach().reshape(2, 3), dim=-1)
    if dreg:
        weights = weights.square()
    grad = torch.autograd.grad(surrogate_loss, particles)[0]
    assert torch.allclose(grad, -weights.flatten() / 2)
    assert not loss.requires_grad

    # With K equal log-weights c, the IWELBO is c for any K.
    monkeypatch.setattr(
        optimizer,
        "generate_elbo_particles",
        lambda x_o, num_samples=None: torch.full((6,), offset),
    )
    _, loss = optimizer.loss(zeros(1))
    assert torch.allclose(loss, torch.tensor(-offset))


class _OffsetPotential(BasePotential):
    """The prior log probability plus a constant, an unnormalized potential."""

    def __init__(self, prior, offset: float):
        super().__init__(prior)
        self.offset = offset

    def __call__(self, theta, track_gradients: bool = True):
        return self.prior.log_prob(theta) + self.offset


class _ZeroPotential(BasePotential):
    """A constant potential: the ELBO gradient is the one of the entropy alone."""

    def __call__(self, theta, track_gradients: bool = True):
        return torch.zeros(theta.shape[0])


def test_elbo_differentiates_a_vector_field_potential():
    """The ELBO gradient includes the potential, also for a vector field potential,
    which does not track gradients by default."""
    prior = MultivariateNormal(zeros(2), eye(2))
    theta = prior.sample((100,))
    estimator = posterior_flow_nn(hidden_features=8, num_layers=1)(
        theta, theta + 0.1 * torch.randn_like(theta)
    )
    potentials = {
        "vector field": VectorFieldPosterior(estimator, prior).potential_fn,
        "zero": _ZeroPotential(prior),
    }

    def location_gradient(potential_fn):
        torch.manual_seed(0)
        q = LearnableGaussian(dim=2, full_covariance=False)
        optimizer = ElboOptimizer(
            potential_fn=potential_fn,
            q=q,
            prior=prior,
            n_particles=8,
            lr=1e-3,
            gamma=0.999,
        )
        surrogate, _ = optimizer.loss(zeros(1, 2))
        return torch.autograd.grad(surrogate, next(q.parameters()))[0]

    assert not torch.allclose(
        location_gradient(potentials["vector field"]),
        location_gradient(potentials["zero"]),
    )


def test_forward_kl_loss_ignores_a_constant_in_the_potential():
    """Self-normalized fKL weights do not depend on a constant in the potential.

    With an offset of -200, exp() of every log weight underflows in float32.
    """
    prior = MultivariateNormal(zeros(2), eye(2))

    def loss_with_offset(offset: float):
        torch.manual_seed(0)
        optimizer = ForwardKLOptimizer(
            potential_fn=_OffsetPotential(prior, offset),
            q=LearnableGaussian(dim=2, full_covariance=False),
            prior=prior,
            n_particles=64,
            lr=1e-3,
            gamma=0.999,
        )
        return optimizer.loss(zeros(2))

    surrogate, loss = loss_with_offset(0.0)
    surrogate_low, loss_low = loss_with_offset(-200.0)
    assert torch.isfinite(surrogate_low)
    assert torch.allclose(surrogate_low, surrogate)
    assert torch.allclose(loss_low, loss - 200.0)


def test_forward_kl_loss_leaves_out_particles_outside_the_support(monkeypatch):
    """A particle outside the target's support adds nothing to the displayed fKL
    loss (0 * log 0 = 0), even when `weight_transform` gives it a nonzero weight.
    `weight_transform` gets weights whose largest is 1, and its output is normalized
    again."""
    prior = MultivariateNormal(zeros(1), eye(1))
    shifted = MultivariateNormal(ones(1), 0.25 * eye(1))

    def target_log_prob(theta):
        outside = theta[..., 0] < -1.5
        return torch.where(outside, -torch.inf, shifted.log_prob(theta))

    target = SimpleNamespace(log_prob=target_log_prob)
    weight_maxima = []

    def floor_weights(weights):
        weight_maxima.append(weights.max())
        return weights + 0.1

    optimizer = ForwardKLOptimizer(
        potential_fn=make_tractable_potential(target, prior),
        q=LearnableGaussian(dim=1, full_covariance=False),
        prior=prior,
        n_particles=6,
        weight_transform=floor_weights,
        lr=1e-3,
        gamma=0.999,
    )
    particles = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]).unsqueeze(-1)
    monkeypatch.setattr(optimizer.q, "sample", lambda sample_shape: particles)
    surrogate_loss, loss = optimizer.loss(zeros(1))

    logq = optimizer.q.log_prob(particles)
    logweights = target.log_prob(particles) - logq.detach()
    weights = (logweights - logweights.max()).exp() + 0.1
    weights /= weights.sum()
    assert torch.allclose(surrogate_loss, -(weights * logq).sum())

    # The displayed loss averages over the particles inside the support only.
    inside = logweights[1:]
    inside_weights = (inside - inside.max()).exp() + 0.1
    assert torch.allclose(loss, (inside_weights / inside_weights.sum() * inside).sum())
    assert torch.allclose(torch.stack(weight_maxima), torch.ones(1))


def _optimizer_after_losses(losses: list[float]) -> ForwardKLOptimizer:
    prior = MultivariateNormal(zeros(1), eye(1))
    optimizer = ForwardKLOptimizer(
        potential_fn=FakePotential(prior=prior),
        q=LearnableGaussian(dim=1, full_covariance=False),
        prior=prior,
        n_particles=4,
        lr=1e-3,
        gamma=0.999,
    )
    for loss in losses:
        optimizer.update_loss_stats(torch.tensor(loss))
    return optimizer


def test_non_finite_losses_are_left_out_of_loss_stats():
    """Non-finite losses, as from a batch entirely outside the support, neither
    enter the running statistics nor count towards convergence."""
    nan = float("nan")
    before_first_finite = _optimizer_after_losses([nan]).get_loss_stats()
    assert torch.isnan(torch.stack(before_first_finite)).all()

    assert not _optimizer_after_losses([nan, 2.0, 2.0] + [nan] * 49).converged()
    assert _optimizer_after_losses([nan, 2.0, 2.0] + [2.0] * 49).converged()

    constant = [50.0] * 60
    with_nan = constant[:30] + [nan] + constant[30:]
    assert torch.allclose(
        torch.stack(_optimizer_after_losses(with_nan).get_loss_stats()),
        torch.stack(_optimizer_after_losses(constant).get_loss_stats()),
    )


@pytest.mark.parametrize("surrogate", ["nan", "nan gradient"])
def test_rolled_back_step_does_not_end_training(monkeypatch, surrogate: str):
    """A non-finite gradient loss, or a finite one with nan gradients (as from a
    potential `log(density * mask)`), rolls q back and warms it up towards the prior.
    The step makes no update with those gradients, restarts the loss statistics, and
    counts towards the snapshot schedule without a snapshot of the rolled-back q."""
    optimizer = _optimizer_after_losses([2.0] * 60)
    assert optimizer.converged()

    loc = next(optimizer.q.parameters())  # zeros: sqrt has an infinite derivative
    surrogates = {
        "nan": lambda: loc.sum() * float("nan"),
        "nan gradient": lambda: (loc.sqrt() * 0).sum(),
    }
    loss = lambda x_o: (surrogates[surrogate](), torch.tensor(2.0))
    monkeypatch.setattr(optimizer, "loss", loss)
    resolve_state, recovered = optimizer.resolve_state, []

    def record_recovery():
        resolve_state()
        recovered.extend(p.detach().clone() for p in optimizer.q.parameters())

    monkeypatch.setattr(optimizer, "resolve_state", record_recovery)
    optimizer._num_updates = 49  # this step is due for a snapshot
    snapshot = [p.clone() for p in optimizer.state_dict]
    optimizer.step(zeros(1))
    assert optimizer.num_step == 0
    assert optimizer._num_updates == 50
    for p, p_recovered in zip(optimizer.q.parameters(), recovered, strict=True):
        assert torch.equal(p, p_recovered)
    for p_snapshot, p_before in zip(optimizer.state_dict, snapshot, strict=True):
        assert torch.equal(p_snapshot, p_before)


def _loss_and_grads(monkeypatch, optimizer_class, kwargs, noise, outside_value):
    """Gradient loss and its gradients for fixed reparameterized particles, under a
    target whose log density is `outside_value` for theta < -1.5."""
    prior = MultivariateNormal(zeros(1), eye(1))
    shifted = MultivariateNormal(ones(1), 0.25 * eye(1))

    def target_log_prob(theta):
        outside = theta[..., 0] < -1.5
        return torch.where(outside, outside_value, shifted.log_prob(theta))

    target = SimpleNamespace(log_prob=target_log_prob)
    optimizer = optimizer_class(
        potential_fn=make_tractable_potential(target, prior),
        q=LearnableGaussian(dim=1, full_covariance=False),
        prior=prior,
        lr=1e-3,
        gamma=0.999,
        **kwargs,
    )
    q = optimizer.q

    def sample_and_log_prob(sample_shape):
        # Reparameterized, so gradients also flow through the potential.
        samples = q.loc + q.log_scale.exp() * torch.tensor(noise).unsqueeze(-1)
        return samples, q.log_prob(samples)

    monkeypatch.setattr(q, "sample_and_log_prob", sample_and_log_prob)
    surrogate, _ = optimizer.loss(zeros(1))
    return surrogate, torch.autograd.grad(surrogate, list(q.parameters()))


@pytest.mark.parametrize(
    "optimizer_class, kwargs",
    [
        (IWElboOptimizer, {"n_particles": 3, "K": 2}),
        (RenyiDivergenceOptimizer, {"n_particles": 6}),
    ],
)
def test_iw_and_alpha_losses_leave_out_particles_outside_the_support(
    monkeypatch, optimizer_class, kwargs
):
    """A particle where the potential is -inf (as a `RestrictedPrior` gives) has
    weight 0 and adds nothing to the IW or alpha gradient loss (0 * log 0 = 0), as a
    particle whose weight underflows to 0."""
    noise = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
    args = (monkeypatch, optimizer_class, kwargs, noise)
    surrogate, grads = _loss_and_grads(*args, -torch.inf)
    surrogate_underflow, grads_underflow = _loss_and_grads(*args, -1e4)
    assert torch.allclose(surrogate, surrogate_underflow)
    for grad, grad_underflow in zip(grads, grads_underflow, strict=True):
        assert torch.isfinite(grad).all()
        assert torch.allclose(grad, grad_underflow)


def test_iw_loss_leaves_out_groups_entirely_outside_the_support(monkeypatch):
    """An IW group with every particle outside the support has no estimate, so the
    loss leaves it out, as if it had not been drawn."""
    inside = [0.0, 0.5, 1.0, 2.0]
    surrogate, grads = _loss_and_grads(
        monkeypatch,
        IWElboOptimizer,
        {"n_particles": 3, "K": 2},
        [-2.0, -1.8, *inside],
        -torch.inf,
    )
    surrogate_ref, grads_ref = _loss_and_grads(
        monkeypatch, IWElboOptimizer, {"n_particles": 2, "K": 2}, inside, -torch.inf
    )
    assert torch.allclose(surrogate, surrogate_ref)
    for grad, grad_ref in zip(grads, grads_ref, strict=True):
        assert torch.allclose(grad, grad_ref)


def test_rollback_keeps_the_optimizer_settings(monkeypatch):
    """Recovery resets the optimizer's state, not its settings, including those a
    scheduler has changed, nor hooks registered on it."""
    prior = MultivariateNormal(zeros(1), eye(1))
    optimizer = ForwardKLOptimizer(
        potential_fn=FakePotential(prior=prior),
        q=LearnableGaussian(dim=1, full_covariance=False),
        prior=prior,
        optimizer=torch.optim.SGD,
        momentum=0.9,
        weight_decay=0.1,
        lr=1e-3,
        gamma=0.999,
    )
    monkeypatch.setattr(optimizer, "warm_up", lambda num_steps: None)
    hook_calls = []
    optimizer._optimizer.register_step_post_hook(lambda *args: hook_calls.append(1))
    for p in optimizer.q.parameters():
        p.grad = torch.ones_like(p)
    optimizer._optimizer.step()  # fills the momentum buffers
    settings = optimizer._optimizer.param_groups[0]
    settings.update(lr=1e-4, momentum=0.5)  # as a scheduler may set them
    optimizer.resolve_state()
    assert optimizer._optimizer.param_groups[0] is settings
    kept = [settings[key] for key in ("lr", "momentum", "weight_decay")]
    assert kept == [1e-4, 0.5, 0.1]
    assert not optimizer._optimizer.state
    optimizer._optimizer.step()
    assert len(hook_calls) == 2


def test_alpha_loss_at_alpha_one_is_the_elbo(monkeypatch):
    """At alpha = 1 the displayed loss is the negative ELBO, not the formula's 0 / 0,
    which would never let training converge."""
    prior = MultivariateNormal(zeros(1), eye(1))
    optimizer = RenyiDivergenceOptimizer(
        potential_fn=FakePotential(prior=prior),
        q=LearnableGaussian(dim=1, full_covariance=False),
        prior=prior,
        n_particles=4,
        alpha=1.0,
        lr=1e-3,
        gamma=0.999,
    )
    elbo_particles = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    monkeypatch.setattr(
        optimizer,
        "generate_elbo_particles",
        lambda x_o, num_samples=None: elbo_particles,
    )
    _, loss = optimizer.loss(zeros(1))
    assert torch.allclose(loss, torch.tensor(2.5))


def test_optimizer_without_lr_gets_the_default_lr():
    """Without `lr`, the optimizer gets sbi's default, not torch's (1.0 for
    Adadelta)."""
    prior = MultivariateNormal(zeros(1), eye(1))
    optimizer = ForwardKLOptimizer(
        potential_fn=FakePotential(prior=prior),
        q=LearnableGaussian(dim=1, full_covariance=False),
        prior=prior,
        optimizer=torch.optim.Adadelta,
        gamma=0.999,
    )
    assert optimizer._optimizer.param_groups[0]["lr"] == 1e-3


def test_reset_loss_stats_snapshots_the_current_q():
    """Each `train` call starts with a snapshot of the current q, so a rollback in it
    does not restore a q from an earlier call."""
    optimizer = _optimizer_after_losses([2.0])
    with torch.no_grad():
        for p in optimizer.q.parameters():
            p.add_(1.0)
    optimizer.reset_loss_stats()
    q_params = optimizer.q.parameters()
    for p_snapshot, p in zip(optimizer.state_dict, q_params, strict=True):
        assert torch.equal(p_snapshot, p)


@pytest.mark.parametrize("optimizer_class", [IWElboOptimizer, RenyiDivergenceOptimizer])
@pytest.mark.parametrize("dreg_before, dreg_after", [(False, True), (True, False)])
def test_dreg_changed_by_update_matches_new_optimizer(
    optimizer_class, dreg_before: bool, dreg_after: bool
):
    """Switching `dreg` between `train` calls gives the gradient of a new optimizer."""
    prior = MultivariateNormal(2 * ones(1), eye(1))
    q = LearnableGaussian(dim=1, full_covariance=False)

    def gradient(dreg: bool, update_to: bool | None = None):
        optimizer = optimizer_class(
            potential_fn=FakePotential(prior=prior),
            q=deepcopy(q),
            prior=prior,
            n_particles=4,
            dreg=dreg,
            lr=1e-3,
            gamma=0.999,
        )
        if update_to is not None:
            # As a second `train` call does; it also rebuilds the scheduler.
            optimizer.update({"dreg": update_to, "gamma": 0.999})
        torch.manual_seed(0)
        surrogate_loss, _ = optimizer.loss(zeros(1))
        return torch.autograd.grad(surrogate_loss, list(optimizer.q.parameters()))

    for updated, new in zip(
        gradient(dreg_before, update_to=dreg_after), gradient(dreg_after), strict=True
    ):
        assert torch.allclose(updated, new)


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


@pytest.mark.parametrize("vi_method", ["fKL", "IW", "alpha"])
@pytest.mark.parametrize("via_params", [False, True])
def test_train_amortized_rejects_methods_other_than_rkl(
    vi_method: str, via_params: bool
):
    """Amortized VI fits only rKL, so other methods must raise, not run as rKL."""
    prior = MultivariateNormal(zeros(2), eye(2))
    theta = prior.sample((8,))
    if via_params:
        posterior = VIPosterior(FakePotential(prior=prior), prior)
        kwargs = {"params": VIPosteriorParameters(vi_method=vi_method)}
    else:
        posterior = VIPosterior(FakePotential(prior=prior), prior, vi_method=vi_method)
        kwargs = {}

    with pytest.raises(NotImplementedError, match="rKL"):
        posterior.train_amortized(theta, theta, **kwargs)


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


@pytest.mark.parametrize("mode", ["single_x", "amortized"])
@pytest.mark.parametrize(
    "make_prior, impossible",
    [
        (lambda: BoxUniform(-3 * ones(2), 3 * ones(2)), [4.0, 0.0]),  # outside
        (lambda: Independent(Gamma(2 * ones(2), ones(2)), 1), [0.0, 1.0]),  # edge
    ],
    ids=["box_outside", "gamma_edge"],
)
def test_vi_with_bounded_prior(mode, make_prior, impossible):
    """Samples stay in the support, and θ of density zero gets `-inf`, not a value or
    NaN, also on a `q=` copy whose source moved to another device."""
    prior = make_prior()  # one per case: moving the posterior moves it in place
    theta = prior.sample((300,))
    x = theta + 0.3 * torch.randn_like(theta)
    nle = NLE(prior, show_progress_bars=False)
    nle.append_simulations(theta, x).train(max_num_epochs=3)
    posterior = nle.build_posterior(sample_with="vi")
    x_o = x[:1]
    if mode == "amortized":
        posterior.train_amortized(theta, x, max_num_iters=5, show_progress_bar=False)
    else:
        posterior.set_default_x(x_o)
        posterior.train(max_num_iters=5, quality_control=False, show_progress_bar=False)
    clone = VIPosterior(posterior.potential_fn, prior, q=posterior)

    samples = posterior.sample((50,), x=x_o, show_progress_bars=False)
    assert within_support(prior, samples).all()
    assert torch.isfinite(posterior.log_prob(samples, x=x_o)).all()
    impossible = torch.tensor([impossible])
    assert posterior.log_prob(impossible, x=x_o).item() == -torch.inf

    posterior.to("meta")
    assert clone.log_prob(impossible, x=x_o).item() == -torch.inf


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


def _fast_amortized_posterior():
    """Return a cheaply trained amortized VI posterior and its prior."""
    prior = MultivariateNormal(zeros(2), eye(2))
    theta = prior.sample((100,))
    posterior = VIPosterior(FakePotential(prior=prior), prior=prior)
    posterior.train_amortized(
        theta=theta,
        x=theta + 0.1 * torch.randn_like(theta),
        max_num_iters=3,
        show_progress_bar=False,
        num_transforms=1,
        hidden_features=8,
    )
    return posterior, prior


def test_amortized_vi_takes_one_observation():
    """Amortized q, like a direct posterior, takes one observation in `sample()`,
    `log_prob()` and `potential()`, instead of one posterior per observation or iid
    trials."""
    posterior, _ = _fast_amortized_posterior()
    x_3 = zeros(3, 2)
    with pytest.raises(ValueError, match="takes one observation, but x holds 3"):
        posterior.sample((10,), x=x_3)
    with pytest.raises(ValueError, match="takes one observation, but x holds 3"):
        posterior.log_prob(zeros(3, 2), x=x_3)
    with pytest.raises(ValueError, match="takes one observation, but x holds 3"):
        posterior.potential(zeros(4, 2), x=x_3)


def test_amortized_vi_checks_a_default_x_set_before_training():
    """A default x set before `train_amortized()` is checked against the observation
    shape q was trained on, instead of being broadcast to it."""
    prior = MultivariateNormal(zeros(2), eye(2))
    theta = prior.sample((100,))
    posterior = VIPosterior(FakePotential(prior=prior), prior=prior)
    posterior.set_default_x(zeros(1))
    posterior.train_amortized(
        theta=theta,
        x=theta,
        max_num_iters=3,
        show_progress_bar=False,
        num_transforms=1,
        hidden_features=8,
    )
    with pytest.raises(ValueError, match="one observation has shape"):
        posterior.sample((5,))


def test_amortized_vi_log_prob_batched_matches_log_prob():
    """`log_prob_batched()` scores each theta under the observation at its position,
    as `log_prob()` does one observation at a time, also for several sample axes."""
    posterior, prior = _fast_amortized_posterior()
    x_batch = prior.sample((3,))
    theta = prior.sample((2, 5, 3))

    batched = posterior.log_prob_batched(theta, x_batch)
    assert batched.shape == (2, 5, 3)
    for i in range(3):
        single = posterior.log_prob(theta[..., i, :], x=x_batch[i])
        assert torch.allclose(batched[..., i], single, atol=1e-5)


def test_amortized_vi_ignores_the_x_of_an_earlier_single_x_fit():
    """Amortized q(theta|x) needs an x; it ignores the x and the MAP of a stale
    single-x fit."""
    prior = MultivariateNormal(zeros(2), eye(2))
    theta = prior.sample((100,))
    posterior = VIPosterior(FakePotential(prior=prior), prior, q="gaussian")
    posterior.train(x=zeros(1, 2), **TRAIN_KWARGS)
    posterior._map = zeros(1, 2)
    posterior.train_amortized(
        theta=theta,
        x=theta,
        max_num_iters=3,
        show_progress_bar=False,
        num_transforms=1,
        hidden_features=8,
    )
    assert posterior._map is None
    with pytest.raises(ValueError, match="Context `x` needed"):
        posterior.sample((5,))


def test_single_x_fit_after_an_amortized_fit_takes_several_observations():
    """While an amortized q is in place, the posterior takes one observation, but
    `train(x)` fits a single-x q to all observations in x."""
    prior = MultivariateNormal(zeros(2), eye(2))
    theta = prior.sample((100,))
    posterior = VIPosterior(FakePotential(prior=prior), prior, q="gaussian")
    posterior.train_amortized(
        theta=theta,
        x=theta,
        max_num_iters=3,
        show_progress_bar=False,
        num_transforms=1,
        hidden_features=8,
    )
    posterior.train(x=zeros(3, 2), **TRAIN_KWARGS)
    assert posterior.sample((5,)).shape == (5, 2)


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


def test_vi_posterior_stores_resolved_device():
    """Test that potential and posterior resolve devices, whatever the spelling."""
    prior = MultivariateNormal(zeros(2), eye(2))
    potential_fn = FakePotential(prior=prior, device="cpu:0")
    assert potential_fn.device == "cpu"

    potential_fn.to("cpu:0")
    assert potential_fn.device == "cpu"

    posterior = VIPosterior(potential_fn, prior, device="cpu:0")
    assert posterior._device == "cpu"

    posterior.to("cpu:0")
    assert posterior._device == "cpu"


def test_learnable_gaussian_ignores_the_upper_triangle_of_scale_tril():
    """`scale_tril` is optimized unconstrained, so it drifts off the identity.

    `rsample` then used the full matrix while `log_prob` ignored the upper triangle,
    which biased the ELBO by the difference.
    """
    q = LearnableGaussian(dim=3, full_covariance=True)
    with torch.no_grad():
        q.scale_tril.add_(0.5)

    scale_tril = q._base_dist().scale_tril
    assert torch.equal(scale_tril, scale_tril.tril())


class _CenteredPotential(BasePotential):
    """A potential with its mode at the bound observation."""

    def __call__(self, theta, track_gradients: bool = True):
        return -0.5 * ((theta - self.x_o) ** 2).sum(-1)


@pytest.mark.parametrize("refit", ["train", "set_q"])
def test_map_follows_a_refit_on_another_observation(refit):
    """Without a default x, `map()` uses the x that q was last fit to, not a cached
    MAP of an earlier fit."""
    prior = MultivariateNormal(zeros(2), eye(2))

    def make_posterior():
        return VIPosterior(
            _CenteredPotential(prior),
            prior,
            q="gaussian",
            theta_transform=torch_tf.identity_transform,
        )

    posterior = make_posterior()
    map_kwargs = dict(init_method=zeros(1, 2), num_iter=300, learning_rate=0.1)
    posterior.train(x=zeros(1, 2), **TRAIN_KWARGS)
    posterior.map(**map_kwargs)
    if refit == "train":
        posterior.train(x=3 * ones(1, 2), **TRAIN_KWARGS)
    else:
        other = make_posterior()
        other.train(x=3 * ones(1, 2), **TRAIN_KWARGS)
        posterior.set_q(other)

    assert torch.allclose(posterior.map(**map_kwargs), 3 * ones(1, 2), atol=0.3)


@pytest.mark.parametrize("vi_method", ["rKL", "IW", "fKL", "alpha"])
def test_npe_vi_posterior_fits_several_observations(vi_method):
    """Each VI method binds several observations as iid trials of the NPE potential,
    instead of pairing them with its particles."""
    prior = MultivariateNormal(zeros(2), eye(2))
    theta = prior.sample((200,))
    trainer = NPE(prior, show_progress_bars=False)
    trainer.append_simulations(theta, theta + 0.3 * torch.randn_like(theta))
    trainer.train(max_num_epochs=1)
    posterior = trainer.build_posterior(
        sample_with="vi",
        posterior_parameters=VIPosteriorParameters(vi_method=vi_method),
    )

    with pytest.warns(UserWarning, match="combined as iid trials"):
        posterior.train(x=zeros(3, 2), **TRAIN_KWARGS)

    assert posterior.sample((5,)).shape == (5, 2)


def test_q_fit_on_one_observation_rejects_copies_of_it():
    """A q fit on one observation is not the posterior of several iid copies."""
    posterior = _make_posterior("gaussian")
    posterior.train(**TRAIN_KWARGS)
    copies = posterior.default_x.repeat(3, 1)

    with pytest.raises(ValueError, match="not fit on the specified observation"):
        posterior.sample((5,), x=copies)
    with pytest.raises(ValueError, match="not fit on the specified observation"):
        posterior.log_prob(zeros(1, 2), x=copies)


def test_ensemble_of_vi_fits_to_different_x_raises():
    """Without an x, an ensemble of q fit to different observations would mix their
    posteriors. With one shared x, it uses that x."""
    prior = MultivariateNormal(zeros(2), eye(2))
    fits = []
    for x_o in (2 * ones(1, 2), -2 * ones(1, 2)):
        posterior = VIPosterior(_CenteredPotential(prior), prior, q="gaussian")
        posterior.train(x=x_o, **TRAIN_KWARGS)
        fits.append(posterior)

    with pytest.raises(ValueError, match="do not all use the same observation"):
        EnsemblePosterior(fits).sample((5,))
    assert EnsemblePosterior([fits[0], fits[0]]).sample((5,)).shape == (5, 2)

    weighted = EnsemblePosterior(fits, weights=[1.0, 0.0])
    with pytest.raises(ValueError, match="not fit on the specified observation"):
        weighted.sample((5,), x=2 * ones(1, 2))


def test_ensemble_of_amortized_vi_takes_one_observation_in_log_prob():
    """`log_prob()` of the ensemble raises like `sample()`, instead of pairing each
    theta with one observation."""
    prior = MultivariateNormal(zeros(2), eye(2))
    theta = prior.sample((100,))
    posterior = VIPosterior(FakePotential(prior=prior), prior, q="gaussian")
    posterior.train_amortized(
        theta=theta,
        x=theta,
        max_num_iters=3,
        show_progress_bar=False,
        num_transforms=1,
        hidden_features=8,
    )
    ensemble = EnsemblePosterior([posterior, posterior])
    with pytest.raises(ValueError, match="takes one observation, but x holds 3"):
        ensemble.log_prob(zeros(3, 2), x=zeros(3, 2))


def test_evaluate_scores_q_at_the_x_it_was_fit_to(capsys):
    """A potential evaluated at another x does not change the quality score of q."""
    prior = MultivariateNormal(zeros(2), eye(2))
    posterior = VIPosterior(_CenteredPotential(prior), prior, q="gaussian")
    posterior.train(x=zeros(1, 2), **TRAIN_KWARGS)
    torch.manual_seed(0)
    posterior.evaluate(quality_control_metric="prop", N=1000)
    expected = capsys.readouterr().out

    posterior.potential(zeros(1, 2), x=3 * ones(1, 2))
    torch.manual_seed(0)
    posterior.evaluate(quality_control_metric="prop", N=1000)
    assert capsys.readouterr().out == expected
