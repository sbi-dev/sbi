# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from torch import eye, ones, randn_like, tensor, zeros
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
from torch.testing import assert_close

from sbi.inference import NLE_A, NPE_C, NRE_A
from sbi.inference.posteriors import (
    DirectPosterior,
    EnsemblePosterior,
    MCMCPosterior,
)
from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters
from sbi.neural_nets import posterior_nn
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils import BoxUniform
from sbi.utils.metrics import check_c2st
from tests.test_utils import (
    PosteriorPotential,
    TractablePosterior,
    get_dkl_gaussian_prior,
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "inference_method, num_trials",
    (
        (NPE_C, 1),
        pytest.param(NPE_C, 5, marks=pytest.mark.xfail),
        pytest.param(NLE_A, 1, marks=pytest.mark.mcmc),
        pytest.param(NLE_A, 5, marks=pytest.mark.mcmc),
        pytest.param(NRE_A, 1, marks=pytest.mark.mcmc),
        pytest.param(NRE_A, 5, marks=pytest.mark.mcmc),
    ),
)
def test_c2st_posterior_ensemble_on_linearGaussian(
    inference_method, num_trials, mcmc_params_accurate: MCMCPosteriorParameters
):
    """Test whether EnsemblePosterior infers well a simple example with available
    ground truth.
    """

    num_dim = 2
    ensemble_size = 2
    x_o = zeros(num_trials, num_dim)
    num_samples = 500
    num_simulations = 2000

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

    # train ensemble components
    posteriors = []
    for _ in range(ensemble_size):
        theta = prior.sample((num_simulations,))
        x = simulator(theta)
        inferer = inference_method(prior)
        inferer.append_simulations(theta, x).train(
            training_batch_size=100,
            max_num_epochs=1 if inference_method == "NPE" and num_trials > 1 else 100,
        )
        posteriors.append(inferer.build_posterior())

    # create ensemble
    posterior = EnsemblePosterior(posteriors)
    posterior.set_default_x(x_o)

    # test sampling and evaluation.
    if isinstance(inferer, (NLE_A, NRE_A)):
        samples = posterior.sample(
            (num_samples,),
            **asdict(mcmc_params_accurate),
        )
    else:
        samples = posterior.sample((num_samples,))
    _ = posterior.potential(samples)

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(
        samples,
        target_samples,
        alg="{} posterior ensemble".format(inference_method.__name__),
    )

    map_ = posterior.map(init_method=samples, show_progress_bars=False)
    assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5

    # Checks for log_prob()
    # For the Gaussian prior, we compute the KLd between ground truth and posterior.
    # This step is skipped for NLE since the probabilities are not normalised.
    if isinstance(inferer, (NPE_C, NRE_A)):
        dkl = get_dkl_gaussian_prior(
            posterior,
            x_o[0],
            likelihood_shift,
            likelihood_cov,
            prior_mean,
            prior_cov,
            num_samples=num_samples,
        )
        max_dkl = 0.15
        assert dkl < max_dkl, (
            f"D-KL={dkl} is more than 2 stds above the average performance."
        )

    # test individual log_prob and map
    posterior.log_prob(samples, individually=True)

    # Test sample_batched
    x_o_batch_dim = 2
    if isinstance(inferer, (NLE_A, NRE_A)):
        # TODO: Implement batched sampling for MCMC methods
        return
    else:
        samples = posterior.sample_batched((10,), ones(x_o_batch_dim, num_dim))

    assert samples.shape == (10, x_o_batch_dim, num_dim), "Sample shape wrong"


@pytest.mark.parametrize(
    "weights, expected_exception",
    [
        (None, None),
        ([0.3, 0.7], None),
        (tensor([0.4, 0.6]), None),
        ({"w1": 0.5, "w2": 0.5}, TypeError),
        (0.5, TypeError),
        ((0.5, 0.5), TypeError),
    ],
)
def test_ensemble_posterior_weights(weights, expected_exception):
    """Test EnsemblePosterior weight handling for valid and invalid formats."""
    num_dim = 2
    ensemble_size = 2
    num_simulations = 50
    x_o = zeros(1, num_dim)

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))

    posteriors = []
    for _ in range(ensemble_size):
        theta = prior.sample((num_simulations,))
        x = theta + 0.1 * randn_like(theta)
        inferer = NPE_C(prior)
        inferer.append_simulations(theta, x).train(max_num_epochs=1)
        posteriors.append(inferer.build_posterior())

    if expected_exception:
        with pytest.raises(expected_exception):
            EnsemblePosterior(posteriors, weights=weights)
    else:
        posterior = EnsemblePosterior(posteriors, weights=weights)
        posterior.set_default_x(x_o)
        _ = posterior.sample((2,))


class _GaussianShiftPotential(PosteriorPotential):
    """Potential whose MAP is at the bound `x`."""

    def __call__(self, theta, track_gradients=True):
        return -((theta - self.x_o) ** 2).sum(dim=-1)


def test_ensemble_map_uses_default_x():
    """`map()` must use the default `x` for both the joint and the individual MAP."""
    num_dim = 2
    ensemble_size = 2
    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    posterior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))

    posteriors = [
        TractablePosterior(_GaussianShiftPotential(posterior, prior))
        for _ in range(ensemble_size)
    ]
    ensemble = EnsemblePosterior(posteriors)
    with pytest.raises(ValueError, match="default has not been set"):
        ensemble.map()
    x_o = 2.0 * ones(1, num_dim)
    ensemble.set_default_x(x_o)

    map_kwargs = {
        "num_iter": 100,
        "num_to_optimize": 2,
        "learning_rate": 0.1,
        "init_method": zeros(1, num_dim),
        "show_progress_bars": False,
    }

    log_weights, maps = ensemble.map(individually=True, **map_kwargs)
    assert log_weights.shape == (ensemble_size, 1)
    assert_close(maps, x_o.expand(ensemble_size, -1, -1), atol=0.01, rtol=0)

    # A stale cached MAP is returned unless `force_update` reaches the components.
    stale = -ones(1, num_dim)
    for component in posteriors:
        component._map = stale
    _, maps = ensemble.map(individually=True, **map_kwargs)
    assert_close(maps, stale.expand(ensemble_size, -1, -1))
    _, maps = ensemble.map(individually=True, force_update=True, **map_kwargs)
    assert_close(maps, x_o.expand(ensemble_size, -1, -1), atol=0.01, rtol=0)

    assert_close(ensemble.map(**map_kwargs), x_o, atol=0.01, rtol=0)
    assert_close(ensemble.map(force_update=True, **map_kwargs), x_o, atol=0.01, rtol=0)

    with pytest.raises(ValueError, match=r"supported\. Use `\.set_default_x\(\)`"):
        ensemble.map(x=x_o, individually=True)

    with pytest.raises(ValueError, match=r"supported\. Use `\.set_default_x\(\)`"):
        ensemble.map(x=x_o)


def _posterior_with_x_shape(x_event_shape=None):
    """Return an untrained direct posterior, which declares `x_event_shape`, or for
    None a posterior with a hand-written potential, which declares no shape."""
    gaussian = MultivariateNormal(zeros(2), eye(2))
    if x_event_shape is None:
        return TractablePosterior(_GaussianShiftPotential(gaussian, gaussian))
    estimator = posterior_nn("maf", hidden_features=8, num_transforms=1)(
        gaussian.sample((10,)), torch.randn(10, *x_event_shape)
    )
    return DirectPosterior(estimator, gaussian)


def test_ensemble_takes_the_observation_shape_its_components_declare():
    """Components without a declared shape do not decide it, in either order, and
    components that declare different shapes cannot form an ensemble."""
    unknown, known = _posterior_with_x_shape(), _posterior_with_x_shape((2, 3))
    for components in ([unknown, known], [known, unknown]):
        assert EnsemblePosterior(components)._process_x(zeros(2, 3)).shape == (1, 2, 3)
    with pytest.raises(ValueError, match="disagree on the shape of one observation"):
        EnsemblePosterior([known, _posterior_with_x_shape((2, 4))])


def test_ensemble_set_default_x_keeps_defaults_after_rejection():
    """A rejected `x` leaves every default and cached MAP in place. An accepted one
    replaces the defaults and clears every MAP, the ensemble's included."""
    components = [_posterior_with_x_shape(), _posterior_with_x_shape((1, 2))]
    ensemble = EnsemblePosterior(components)
    posteriors = [ensemble, *components]
    old = zeros(1, 1, 2)
    ensemble.set_default_x(old)
    for posterior in posteriors:
        posterior._map = zeros(1, 2)

    with pytest.raises(ValueError, match="one observation has shape"):
        ensemble.set_default_x(zeros(2, 2))
    for posterior in posteriors:
        assert torch.equal(posterior.default_x, old)
        assert posterior._map is not None

    new = ones(1, 1, 2)
    ensemble.set_default_x(new)
    for posterior in posteriors:
        assert torch.equal(posterior.default_x, new)
        assert posterior._map is None


def test_ensemble_to_keeps_default_x():
    """`to()` rebuilds the ensemble and must keep its default `x`."""
    prior = BoxUniform(-ones(2), ones(2))
    theta = prior.sample((100,))
    x = theta + 0.1 * randn_like(theta)
    posteriors = []
    for _ in range(2):
        trainer = NPE_C(prior, show_progress_bars=False)
        trainer.append_simulations(theta, x).train(max_num_epochs=1)
        posteriors.append(trainer.build_posterior())
    ensemble = EnsemblePosterior(posteriors)
    x_o = ones(1, 2)
    ensemble.set_default_x(x_o)

    ensemble.to("cpu")

    assert_close(ensemble.default_x, x_o)
    assert ensemble.sample((2,)).shape == (2, 2)


def test_ensemble_potential_follows_weight_change():
    """`potential()` must use the weights set after construction."""
    prior = MultivariateNormal(zeros(1), 10 * eye(1))
    components = MultivariateNormal(tensor([[-2.0], [2.0]]), eye(1))
    posteriors = [
        TractablePosterior(PosteriorPotential(MultivariateNormal(mean, eye(1)), prior))
        for mean in components.mean
    ]
    ensemble = EnsemblePosterior(posteriors, weights=[3.0, 1.0])
    ensemble.set_default_x(zeros(1, 1))
    theta = tensor([[-2.0], [2.0]])
    # Replaces `potential_fn` with a bound copy, which must follow the change too.
    ensemble.potential(theta)

    ensemble.weights = [1.0, 3.0]

    expected = MixtureSameFamily(Categorical(tensor([1.0, 3.0])), components)
    assert_close(ensemble.potential(theta), expected.log_prob(theta))


class _BatchedTractablePosterior(TractablePosterior):
    def sample_batched(self, sample_shape, x, **kwargs):
        return self.potential_fn.posterior.sample((*sample_shape, x.shape[0]))


@pytest.mark.parametrize("method", ["sample", "sample_batched"])
def test_ensemble_samples_are_not_sorted_by_component(method: str):
    """Every row of a multi-dimensional `sample_shape` must mix the components."""
    prior = MultivariateNormal(zeros(1), 10 * eye(1))
    posteriors = [
        _BatchedTractablePosterior(
            PosteriorPotential(MultivariateNormal(tensor([mean]), 0.01 * eye(1)), prior)
        )
        for mean in [-2.0, 2.0]
    ]
    ensemble = EnsemblePosterior(posteriors)

    samples = getattr(ensemble, method)((10, 100), x=zeros(1, 1))

    assert ((samples < 0).any(dim=1) & (samples > 0).any(dim=1)).all()


class _TimedOutTractablePosterior(TractablePosterior):
    """Returns 3 samples, like a sampler that timed out with partial results."""

    def sample(self, sample_shape, x=None, **kwargs):
        return self.potential_fn.posterior.sample((3,))

    def sample_batched(self, sample_shape, x, **kwargs):
        return self.potential_fn.posterior.sample((3, x.shape[0]))


@pytest.mark.parametrize("method", ["sample", "sample_batched"])
def test_ensemble_raises_for_partial_samples(method: str):
    """Partial samples, e.g., after a timeout, would not follow the weights."""
    prior = MultivariateNormal(zeros(1), eye(1))
    potential = PosteriorPotential(MultivariateNormal(zeros(1), eye(1)), prior)
    ensemble = EnsemblePosterior([_TimedOutTractablePosterior(potential)] * 2)

    with pytest.raises(ValueError, match="of 100 samples"):
        getattr(ensemble, method)((10, 10), x=zeros(1, 1))


def test_ensemble_weights_length_and_value_validation():
    """`EnsemblePosterior` must raise `ValueError` for mismatched weight count,
    negative weights, or zero sum.
    """
    prior = MultivariateNormal(zeros(2), eye(2))
    posteriors = [
        TractablePosterior(
            PosteriorPotential(MultivariateNormal(zeros(2), eye(2)), prior)
        )
        for _ in range(2)
    ]
    with pytest.raises(ValueError, match="does not match number of posteriors"):
        EnsemblePosterior(posteriors, weights=[0.2, 0.3, 0.5])

    with pytest.raises(ValueError, match="does not match number of posteriors"):
        EnsemblePosterior(posteriors, weights=[1.0])

    with pytest.raises(ValueError, match="Weights must be 1D"):
        EnsemblePosterior(posteriors, weights=tensor(1.0))

    with pytest.raises(ValueError, match="Weights must be 1D"):
        EnsemblePosterior(posteriors, weights=ones(2, 2))

    with pytest.raises(ValueError, match="must be non-negative and sum to a positive"):
        EnsemblePosterior(posteriors, weights=[-1.0, 2.0])

    with pytest.raises(ValueError, match="must be non-negative and sum to a positive"):
        EnsemblePosterior(posteriors, weights=[0.0, 0.0])

    # Preserve float64 precision through validation and normalization
    weights_small = torch.tensor([1e-100, 1e-100], dtype=torch.float64)
    ens_small = EnsemblePosterior(posteriors, weights=weights_small)
    assert torch.allclose(ens_small.weights, torch.tensor([0.5, 0.5]))

    weights_large = torch.tensor([1e200, 1e200], dtype=torch.float64)
    ens_large = EnsemblePosterior(posteriors, weights=weights_large)
    assert torch.allclose(ens_large.weights, torch.tensor([0.5, 0.5]))


class _MultiDimEventPosterior(TractablePosterior):
    def __init__(self, event_shape=(2, 3)):
        self.event_shape = event_shape
        prior = MultivariateNormal(zeros(6), eye(6))
        super().__init__(PosteriorPotential(prior, prior))

    def sample(self, sample_shape=torch.Size(), x=None, **kwargs):
        sample_shape = torch.Size(sample_shape)
        return zeros(*sample_shape, *self.event_shape)

    def sample_batched(self, sample_shape=torch.Size(), x=None, **kwargs):
        sample_shape = torch.Size(sample_shape)
        num_x = x.shape[0] if x is not None else 1
        return zeros(*sample_shape, num_x, *self.event_shape)


@pytest.mark.parametrize("method", ["sample", "sample_batched"])
def test_ensemble_zero_samples(method: str):
    """Zero-sample requests must return an empty tensor with the correct shape
    without calling multinomial sampling.
    """
    prior = MultivariateNormal(zeros(2), eye(2))
    posterior_cls = (
        TractablePosterior if method == "sample" else _BatchedTractablePosterior
    )
    posteriors = [
        posterior_cls(PosteriorPotential(MultivariateNormal(zeros(2), eye(2)), prior))
        for _ in range(2)
    ]
    ensemble = EnsemblePosterior(posteriors)
    ensemble.set_default_x(zeros(1, 2))

    if method == "sample":
        samples_zero = ensemble.sample((0,))
        assert samples_zero.shape == (0, 2)
        samples_zero_2d = ensemble.sample((0, 5))
        assert samples_zero_2d.shape == (0, 5, 2)
    else:
        samples_zero = ensemble.sample_batched((0,), x=zeros(3, 2))
        assert samples_zero.shape == (0, 3, 2)
        samples_zero_2d = ensemble.sample_batched((0, 4), x=zeros(3, 2))
        assert samples_zero_2d.shape == (0, 4, 3, 2)


@pytest.mark.parametrize("method", ["sample", "sample_batched"])
def test_ensemble_zero_samples_direct_posterior(method: str):
    """Zero-sample requests with DirectPosterior components (which use rejection
    sampling) must return an empty tensor cleanly without crashing.
    """
    gaussian = MultivariateNormal(zeros(2), eye(2))
    estimator = posterior_nn("maf", hidden_features=8, num_transforms=1)(
        gaussian.sample((10,)), torch.randn(10, 2)
    )
    posteriors = [DirectPosterior(estimator, gaussian) for _ in range(2)]
    ensemble = EnsemblePosterior(posteriors)
    ensemble.set_default_x(zeros(1, 2))

    if method == "sample":
        samples_zero = ensemble.sample((0,))
        assert samples_zero.shape == (0, 2)
        samples_zero_2d = ensemble.sample((0, 5))
        assert samples_zero_2d.shape == (0, 5, 2)
    else:
        samples_zero = ensemble.sample_batched((0,), x=zeros(3, 2))
        assert samples_zero.shape == (0, 3, 2)
        samples_zero_2d = ensemble.sample_batched((0, 4), x=zeros(3, 2))
        assert samples_zero_2d.shape == (0, 4, 3, 2)


def test_ensemble_sample_preserves_multidimensional_event_shape():
    """`sample()` must preserve multi-dimensional parameter event shapes without
    flattening them to a single dimension.
    """
    posteriors = [_MultiDimEventPosterior((2, 3)), _MultiDimEventPosterior((2, 3))]
    ensemble = EnsemblePosterior(posteriors)
    ensemble.set_default_x(zeros(1, 2))

    samples_single = ensemble.sample(())
    assert samples_single.shape == (2, 3)

    samples_1d = ensemble.sample((5,))
    assert samples_1d.shape == (5, 2, 3)

    samples_2d = ensemble.sample((4, 7))
    assert samples_2d.shape == (4, 7, 2, 3)

    samples_batched = ensemble.sample_batched((5,), x=zeros(3, 2))
    assert samples_batched.shape == (5, 3, 2, 3)


@pytest.mark.parametrize("method", ["sample", "sample_batched"])
@pytest.mark.parametrize("event_shape", [(4,), (2, 3)])
def test_ensemble_zero_samples_prior_free_fallback(method: str, event_shape: tuple):
    """Zero-sample requests on an ensemble without a prior where components raise
    must derive the event shape from the components rather than hardcoding [1],
    supporting both 1D and multi-axis event shapes.
    """

    class _FailingPriorFreePosterior(TractablePosterior):
        def __init__(self, event_shape=(4,)):
            self.event_shape = event_shape
            prod_dim = int(torch.prod(torch.tensor(event_shape)))
            super().__init__(
                PosteriorPotential(
                    MultivariateNormal(zeros(prod_dim), eye(prod_dim)), prior=None
                )
            )

        def sample(self, sample_shape=torch.Size(), x=None, **kwargs):
            raise RuntimeError("Cannot sample")

        def sample_batched(self, sample_shape=torch.Size(), x=None, **kwargs):
            raise RuntimeError("Cannot sample batched")

    posteriors = [
        _FailingPriorFreePosterior(event_shape),
        _FailingPriorFreePosterior(event_shape),
    ]
    ensemble = EnsemblePosterior(posteriors)
    ensemble.set_default_x(zeros(1, 2))
    assert ensemble.prior is None

    if method == "sample":
        samples = ensemble.sample((0,))
        assert samples.shape == (0, *event_shape)
        samples_2d = ensemble.sample((0, 3))
        assert samples_2d.shape == (0, 3, *event_shape)
    else:
        samples = ensemble.sample_batched((0,), x=zeros(5, 2))
        assert samples.shape == (0, 5, *event_shape)
        samples_2d = ensemble.sample_batched((0, 2), x=zeros(5, 2))
        assert samples_2d.shape == (0, 2, 5, *event_shape)


@pytest.mark.parametrize("method", ["sample", "sample_batched"])
def test_ensemble_zero_samples_mcmc_restricted_prior(method: str):
    """Zero-sample requests on an ensemble of prior-free MCMCPosteriors with
    RestrictedPrior proposal must derive the event shape from the wrapped prior.
    """
    from sbi.inference.potentials.base_potential import BasePotential
    from sbi.utils.restriction_estimator import RestrictedPrior

    class _DummyPotential(BasePotential):
        def __init__(self):
            super().__init__(prior=None, x_o=None)

        def __call__(self, theta, track_gradients=True):
            return -0.5 * (theta**2).sum(-1)

    prior = MultivariateNormal(zeros(3), eye(3))
    rp = RestrictedPrior(
        prior=prior,
        accept_reject_fn=lambda th: torch.ones(len(th), dtype=torch.bool),
    )
    # Spy on proposal sampling to verify zero-sample calls do not draw proposal samples
    proposal_sample_calls = []
    orig_rp_sample = rp.sample

    def spy_rp_sample(*args, **kwargs):
        proposal_sample_calls.append((args, kwargs))
        return orig_rp_sample(*args, **kwargs)

    rp.sample = spy_rp_sample

    posteriors = [MCMCPosterior(_DummyPotential(), proposal=rp) for _ in range(2)]
    ensemble = EnsemblePosterior(posteriors)
    ensemble.set_default_x(zeros(1, 3))
    assert ensemble.prior is None

    if method == "sample":
        samples = ensemble.sample((0,))
        assert samples.shape == (0, 3)
        samples_2d = ensemble.sample((0, 4))
        assert samples_2d.shape == (0, 4, 3)
    else:
        samples = ensemble.sample_batched((0,), x=zeros(5, 3))
        assert samples.shape == (0, 5, 3)
        samples_2d = ensemble.sample_batched((0, 2), x=zeros(5, 3))
        assert samples_2d.shape == (0, 2, 5, 3)

    # Verify that zero-sample requests do not initialize samplers or
    # draw proposal candidates
    assert len(proposal_sample_calls) == 0
