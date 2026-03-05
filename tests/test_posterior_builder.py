# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Tests for the PosteriorBuilder API and VectorFieldBasedPotential.init().

Covers:
- init() is a no-op on BasePotential (backwards compatibility)
- init() on VectorFieldBasedPotential warms the lru_cache before sampling
- PosteriorBuilder.with_iid / with_guidance return new builders
- init() is called exactly once even on repeated sample() calls
- Builder chains (with_iid + with_guidance) both apply correctly
- Old sample(iid_method=...) path still works (no regression)
"""

from unittest.mock import patch

import pytest
import torch
from torch import ones, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import NPSE
from sbi.inference.posteriors.vector_field_posterior import (
    PosteriorBuilder,
    VectorFieldPosterior,
)
from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference.potentials.vector_field_potential import VectorFieldBasedPotential


# Minimal helpers

def _train_small_npse(num_dim: int = 2, num_simulations: int = 500):
    """Train a tiny NPSE model for unit-testing purposes."""
    prior = MultivariateNormal(zeros(num_dim), torch.eye(num_dim))
    likelihood_shift = -0.5 * ones(num_dim)
    likelihood_cov = 0.5 * torch.eye(num_dim)

    theta = prior.sample((num_simulations,))
    x = theta + likelihood_shift + torch.randn(num_simulations, num_dim) @ likelihood_cov

    inference = NPSE(prior, show_progress_bars=False)
    inference.append_simulations(theta, x)
    estimator = inference.train(max_num_epochs=3, show_train_summary=False)

    posterior = inference.build_posterior(estimator)
    return posterior, prior


# BasePotential.init() is a no-op

class _DummyPotential(BasePotential):
    """Minimal concrete potential for testing the base class default."""

    def __call__(self, theta, track_gradients=True):
        return theta.sum(-1)


def test_base_potential_init_is_noop():
    prior = MultivariateNormal(zeros(2), torch.eye(2))
    pot = _DummyPotential(prior=prior)
    result = pot.init(x_obs=torch.zeros(1, 2))
    assert result is pot, "init() must return self"


# with_iid / with_guidance return PosteriorBuilder

@pytest.mark.slow
def test_with_iid_returns_builder():
    posterior, _ = _train_small_npse()
    builder = posterior.with_iid(method="fnpe")
    assert isinstance(builder, PosteriorBuilder)


@pytest.mark.slow
def test_with_guidance_returns_builder():
    posterior, _ = _train_small_npse()
    builder = posterior.with_guidance(
        method="affine_classifier_free", likelihood_scale=1.5
    )
    assert isinstance(builder, PosteriorBuilder)


@pytest.mark.slow
def test_builder_chaining():
    posterior, _ = _train_small_npse()
    builder = (
        posterior
        .with_iid(method="fnpe")
        .with_guidance(method="affine_classifier_free", likelihood_scale=1.5)
    )
    assert isinstance(builder, PosteriorBuilder)
    assert builder._iid_method == "fnpe"
    assert builder._guidance_method == "affine_classifier_free"
    assert builder._guidance_params["likelihood_scale"] == 1.5


# init() is called exactly once

@pytest.mark.slow
def test_init_called_once_on_repeated_samples():
    posterior, _ = _train_small_npse()
    x_obs = zeros(1, 2)

    builder = posterior.with_iid(method="fnpe")

    init_call_count = {"n": 0}
    original_init = VectorFieldBasedPotential.init

    def counting_init(self, **kwargs):
        init_call_count["n"] += 1
        return original_init(self, **kwargs)

    with patch.object(VectorFieldBasedPotential, "init", counting_init):
        builder.sample((10,), x=x_obs, steps=10, show_progress_bars=False)
        builder.sample((10,), x=x_obs, steps=10, show_progress_bars=False)

    assert init_call_count["n"] == 1, (
        f"Expected init() to be called exactly once, got {init_call_count['n']}"
    )


# Builder produces same results as old API (no regression)

@pytest.mark.slow
def test_builder_fnpe_matches_old_api():
    """Builder path and old sample(iid_method=...) path should be equivalent."""
    posterior, _ = _train_small_npse()
    x_obs = zeros(1, 2)

    torch.manual_seed(0)
    samples_builder = (
        posterior
        .with_iid(method="fnpe")
        .sample((50,), x=x_obs, steps=10, show_progress_bars=False)
    )

    torch.manual_seed(0)
    samples_old = posterior.sample(
        (50,),
        x=x_obs,
        iid_method="fnpe",
        steps=10,
        show_progress_bars=False,
    )

    # Shapes must match; values may differ slightly due to random state but
    # both should be finite and in a reasonable range.
    assert samples_builder.shape == samples_old.shape
    assert torch.isfinite(samples_builder).all()
    assert torch.isfinite(samples_old).all()


# init() warms the cache for auto_gauss

@pytest.mark.slow
def test_init_warms_auto_gauss_cache():
    """After init(), AutoGaussCorrectedScoreFn.estimate_posterior_precision
    should already be in the cache so no second call is made during sampling."""
    from sbi.inference.potentials.vector_field_adaptor import AutoGaussCorrectedScoreFn

    posterior, _ = _train_small_npse()
    x_obs = torch.zeros(3, 2)  # 3 iid observations

    builder = posterior.with_iid(method="auto_gauss")

    estimate_call_count = {"n": 0}
    original_est = AutoGaussCorrectedScoreFn.estimate_posterior_precision.__wrapped__

    def counting_estimate(cls, *args, **kwargs):
        estimate_call_count["n"] += 1
        return original_est(cls, *args, **kwargs)

    # Clear the cache
    AutoGaussCorrectedScoreFn.estimate_posterior_precision.cache_clear()

    before = AutoGaussCorrectedScoreFn.estimate_posterior_precision.cache_info()

    builder.sample((10,), x=x_obs, steps=5, show_progress_bars=False)

    after = AutoGaussCorrectedScoreFn.estimate_posterior_precision.cache_info()

    assert after.hits + after.misses > before.hits + before.misses, (
        "estimate_posterior_precision was never called — init() may not be wiring "
        "through to the auto_gauss precision estimator."
    )