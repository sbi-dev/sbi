# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch

from sbi.inference import VectorFieldPosterior
from sbi.neural_nets.net_builders.vector_field_nets import build_vector_field_estimator
from sbi.utils import BoxUniform


def _posterior(estimator_type: str, prior) -> VectorFieldPosterior:
    # The ODE density of any vector field integrates to one on R^d, so the
    # estimator does not need training.
    theta = torch.randn(200, 2)
    estimator = build_vector_field_estimator(
        batch_x=theta,
        batch_y=theta + 0.1 * torch.randn_like(theta),
        estimator_type=estimator_type,
        hidden_features=16,
        num_layers=2,
    )
    return VectorFieldPosterior(estimator, prior)


@pytest.mark.parametrize("estimator_type", ["flow", "score"])
def test_log_prob_is_normalized_inside_bounded_prior(estimator_type):
    """`exp(log_prob)` must integrate to one over a box that cuts off most mass."""
    prior = BoxUniform(torch.zeros(2), 3 * torch.ones(2))
    posterior = _posterior(estimator_type, prior).set_default_x(torch.zeros(1, 2))

    theta = prior.sample((20_000,))
    volume = 9.0
    integral = volume * posterior.log_prob(theta).exp().mean()
    uncorrected = volume * posterior.log_prob(theta, norm_posterior=False).exp().mean()

    assert uncorrected < 0.7, "The box must truncate the density for this test."
    assert abs(integral - 1.0) < 0.1


def test_leakage_correction_caching(monkeypatch):
    """The factor is cached at the default `x` only, and uses the `ode_kwargs`."""
    prior = BoxUniform(torch.zeros(2), 3 * torch.ones(2))
    posterior = _posterior("flow", prior)
    calls = []
    sample_via_ode = posterior.sample_via_ode
    monkeypatch.setattr(
        posterior,
        "sample_via_ode",
        lambda *args, **kwargs: calls.append(kwargs) or sample_via_ode(*args, **kwargs),
    )
    theta = torch.ones(1, 2)
    params = {"num_rejection_samples": 100}

    posterior.set_default_x(torch.zeros(1, 2))
    posterior.log_prob(theta, leakage_correction_params=params)
    num_calls = len(calls)
    posterior.log_prob(theta, leakage_correction_params=params)
    assert len(calls) == num_calls, "The factor at the default x must be cached."

    posterior.set_default_x(torch.ones(1, 2))
    posterior.log_prob(theta, leakage_correction_params=params)
    assert len(calls) > num_calls

    num_calls = len(calls)
    ode_kwargs = {"atol": 1e-4, "rtol": 1e-4}
    posterior.log_prob(theta, ode_kwargs=ode_kwargs, leakage_correction_params=params)
    assert len(calls) > num_calls and calls[-1] == ode_kwargs
