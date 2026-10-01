# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch

from sbi.inference import VectorFieldPosterior
from sbi.neural_nets import posterior_flow_nn, posterior_score_nn
from sbi.utils import BoxUniform


def _posterior(estimator_type: str, prior) -> VectorFieldPosterior:
    # The ODE density of any vector field integrates to one on R^d, so the
    # estimator does not need training.
    theta = torch.randn(200, 2)
    build = posterior_flow_nn if estimator_type == "flow" else posterior_score_nn
    estimator = build(hidden_features=16, num_layers=2)(
        theta, theta + 0.1 * torch.randn_like(theta)
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
    """The factor is saved for the last `x` only, and samples at the given `x` with
    the given `ode_kwargs`."""
    prior = BoxUniform(torch.zeros(2), 3 * torch.ones(2))
    posterior = _posterior("flow", prior)
    calls = []
    sample_via_ode = posterior.sample_via_ode
    monkeypatch.setattr(
        posterior,
        "sample_via_ode",
        lambda *args, **kwargs: (
            calls.append((kwargs, posterior.potential_fn.x_o))
            or sample_via_ode(*args, **kwargs)
        ),
    )
    theta = torch.ones(1, 2)
    params = {"num_rejection_samples": 100}
    x_a, x_b = torch.zeros(1, 2), torch.ones(1, 2)

    def estimates(**kwargs) -> int:
        num_calls = len(calls)
        posterior.log_prob(theta, leakage_correction_params=params, **kwargs)
        return len(calls) - num_calls

    assert estimates(x=x_a) > 0
    assert estimates(x=x_a.clone()) == 0, "The factor for the last x must be saved."
    assert estimates(x=x_b) > 0, "A different x must never reuse the saved factor."
    assert torch.equal(calls[-1][1], x_b)
    assert estimates(x=x_a) > 0

    posterior.set_default_x(x_b)
    assert estimates() > 0
    assert estimates() == 0

    ode_kwargs = {"atol": 1e-4, "rtol": 1e-4}
    assert estimates(ode_kwargs=ode_kwargs) > 0 and calls[-1][0] == ode_kwargs
    assert estimates() == 0, "A call with `ode_kwargs` must not replace the saved one."

    posterior.log_prob(theta, x=2 * x_b, leakage_correction_params=params)
    posterior.leakage_correction(posterior.default_x, force_update=True, **params)
    assert torch.equal(calls[-1][1], posterior.default_x)
