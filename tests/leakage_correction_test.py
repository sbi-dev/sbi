# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch

from sbi.inference import DirectPosterior, VectorFieldPosterior
from sbi.inference.posteriors.npe_a_posterior import NPE_A_Posterior
from sbi.neural_nets import posterior_flow_nn, posterior_nn, posterior_score_nn
from sbi.utils import BoxUniform
from sbi.utils.torchutils import process_device
from tests.test_utils import mps_fallback_disabled


def _posterior(estimator_type: str, prior, device: str = "cpu"):
    # The ODE density of any vector field integrates to one on R^d, so the
    # estimator does not need training.
    theta = torch.randn(200, 2)
    x = theta + 0.1 * torch.randn_like(theta)
    if estimator_type in ("flow", "score"):
        build = posterior_flow_nn if estimator_type == "flow" else posterior_score_nn
        estimator = build(hidden_features=16, num_layers=2)(theta, x).to(device)
        return VectorFieldPosterior(estimator, prior, device=device)
    estimator = posterior_nn("mdn")(theta, x).to(device)
    posterior_class = {"direct": DirectPosterior, "npe_a": NPE_A_Posterior}
    return posterior_class[estimator_type](estimator, prior, device=device)


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


@pytest.mark.parametrize("estimator_type", ["direct", "npe_a", "flow"])
def test_leakage_correction_caching(estimator_type, monkeypatch):
    """The factor is saved for the last `x` only, and samples at the given `x` (and
    for vector fields, with the given `ode_kwargs`)."""
    prior = BoxUniform(torch.zeros(2), 3 * torch.ones(2))
    posterior = _posterior(estimator_type, prior)
    is_vf = isinstance(posterior, VectorFieldPosterior)
    sampler = "sample_via_ode" if is_vf else "_sample_estimator"
    calls = []
    sample = getattr(posterior, sampler)

    def spy(*args, **kwargs):
        x = posterior.potential_fn.x_o if is_vf else kwargs["condition"]
        calls.append((kwargs, x))
        return sample(*args, **kwargs)

    monkeypatch.setattr(posterior, sampler, spy)
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

    if is_vf:
        ode_kwargs = {"atol": 1e-4, "rtol": 1e-4}
        assert estimates(ode_kwargs=ode_kwargs) > 0 and calls[-1][0] == ode_kwargs
        assert estimates() == 0, "A call with `ode_kwargs` must not replace the cache."

    posterior.log_prob(theta, x=2 * x_b, leakage_correction_params=params)
    posterior.leakage_correction(posterior.default_x, force_update=True, **params)
    assert torch.equal(calls[-1][1], posterior.default_x)


def test_leakage_cache_matches_nan_padded_x():
    """NaN padding in `x` (e.g., for varying trial counts) must not defeat the cache."""
    prior = BoxUniform(torch.zeros(2), 3 * torch.ones(2))
    posterior = _posterior("direct", prior)
    calls = []
    x = torch.tensor([[0.0, float("nan")]])
    for _ in range(2):
        posterior._cached_leakage_factor(
            x.clone(), prior, lambda: calls.append(1) or torch.ones(())
        )
    assert len(calls) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("estimator_type", ["direct", "npe_a", "flow"])
def test_explicit_cpu_x_for_posterior_on_gpu(estimator_type):
    device = process_device("gpu")
    if mps_fallback_disabled(device):
        pytest.skip("Needs PYTORCH_ENABLE_MPS_FALLBACK=1 on MPS.")
    prior = BoxUniform(torch.zeros(2, device=device), 3 * torch.ones(2, device=device))
    posterior = _posterior(estimator_type, prior, device)
    theta, x = torch.ones(1, 2, device=device), torch.ones(1, 2)
    params = {"num_rejection_samples": 100}

    posterior.log_prob(theta, x=x, norm_posterior=False)
    posterior.log_prob(theta, x=x, leakage_correction_params=params)
    posterior.sample((2,), x=x, show_progress_bars=False)
    if isinstance(posterior, DirectPosterior):
        posterior.log_prob_batched(theta, x=x, leakage_correction_params=params)
        posterior.sample_batched((2,), x=x, show_progress_bars=False)


@pytest.mark.gpu
@pytest.mark.parametrize("estimator_type", ["direct", "npe_a", "flow"])
def test_explicit_cpu_theta_for_posterior_on_gpu(estimator_type):
    device = process_device("gpu")
    if mps_fallback_disabled(device):
        pytest.skip("Needs PYTORCH_ENABLE_MPS_FALLBACK=1 on MPS.")
    prior = BoxUniform(torch.zeros(2, device=device), 3 * torch.ones(2, device=device))
    posterior = _posterior(estimator_type, prior, device)
    theta, x = torch.ones(1, 2), torch.ones(1, 2)
    params = {"num_rejection_samples": 100}

    lp_unnorm = posterior.log_prob(theta, x=x, norm_posterior=False)
    assert lp_unnorm.device.type == torch.device(device).type
    lp_norm = posterior.log_prob(theta, x=x, leakage_correction_params=params)
    assert lp_norm.device.type == torch.device(device).type

    expected = posterior.log_prob(
        theta.to(device), x=x.to(device), norm_posterior=False
    )
    assert torch.allclose(lp_unnorm, expected)

    if isinstance(posterior, DirectPosterior):
        lpb = posterior.log_prob_batched(theta, x=x, leakage_correction_params=params)
        assert lpb.device.type == torch.device(device).type

        theta_grad = torch.ones(1, 2, requires_grad=True)
        lp_grad = posterior.log_prob(
            theta_grad, x=x, track_gradients=True, norm_posterior=False
        )
        assert lp_grad.requires_grad


@pytest.mark.gpu
def test_direct_posterior_log_prob_array_and_list_on_gpu():
    import numpy as np

    device = process_device("gpu")
    if mps_fallback_disabled(device):
        pytest.skip("Needs PYTORCH_ENABLE_MPS_FALLBACK=1 on MPS.")
    prior = BoxUniform(torch.zeros(2, device=device), 3 * torch.ones(2, device=device))
    posterior = _posterior("direct", prior, device)

    theta_np = np.ones((1, 2), dtype=np.float32)
    x_np = np.ones((1, 2), dtype=np.float32)
    lp_np = posterior.log_prob(theta_np, x=x_np, norm_posterior=False)
    assert lp_np.device.type == torch.device(device).type
    lpb_np = posterior.log_prob_batched(theta_np, x=x_np, norm_posterior=False)
    assert lpb_np.device.type == torch.device(device).type

    theta_list = [[1.0, 1.0]]
    x_list = [[1.0, 1.0]]
    lp_list = posterior.log_prob(theta_list, x=x_list, norm_posterior=False)
    assert lp_list.device.type == torch.device(device).type
