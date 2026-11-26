# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pickle

import pytest
import torch

from sbi.inference import DirectPosterior, VectorFieldPosterior
from sbi.inference.posteriors.npe_a_posterior import NPE_A_Posterior
from sbi.neural_nets import posterior_flow_nn, posterior_nn, posterior_score_nn
from sbi.samplers.rejection.rejection import count_accepted
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


@pytest.mark.parametrize("estimator_type", ["direct", "flow", "score"])
def test_log_prob_is_normalized_inside_bounded_prior(estimator_type):
    """`exp(log_prob)` must integrate to one over a box that cuts off most mass."""
    prior = BoxUniform(torch.zeros(2), 3 * torch.ones(2))
    posterior = _posterior(estimator_type, prior).set_default_x(torch.zeros(1, 2))

    num_cells = 60
    midpoints = (torch.arange(num_cells) + 0.5) * 3 / num_cells
    grid = torch.cartesian_prod(midpoints, midpoints)
    cell_area = (3 / num_cells) ** 2
    integral = cell_area * posterior.log_prob(grid).exp().sum()
    uncorrected = cell_area * posterior.log_prob(grid, norm_posterior=False).exp().sum()

    assert uncorrected < 0.7, "The box must truncate the density for this test."
    assert abs(integral - 1.0) < 0.06


def _shifted_uniform_sample(sample_shape, condition):
    """Draws uniform in a box of half-width 1.5 around each condition, so the mass
    inside the prior box [0, 3]^2 is known."""
    noise = 2 * torch.rand(*sample_shape, condition.shape[0], 2) - 1
    return condition + 1.5 * noise


def test_batched_normalizer_equals_one_observation_at_a_time(monkeypatch):
    """Each observation stops on its own, and gets 0 only for itself."""
    prior = BoxUniform(torch.zeros(2), 3 * torch.ones(2))
    posterior = _posterior("direct", prior)
    monkeypatch.setattr(
        posterior.posterior_estimator, "sample", _shifted_uniform_sample
    )
    x = torch.tensor([[1.5, 1.5], [0.0, 1.5], [0.0, 0.0], [30.0, 30.0]])
    theta = torch.ones(4, 2)
    params = {"max_num_draws": 100_000}

    with pytest.warns(UserWarning, match="1 of 4 observation"):
        batched = posterior.log_prob_batched(
            theta, x=x, leakage_correction_params=params
        ).reshape(-1)
    with pytest.warns(UserWarning, match="1 of 1 observation"):
        single = torch.cat([
            posterior.log_prob(
                theta[i : i + 1], x=x[i : i + 1], leakage_correction_params=params
            )
            for i in range(4)
        ])
    unnormalized = posterior.log_prob_batched(theta, x=x, norm_posterior=False)

    expected = unnormalized.reshape(-1)[:3] - torch.tensor([1.0, 0.5, 0.25]).log()
    assert torch.allclose(batched[:3], expected, atol=0.03)
    assert torch.allclose(single[:3], expected, atol=0.03)
    assert batched[3].isnan() and single[3].isnan()


def test_zero_mass_gives_nan_inside_and_minus_inf_outside():
    """Without any draw inside the prior, normalizing would divide by 0."""
    prior = BoxUniform(20 * torch.ones(2), 21 * torch.ones(2))
    posterior = _posterior("flow", prior).set_default_x(torch.zeros(1, 2))
    theta = torch.tensor([[20.5, 20.5], [0.0, 0.0]])
    params = {"num_rejection_samples": 300, "max_num_draws": 300}

    with pytest.warns(UserWarning, match="below about 1.0e-02"):
        log_prob = posterior.log_prob(theta, leakage_correction_params=params)
        assert posterior.leakage_correction(posterior.default_x, **params) == 0

    assert log_prob[0].isnan() and log_prob[1] == float("-inf")


def test_count_accepted_stops_at_target_or_cap_within_batch_size():
    """Each observation stops at its target of accepted draws or at the cap, and no
    call of the proposal draws more than the batch size."""
    draws_per_call = []

    def proposal(num_draws, indices):
        draws_per_call.append(num_draws * len(indices))
        return indices.expand(num_draws, -1)

    accepted, drawn = count_accepted(
        proposal,
        lambda theta: theta == 0,
        num_xos=2,
        num_accepted=100,
        max_num_draws=250,
        max_sampling_batch_size=40,
    )

    assert accepted.tolist() == [100, 0] and drawn.tolist() == [100, 250]
    assert max(draws_per_call) <= 40


@pytest.mark.parametrize("estimator_type", ["flow", "score"])
def test_vector_field_leakage_correction_and_map_take_one_observation(
    estimator_type,
):
    """Several observations raise: the leakage factor would add up their masses above
    one, and `map()` binds its ODE flow to one observation."""
    prior = BoxUniform(torch.zeros(2), 3 * torch.ones(2))
    posterior = _posterior(estimator_type, prior)
    with pytest.raises(ValueError, match="takes one observation"):
        posterior.leakage_correction(x=torch.zeros(3, 2), num_rejection_samples=10)
    with pytest.raises(ValueError, match="`map\\(\\)` of a vector field posterior"):
        posterior.set_default_x(torch.zeros(3, 2)).map(num_iter=1)


@pytest.mark.parametrize("estimator_type", ["direct", "npe_a", "flow"])
def test_leakage_correction_caching(estimator_type, monkeypatch):
    """The factor is saved for the last `x` and number of samples only, and samples at
    the given `x` (and for vector fields, with the given `ode_kwargs`)."""
    prior = BoxUniform(torch.zeros(2), 3 * torch.ones(2))
    posterior = _posterior(estimator_type, prior)
    is_vf = isinstance(posterior, VectorFieldPosterior)
    owner = posterior if is_vf else posterior.posterior_estimator
    sampler = "sample_via_ode" if is_vf else "sample"
    calls = []
    sample = getattr(owner, sampler)

    def spy(*args, **kwargs):
        x = posterior.potential_fn.x_o if is_vf else kwargs["condition"]
        calls.append((kwargs, x))
        return sample(*args, **kwargs)

    monkeypatch.setattr(owner, sampler, spy)
    theta = torch.ones(1, 2)
    params = {"num_rejection_samples": 100}
    x_a, x_b = torch.zeros(1, 2), torch.ones(1, 2)

    def estimates(params=params, **kwargs) -> int:
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

    assert estimates({"num_rejection_samples": 50}) > 0, (
        "Another number of samples must not reuse the saved factor."
    )
    assert estimates({"num_rejection_samples": 50, "max_num_draws": 60}) > 0, (
        "Another limit of draws must not reuse the saved factor."
    )
    posterior.log_prob(theta, x=2 * x_b, leakage_correction_params=params)
    posterior.leakage_correction(posterior.default_x, force_update=True, **params)
    assert torch.equal(calls[-1][1], posterior.default_x)
    num_calls = len(calls)
    posterior.leakage_correction(posterior.default_x.double(), **params)
    assert len(calls) == num_calls, "A float64 `x` must find the saved factor."


def test_leakage_cache_matches_nan_padded_x():
    """NaN padding in `x` (e.g., for varying trial counts) must not defeat the cache."""
    prior = BoxUniform(torch.zeros(2), 3 * torch.ones(2))
    posterior = _posterior("direct", prior)
    calls = []
    x = torch.tensor([[0.0, float("nan")]])
    ones = torch.ones(1)
    for _ in range(2):
        posterior._cached_leakage_factor(
            x.clone(),
            100,
            prior,
            lambda: calls.append(1) or (ones, ones),
        )
    assert len(calls) == 1


def test_leakage_cache_pickled_without_num_samples():
    """Posteriors pickled before the cache saved the number of samples still load."""
    prior = BoxUniform(torch.zeros(2), 3 * torch.ones(2))
    posterior = _posterior("direct", prior).set_default_x(torch.zeros(1, 2))
    posterior._leakage_cache = (torch.zeros(1, 2), torch.ones(()))

    loaded = pickle.loads(pickle.dumps(posterior))
    assert torch.isfinite(loaded.log_prob(torch.ones(1, 2))).all()


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
