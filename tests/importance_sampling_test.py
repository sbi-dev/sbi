# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch.distributions import Independent, Normal, Uniform

from sbi.inference.posteriors.importance_posterior import ImportanceSamplingPosterior
from sbi.samplers.importance.importance_sampling import (
    exponentiate_weights,
    importance_sample,
    largest_weight_indices,
)
from sbi.samplers.importance.sir import sampling_importance_resampling


def quadratic_potential(
    theta: torch.Tensor, x: torch.Tensor | None = None
) -> torch.Tensor:
    if theta.ndim == 1:
        return -0.5 * theta.pow(2)
    return -0.5 * theta.pow(2).sum(dim=-1)


def test_importance_sample_multidim_proposal_without_independent():
    """Verify that multi-dimensional proposals whose log_prob returns (N, D)
    do not crash.
    """
    proposal = Normal(torch.zeros(3), torch.ones(3))
    samples, log_weights = importance_sample(
        quadratic_potential, proposal, num_samples=10
    )
    assert samples.shape == (10, 3)
    assert log_weights.shape == (10,)

    proposal_uniform = Uniform(torch.zeros(4), torch.ones(4))
    samples_u, log_weights_u = importance_sample(
        quadratic_potential, proposal_uniform, num_samples=8
    )
    assert samples_u.shape == (8, 4)
    assert log_weights_u.shape == (8,)


def test_importance_sample_multidim_proposal_with_independent():
    """Verify that wrapped Independent distributions continue to work as expected."""
    proposal = Independent(Normal(torch.zeros(3), torch.ones(3)), 1)
    samples, log_weights = importance_sample(
        quadratic_potential, proposal, num_samples=10
    )
    assert samples.shape == (10, 3)
    assert log_weights.shape == (10,)


def test_importance_sample_1d():
    """Verify 1D proposal behavior."""
    proposal = Normal(0.0, 1.0)
    samples, log_weights = importance_sample(
        quadratic_potential, proposal, num_samples=7
    )
    assert samples.shape[0] == 7
    assert log_weights.shape == (7,)


def test_importance_sample_zero_samples():
    """Verify that requesting 0 samples returns empty tensors without error."""
    proposal = Normal(torch.zeros(3), torch.ones(3))
    samples, log_weights = importance_sample(
        quadratic_potential, proposal, num_samples=0
    )
    assert samples.shape == (0, 3)
    assert log_weights.shape == (0,)


def test_importance_sample_device_transfer():
    """Verify device alignment when potential_fn specifies a device."""

    class DevicePotential:
        def __init__(self, device: str):
            self.device = torch.device(device)

        def __call__(self, theta: torch.Tensor) -> torch.Tensor:
            assert theta.device == self.device
            return -0.5 * theta.pow(2).sum(dim=-1)

    device_pot = DevicePotential("cpu")
    proposal = Normal(torch.zeros(2), torch.ones(2))
    samples, log_weights = importance_sample(device_pot, proposal, num_samples=5)
    assert samples.device == torch.device("cpu")
    assert log_weights.device == torch.device("cpu")


def test_exponentiate_weights_all_nonfinite_and_empty():
    """Verify exponentiate_weights does not crash when all weights are non-finite."""
    all_neg_inf = torch.tensor([-float("inf"), -float("inf")])
    out = exponentiate_weights(all_neg_inf)
    assert out.numel() == 0

    empty = torch.tensor([])
    out_empty = exponentiate_weights(empty)
    assert out_empty.numel() == 0

    mixed = torch.tensor([0.0, -1.0, -float("inf")])
    out_mixed = exponentiate_weights(mixed)
    assert len(out_mixed) == 2
    assert torch.all(torch.isfinite(out_mixed))


def test_largest_weight_indices_small_length():
    """Verify largest_weight_indices with len < 5 does not return all indices
    via inds[-0:].
    """
    for n in range(5):
        weights = torch.arange(n, dtype=torch.float32)
        inds = largest_weight_indices(weights)
        assert len(inds) == 0, f"Expected 0 indices for length {n}, got {len(inds)}"

    weights_10 = torch.arange(10, dtype=torch.float32)
    inds_10 = largest_weight_indices(weights_10)
    assert len(inds_10) > 0
    assert torch.all(inds_10 >= 0)


def test_sampling_importance_resampling_zero_samples():
    """Verify SIR returns empty samples with correct event shape for num_samples=0."""
    proposal = Normal(torch.zeros(3), torch.ones(3))
    samples = sampling_importance_resampling(
        quadratic_potential, proposal, num_samples=0
    )
    assert samples.shape == (0, 3)


def test_sampling_importance_resampling_multidim():
    """Verify SIR draws samples correctly from multi-dimensional unwrapped proposals."""
    proposal = Normal(torch.zeros(3), torch.ones(3))
    samples = sampling_importance_resampling(
        quadratic_potential, proposal, num_samples=15, num_candidate_samples=8
    )
    assert samples.shape == (15, 3)


@pytest.mark.parametrize("method", ["sir", "importance"])
def test_importance_sampling_posterior_zero_samples(method):
    """Verify ImportanceSamplingPosterior handles sample_shape=(0,) cleanly."""
    proposal = Normal(torch.zeros(3), torch.ones(3))
    post = ImportanceSamplingPosterior(quadratic_potential, proposal=proposal)
    post.set_default_x(torch.zeros(1))

    if method == "sir":
        samples = post.sample((0,), method=method)
        assert samples.shape == (0, 3)
    else:
        samples, log_weights = post.sample((0,), method=method)
        assert samples.shape == (0, 3)
        assert log_weights.shape == (0,)


@pytest.mark.parametrize("method", ["sir", "importance"])
def test_importance_sampling_posterior_shapes(method):
    """Verify ImportanceSamplingPosterior sample shapes: (), (5,), and (2, 3)."""
    proposal = Normal(torch.zeros(3), torch.ones(3))
    post = ImportanceSamplingPosterior(quadratic_potential, proposal=proposal)
    post.set_default_x(torch.zeros(1))

    # Single sample: ()
    if method == "sir":
        s_single = post.sample((), method=method)
        assert s_single.shape == (3,)
    else:
        s_single, w_single = post.sample((), method=method)
        assert s_single.shape == (3,)
        assert w_single.shape == ()

    # 1D batch: (5,)
    if method == "sir":
        s_1d = post.sample((5,), method=method)
        assert s_1d.shape == (5, 3)
    else:
        s_1d, w_1d = post.sample((5,), method=method)
        assert s_1d.shape == (5, 3)
        assert w_1d.shape == (5,)

    # 2D batch: (2, 4)
    if method == "sir":
        s_2d = post.sample((2, 4), method=method)
        assert s_2d.shape == (2, 4, 3)
    else:
        s_2d, w_2d = post.sample((2, 4), method=method)
        assert s_2d.shape == (2, 4, 3)
        assert w_2d.shape == (2, 4)
