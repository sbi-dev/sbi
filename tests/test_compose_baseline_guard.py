"""Guard tests for the ``gaussian_baseline`` + ``compose_standardization`` pair.

``gaussian_baseline`` (PR #1752) derives the network velocity from the
original-space data statistics ``mean_0``/``std_0``. Composed standardization
(#1680) instead maps ``theta -> z`` so the network sees unit-scale inputs. The
two are mutually exclusive; the maintainer flagged the combination as untested.

Two guards make the behavior defined instead of silently inconsistent:
  * BUILD-TIME (``_wire_compose``): the public API (``build_vector_field_estimator``
    / ``posterior_flow_nn``) refuses the pair up front.
  * RUNTIME (``FlowMatchingEstimator._check_compose_baseline_compatible``): backstop
    that rejects the pair on loss()/forward() for estimators assembled manually, or
    loaded from a checkpoint predating the build-time guard, where composition is
    re-activated alongside ``gaussian_baseline`` (regardless of the stored stats).

These are fast unit tests (no training); the equivariance behavior itself lives
in ``tests/test_scale_equivariance.py``.
"""

import pytest
import torch

from sbi.neural_nets.estimators.flowmatching_estimator import FlowMatchingEstimator
from sbi.neural_nets.net_builders.vector_field_nets import (
    build_vector_field_estimator,
)

NUM_DIM = 2
BATCH = 8


class _ZeroNet(torch.nn.Module):
    """Minimal VectorFieldNet stub: returns zeros, trainable param keeps autograd."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input, condition, time):
        return torch.zeros_like(input) * self.dummy


def _batches():
    torch.manual_seed(0)
    # Non-unit theta stats on purpose: exercises the build path that zeroes them.
    batch_x = 100.0 + 5.0 * torch.randn(16, NUM_DIM)
    batch_y = torch.randn(16, NUM_DIM)
    return batch_x, batch_y


def _manual_estimator(mean_0, std_0, gaussian_baseline, compose):
    """Assemble a FlowMatchingEstimator directly and (optionally) flip the private
    composition buffers on — mimicking misuse that bypasses the build guard."""
    est = FlowMatchingEstimator(
        net=_ZeroNet(),
        input_shape=torch.Size([NUM_DIM]),
        condition_shape=torch.Size([NUM_DIM]),
        mean_0=mean_0,
        std_0=std_0,
        gaussian_baseline=gaussian_baseline,
    )
    if compose:
        est._theta_shift.copy_(torch.zeros(1, NUM_DIM))
        est._theta_scale.copy_(torch.ones(1, NUM_DIM))
        est._compose_standardization.fill_(True)
        est._compose_enabled = True  # sync plain-Python mirror (as _wire_compose does)
    return est


# --------------------------------------------------------------------------
# Guard II — build-time, public API
# --------------------------------------------------------------------------


def test_build_rejects_compose_plus_baseline():
    """Public build path refuses compose_standardization + gaussian_baseline."""
    batch_x, batch_y = _batches()
    with pytest.raises(
        ValueError, match="gaussian_baseline and compose_standardization"
    ):
        build_vector_field_estimator(
            batch_x,
            batch_y,
            estimator_type="flow",
            z_score_x="independent",
            gaussian_baseline=True,
            compose_standardization=True,
        )


def test_build_compose_without_baseline_ok():
    """Normal compose path (gaussian_baseline=False) builds and enables compose;
    build forces mean_0=0/std_0=1 (unit z-stats)."""
    batch_x, batch_y = _batches()
    est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        gaussian_baseline=False,
        compose_standardization=True,
    )
    assert bool(est._compose_standardization)
    assert not est.gaussian_baseline
    assert torch.allclose(est.mean_0, torch.zeros_like(est.mean_0))
    assert torch.allclose(est.std_0, torch.ones_like(est.std_0))


def test_build_baseline_without_compose_ok():
    """Normal baseline path (compose off) builds; compose flag stays False."""
    batch_x, batch_y = _batches()
    est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        gaussian_baseline=True,
        compose_standardization=False,
    )
    assert est.gaussian_baseline
    assert not bool(est._compose_standardization)


# --------------------------------------------------------------------------
# Guard I — runtime, manual assembly that bypasses the build guard
# --------------------------------------------------------------------------


def _inputs():
    torch.manual_seed(1)
    inp = torch.randn(BATCH, NUM_DIM)
    cond = torch.randn(BATCH, NUM_DIM)
    t = torch.rand(BATCH)
    return inp, cond, t


def test_runtime_rejects_manual_compose_baseline_nonunit_stats():
    """Manual estimator: compose + baseline + NON-unit mean_0/std_0 -> raise on
    both loss() and forward()."""
    est = _manual_estimator(
        mean_0=torch.tensor([100.0, 100.0]),
        std_0=torch.tensor([5.0, 5.0]),
        gaussian_baseline=True,
        compose=True,
    )
    inp, cond, t = _inputs()
    with pytest.raises(ValueError, match="cannot be used together"):
        est.loss(inp, cond, t)
    with pytest.raises(ValueError, match="cannot be used together"):
        est.forward(inp, cond, t)


def test_runtime_rejects_compose_baseline_unit_stats():
    """Policy: the pair is rejected ENTIRELY, not only when stats are non-unit.

    The build path forces mean_0=0/std_0=1 under composition, so a checkpoint
    produced before the build-time guard existed (or a manual assembly) carries
    unit stats. The runtime guard must still reject it on loss() and forward(),
    consistent with the build-time guard that refuses the pair up front."""
    est = _manual_estimator(
        mean_0=torch.zeros(NUM_DIM),
        std_0=torch.ones(NUM_DIM),
        gaussian_baseline=True,
        compose=True,
    )
    inp, cond, t = _inputs()
    with pytest.raises(ValueError, match="cannot be used together"):
        est.loss(inp, cond, t)
    with pytest.raises(ValueError, match="cannot be used together"):
        est.forward(inp, cond, t)


def test_runtime_rejects_compose_with_nonunit_internal_stats():
    """T1.2 invariant: compose ON requires unit internal stats (mean_0=0/std_0=1).

    The build path forces unit z-stats under composition; a manual assembly (or a
    tampered checkpoint) with compose ON but NON-unit mean_0/std_0 is internally
    inconsistent (the network would not see unit-z input) and must RAISE on both
    loss() and forward()."""
    est = _manual_estimator(
        mean_0=torch.tensor([100.0, 100.0]),
        std_0=torch.tensor([5.0, 5.0]),
        gaussian_baseline=False,
        compose=True,
    )
    inp, cond, t = _inputs()
    with pytest.raises(ValueError, match="unit"):
        est.loss(inp, cond, t)
    with pytest.raises(ValueError, match="unit"):
        est.forward(inp, cond, t)


def test_runtime_allows_compose_with_unit_internal_stats():
    """No over-fire: compose ON with unit internal stats (mean_0=0/std_0=1) runs
    fine on both loss() and forward()."""
    est = _manual_estimator(
        mean_0=torch.zeros(NUM_DIM),
        std_0=torch.ones(NUM_DIM),
        gaussian_baseline=False,
        compose=True,
    )
    inp, cond, t = _inputs()
    loss = est.loss(inp, cond, t)
    v = est.forward(inp, cond, t)
    assert torch.isfinite(loss).all()
    assert v.shape == (BATCH, NUM_DIM)
    assert torch.isfinite(v).all()


def test_runtime_allows_baseline_without_compose():
    """Guard does not over-fire: baseline ON, compose OFF, non-unit stats -> runs
    (plain gaussian_baseline FMPE, no composition)."""
    est = _manual_estimator(
        mean_0=torch.tensor([100.0, 100.0]),
        std_0=torch.tensor([5.0, 5.0]),
        gaussian_baseline=True,
        compose=False,
    )
    inp, cond, t = _inputs()
    loss = est.loss(inp, cond, t)
    v = est.forward(inp, cond, t)
    assert torch.isfinite(loss).all()
    assert v.shape == (BATCH, NUM_DIM)
    assert torch.isfinite(v).all()
