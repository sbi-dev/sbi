"""Tier-1 hardening unit tests for the opt-in ``compose_standardization`` feature.

All tests here are training-free (build / wire only) and fast:

  * T1.1 single-source theta-standardization: building an FMPE / NPSE-ve with
    ``compose_standardization=True`` zeroes the internal input-norm stats
    (mean_0=0/std_0=1) and stores the per-dim theta affine in
    ``_theta_shift``/``_theta_scale``; with compose OFF the affine is identity.
  * T1.5 dtype/device roundtrip: ``from_z(to_z(theta)) == theta`` in float32 and
    float64, and ``log_abs_det() == sum(log scale)``.

The build-time / runtime gaussian_baseline guards live in
``tests/test_compose_baseline_guard.py``.
"""

import pytest
import torch

from sbi.neural_nets.net_builders.vector_field_nets import (
    _compute_theta_standardization,
    build_vector_field_estimator,
)
from sbi.utils.sbiutils import z_standardization

NUM_DIM = 3


def _batches():
    torch.manual_seed(0)
    # Non-unit theta stats on purpose: exercises the build path that zeroes them.
    batch_x = 100.0 + 5.0 * torch.randn(32, NUM_DIM)
    batch_y = torch.randn(32, NUM_DIM)
    return batch_x, batch_y


# --------------------------------------------------------------------------
# T1.1 — single-source theta-standardization helper
# --------------------------------------------------------------------------


def test_helper_compose_off_zscore():
    """compose OFF + z_score on -> z-score stats, no compose affine."""
    batch_x, _ = _batches()
    mean_0, std_0, shift, scale = _compute_theta_standardization(
        batch_x, z_score_x="independent", compose_standardization=False
    )
    exp_mean, exp_std = z_standardization(batch_x, structured_dims=False)
    assert torch.allclose(mean_0, exp_mean)
    assert torch.allclose(std_0, exp_std)
    assert shift is None and scale is None


def test_helper_compose_off_no_zscore():
    """compose OFF + z_score off -> identity stats (0/1), no compose affine."""
    batch_x, _ = _batches()
    mean_0, std_0, shift, scale = _compute_theta_standardization(
        batch_x, z_score_x="none", compose_standardization=False
    )
    assert mean_0 == 0 and std_0 == 1
    assert shift is None and scale is None


def test_helper_compose_on_invariant():
    """compose ON -> internal stats unit, compose affine = per-dim z stats."""
    batch_x, _ = _batches()
    mean_0, std_0, shift, scale = _compute_theta_standardization(
        batch_x, z_score_x="independent", compose_standardization=True
    )
    exp_mean, exp_std = z_standardization(batch_x, structured_dims=False)
    assert (mean_0 == 0).all() and (std_0 == 1).all()
    assert torch.allclose(shift, exp_mean)
    assert torch.allclose(scale, exp_std.clamp_min(1e-20))


@pytest.mark.parametrize(
    "estimator_type,kwargs",
    [
        ("flow", {}),
        ("score", {"sde_type": "ve"}),
    ],
)
def test_build_compose_on_unit_stats_and_affine(estimator_type, kwargs):
    """End-to-end build (no training): compose ON => unit internal stats and
    _theta_shift/_theta_scale match per-dim z_standardization of the batch."""
    batch_x, batch_y = _batches()
    est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type=estimator_type,
        z_score_x="independent",
        compose_standardization=True,
        **kwargs,
    )
    exp_mean, exp_std = z_standardization(batch_x, structured_dims=False)
    assert bool(est._compose_standardization) is True
    assert torch.allclose(est.mean_0, torch.zeros_like(est.mean_0))
    assert torch.allclose(est.std_0, torch.ones_like(est.std_0))
    assert torch.allclose(est._theta_shift.reshape(-1), exp_mean.float(), atol=1e-5)
    assert torch.allclose(
        est._theta_scale.reshape(-1), exp_std.clamp_min(1e-20).float(), atol=1e-5
    )


@pytest.mark.parametrize(
    "estimator_type,kwargs",
    [
        ("flow", {}),
        ("score", {"sde_type": "ve"}),
    ],
)
def test_build_compose_off_identity(estimator_type, kwargs):
    """compose OFF => flag False, identity theta affine (byte-identical OFF path)."""
    batch_x, batch_y = _batches()
    est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type=estimator_type,
        z_score_x="independent",
        compose_standardization=False,
        **kwargs,
    )
    assert bool(est._compose_standardization) is False
    assert torch.allclose(est._theta_shift, torch.zeros_like(est._theta_shift))
    assert torch.allclose(est._theta_scale, torch.ones_like(est._theta_scale))


# --------------------------------------------------------------------------
# T1.5 — dtype/device roundtrip of the boundary affine
# --------------------------------------------------------------------------


def _compose_estimator():
    batch_x, batch_y = _batches()
    return build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_affine_roundtrip_dtype(dtype):
    """from_z(to_z(theta)) == theta in float32 and float64."""
    est = _compose_estimator()
    if dtype == torch.float64:
        est = est.double()
    theta = (100.0 + 5.0 * torch.randn(10, NUM_DIM)).to(dtype)
    recon = est.from_z(est.to_z(theta))
    assert recon.dtype == dtype
    assert torch.allclose(recon, theta, atol=1e-5 if dtype == torch.float32 else 1e-10)


def test_log_abs_det_equals_sum_log_scale():
    est = _compose_estimator()
    assert torch.allclose(est.log_abs_det(), torch.log(est._theta_scale).sum())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_affine_roundtrip_cuda():
    est = _compose_estimator().cuda()
    theta = (100.0 + 5.0 * torch.randn(10, NUM_DIM)).cuda()
    recon = est.from_z(est.to_z(theta))
    assert recon.is_cuda
    assert torch.allclose(recon, theta, atol=1e-5)


# --------------------------------------------------------------------------
# T1.3 — partial-checkpoint guard in base._load_from_state_dict
# --------------------------------------------------------------------------

_COMPOSE_KEYS = ("_theta_shift", "_theta_scale", "_compose_standardization")


def test_checkpoint_legacy_no_compose_keys_loads_as_off():
    """All 3 compose buffers absent (legacy pre-compose checkpoint) -> loads fine
    as compose-OFF with identity affine."""
    batch_x, batch_y = _batches()
    src = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=False,
    )
    sd = src.state_dict()
    for k in _COMPOSE_KEYS:
        sd.pop(k, None)

    dst = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=False,
    )
    dst.load_state_dict(sd)  # must not raise
    assert bool(dst._compose_standardization) is False
    assert torch.allclose(dst._theta_shift, torch.zeros_like(dst._theta_shift))
    assert torch.allclose(dst._theta_scale, torch.ones_like(dst._theta_scale))


def test_checkpoint_partial_compose_keys_raises():
    """compose=True checkpoint MISSING _theta_scale -> raise on load (no silent
    identity injection)."""
    batch_x, batch_y = _batches()
    src = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=True,
    )
    sd = src.state_dict()
    sd.pop("_theta_scale", None)  # partial: shift + flag present, scale missing

    dst = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=False,
    )
    with pytest.raises((RuntimeError, ValueError), match="_theta_scale"):
        dst.load_state_dict(sd)


def test_checkpoint_all_compose_keys_loads():
    """All 3 compose buffers present (compose checkpoint) -> normal load."""
    batch_x, batch_y = _batches()
    src = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=True,
    )
    sd = src.state_dict()
    dst = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=False,
    )
    dst.load_state_dict(sd)  # must not raise
    assert bool(dst._compose_standardization) is True
    assert torch.allclose(dst._theta_shift, src._theta_shift)
    assert torch.allclose(dst._theta_scale, src._theta_scale)


# --------------------------------------------------------------------------
# T1.4 — sample_batched compose guard
# --------------------------------------------------------------------------


def test_sample_batched_rejects_compose():
    """sample_batched is untested/unsupported under compose (used by the batched
    score correction in theta-space) -> raise NotImplementedError up front."""
    from torch.distributions import Independent, Normal

    from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior

    batch_x, batch_y = _batches()
    est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=True,
    )
    prior = Independent(Normal(torch.zeros(NUM_DIM), torch.ones(NUM_DIM)), 1)
    posterior = VectorFieldPosterior(vector_field_estimator=est, prior=prior)

    x_batch = torch.randn(2, NUM_DIM)  # batch of 2 observations
    with pytest.raises(NotImplementedError, match="compose_standardization"):
        posterior.sample_batched(torch.Size([3]), x=x_batch)


# --------------------------------------------------------------------------
# Nit 1 — compose-OFF preservation for structured z-scoring
# --------------------------------------------------------------------------


def test_helper_compose_off_structured_zscore():
    """compose OFF + z_score structured -> scalar/structured stats match
    z_standardization(batch, structured_dims=True). The structured branch is
    not covered by the other T1.1 helper tests, which use independent/none."""
    batch_x, _ = _batches()
    mean_0, std_0, shift, scale = _compute_theta_standardization(
        batch_x, z_score_x="structured", compose_standardization=False
    )
    exp_mean, exp_std = z_standardization(batch_x, structured_dims=True)
    assert torch.allclose(torch.as_tensor(mean_0), exp_mean)
    assert torch.allclose(torch.as_tensor(std_0), exp_std)
    assert shift is None and scale is None


# --------------------------------------------------------------------------
# Nit 2 — nested/prefixed checkpoint load exercises prefix + name path
# --------------------------------------------------------------------------


class _WrappedEstimator(torch.nn.Module):
    """Minimal wrapper that places the estimator at a non-root prefix."""

    def __init__(self, est):
        super().__init__()
        self.est = est


def test_checkpoint_prefixed_legacy_loads_as_off():
    """Wrapped estimator (prefix='est.'): all 3 compose buffers stripped ->
    loads as compose-OFF with identity affine (exercises prefix + name path)."""
    batch_x, batch_y = _batches()
    src_est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=False,
    )
    src = _WrappedEstimator(src_est)
    sd = src.state_dict()
    _compose_keys = ("_theta_shift", "_theta_scale", "_compose_standardization")
    for k in list(sd.keys()):
        if any(k.endswith(n) for n in _compose_keys):
            del sd[k]

    dst_est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=False,
    )
    dst = _WrappedEstimator(dst_est)
    dst.load_state_dict(sd)  # must not raise
    assert bool(dst.est._compose_standardization) is False
    assert torch.allclose(dst.est._theta_shift, torch.zeros_like(dst.est._theta_shift))
    assert torch.allclose(dst.est._theta_scale, torch.ones_like(dst.est._theta_scale))


def test_checkpoint_prefixed_partial_raises():
    """Wrapped estimator (prefix='est.'): compose flag present, _theta_scale
    stripped -> raises naming the missing key (exercises prefix + name path)."""
    batch_x, batch_y = _batches()
    src_est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=True,
    )
    src = _WrappedEstimator(src_est)
    sd = src.state_dict()
    # Remove only the scale; keep shift + flag (partial).
    for k in list(sd.keys()):
        if k.endswith("_theta_scale"):
            del sd[k]

    dst_est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=False,
    )
    dst = _WrappedEstimator(dst_est)
    with pytest.raises((RuntimeError, ValueError), match="_theta_scale"):
        dst.load_state_dict(sd)


# --------------------------------------------------------------------------
# Nit 3 — sharper log_abs_det test with hand-set known scale
# --------------------------------------------------------------------------


def test_log_abs_det_known_scale():
    """Non-tautological: hand-set _theta_scale to [2.0, 3.0, 5.0] and assert
    log_abs_det() == log(2)+log(3)+log(5) exactly."""
    batch_x, batch_y = _batches()
    est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=True,
    )
    known_scale = torch.tensor([[[2.0, 3.0, 5.0]]])  # shape (1, 1, NUM_DIM)
    est._theta_scale.copy_(known_scale.reshape(1, NUM_DIM))
    expected = torch.log(torch.tensor([2.0, 3.0, 5.0])).sum()
    assert torch.allclose(est.log_abs_det(), expected)


# --------------------------------------------------------------------------
# Deterministic single-obs log_prob Jacobian correction (no training)
# --------------------------------------------------------------------------


def test_single_obs_log_prob_jacobian_exact():
    """Deterministic (training-free) pin of the affine-Jacobian log_prob
    correction in the single-obs branch of ``VectorFieldBasedPotential.__call__``.

    Wire a tiny compose estimator into the potential, set a single observation
    (builds the neural ODE -- no training), set ``_theta_scale`` to a known
    [2.0, 3.0], and monkeypatch the underlying ``flow.log_prob`` to return a known
    constant. This exercises the REAL ``log_abs_det()`` and the REAL subtraction
    ``log_probs = log_probs - log_abs_det`` in the code path (not a
    reimplementation), so the returned potential must equal
    ``known_constant - log(2) - log(3)`` to tight tolerance.
    """
    from torch.distributions import Independent, Normal

    from sbi.inference.potentials.vector_field_potential import (
        VectorFieldBasedPotential,
    )

    torch.manual_seed(0)
    dim = 2
    batch_x = 100.0 + 5.0 * torch.randn(32, dim)
    batch_y = torch.randn(32, dim)
    est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=True,
    )
    # Wide prior so theta=0 lies in support (within_support uses original theta).
    prior = Independent(Normal(torch.zeros(dim), 100.0 * torch.ones(dim)), 1)
    potential = VectorFieldBasedPotential(est, prior=prior, x_o=None, device="cpu")
    potential.set_x(torch.zeros(1, dim), x_is_iid=False)  # builds flow, no training

    # Known per-dim scale -> log_abs_det = log(2) + log(3).
    est._theta_scale.copy_(torch.tensor([[2.0, 3.0]]))

    known_constant = 7.0
    # Monkeypatch the z-space flow log_prob used in the single-obs branch.
    potential.flow.log_prob = lambda z: torch.full(z.shape[:-1], known_constant)

    out = potential(torch.zeros(1, dim))
    expected = known_constant - torch.log(torch.tensor([2.0, 3.0])).sum()
    assert torch.allclose(out.reshape(-1), expected.reshape(-1), atol=1e-6), (
        f"log_prob Jacobian correction off: got {out.item()}, expected "
        f"{expected.item()}"
    )
