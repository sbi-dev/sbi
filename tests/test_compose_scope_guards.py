"""Scope-guard and boundary tests for opt-in ``compose_standardization``.

These cover the patch lines that the build/checkpoint tests in
``test_compose_standardization_internal.py`` and the baseline guards in
``test_compose_baseline_guard.py`` do not reach:

  * the ``NotImplementedError`` scope guards for iid / guided sampling, iid
    ``log_prob``, MAP, and the potential ``set_x`` chokepoint (single-obs is the
    only supported path under composition);
  * the ``from_z`` sample-output boundary on both the ODE (FMPE) and SDE (NPSE)
    paths, i.e. samples are returned in original theta-space, not z-space;
  * the score-estimator ``loss`` composition hook (``theta -> z`` at the top);
  * the compose-OFF ``else`` branch of the single-obs ``log_prob`` potential.

All tests are training-free and fast: the guards raise up front, and the two
sampling tests run an untrained estimator for a handful of steps (the numbers
are meaningless, only the coordinate space and finiteness are asserted).
"""

import pytest
import torch
from torch.distributions import Independent, Normal

from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior
from sbi.inference.potentials.vector_field_potential import VectorFieldBasedPotential
from sbi.neural_nets.net_builders.vector_field_nets import (
    build_vector_field_estimator,
)

NUM_DIM = 2


def _batches():
    torch.manual_seed(0)
    # Non-unit theta stats: the compose shift is ~100, so from_z-corrected samples
    # sit far from the z-space origin (used by the boundary assertions below).
    batch_x = 100.0 + 5.0 * torch.randn(64, NUM_DIM)
    batch_y = torch.randn(64, NUM_DIM)
    return batch_x, batch_y


def _wide_prior():
    return Independent(Normal(torch.zeros(NUM_DIM), 200.0 * torch.ones(NUM_DIM)), 1)


def _compose_estimator(estimator_type="flow", **kwargs):
    batch_x, batch_y = _batches()
    return build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type=estimator_type,
        z_score_x="independent",
        compose_standardization=True,
        **kwargs,
    )


def _compose_posterior(estimator_type="flow", sample_with="ode", **kwargs):
    est = _compose_estimator(estimator_type=estimator_type, **kwargs)
    return VectorFieldPosterior(
        vector_field_estimator=est, prior=_wide_prior(), sample_with=sample_with
    )


# --------------------------------------------------------------------------
# Scope guards — raise NotImplementedError rather than return wrong results
# --------------------------------------------------------------------------


def test_sample_iid_rejects_compose():
    """sample() with batched (iid) x under compose -> NotImplementedError."""
    posterior = _compose_posterior()
    x_iid = torch.randn(3, NUM_DIM)  # batch > 1 => iid
    with pytest.raises(NotImplementedError, match="iid"):
        posterior.sample((2,), x=x_iid, show_progress_bars=False)


def test_sample_guidance_rejects_compose():
    """sample() with a guidance method under compose -> NotImplementedError."""
    posterior = _compose_posterior()
    x_o = torch.zeros(1, NUM_DIM)
    with pytest.raises(NotImplementedError, match="guided"):
        posterior.sample(
            (2,), x=x_o, guidance_method="classifier_free", show_progress_bars=False
        )


def test_log_prob_iid_rejects_compose():
    """log_prob() with batched (iid) x under compose -> NotImplementedError."""
    posterior = _compose_posterior()
    x_iid = torch.randn(3, NUM_DIM)
    with pytest.raises(NotImplementedError, match="iid"):
        posterior.log_prob(torch.zeros(1, NUM_DIM), x=x_iid)


def test_map_rejects_compose():
    """map() under compose -> NotImplementedError (guard is the first statement,
    so it fires before the default-x check)."""
    posterior = _compose_posterior()
    with pytest.raises(NotImplementedError, match="MAP"):
        posterior.map(show_progress_bars=False)


def test_potential_set_x_iid_rejects_compose():
    """Direct potential use: set_x(x_is_iid=True) under compose -> NotImplementedError
    (the chokepoint guard, reachable without going through the posterior)."""
    est = _compose_estimator()
    potential = VectorFieldBasedPotential(
        est, prior=_wide_prior(), x_o=None, device="cpu"
    )
    with pytest.raises(NotImplementedError, match="iid"):
        potential.set_x(torch.randn(2, NUM_DIM), x_is_iid=True)


def test_potential_set_x_guidance_rejects_compose():
    """Direct potential use: set_x(guidance_method=...) under compose ->
    NotImplementedError."""
    est = _compose_estimator()
    potential = VectorFieldBasedPotential(
        est, prior=_wide_prior(), x_o=None, device="cpu"
    )
    with pytest.raises(NotImplementedError, match="guided"):
        potential.set_x(
            torch.zeros(1, NUM_DIM), x_is_iid=False, guidance_method="classifier_free"
        )


# --------------------------------------------------------------------------
# from_z sample-output boundary — samples returned in ORIGINAL theta-space
# --------------------------------------------------------------------------


def test_sample_ode_returns_theta_space_under_compose():
    """FMPE/ODE single-obs sample under compose applies from_z: output lives in
    original theta-space (shift ~100), not z-space. Untrained -> only coordinate
    space + finiteness asserted."""
    posterior = _compose_posterior(estimator_type="flow", sample_with="ode")
    samples = posterior.sample(
        (5,),
        x=torch.zeros(1, NUM_DIM),
        reject_outside_prior=False,
        show_progress_bars=False,
    )
    assert samples.shape == (5, NUM_DIM)
    assert torch.isfinite(samples).all()
    # from_z applied => samples carry the ~100 shift; z-space samples would be O(1).
    assert samples.abs().mean() > 10.0


def test_sample_sde_returns_theta_space_under_compose():
    """NPSE/SDE single-obs sample under compose applies from_z on the diffusion
    path: output lives in original theta-space."""
    posterior = _compose_posterior(
        estimator_type="score", sample_with="sde", sde_type="ve"
    )
    samples = posterior.sample(
        (5,),
        x=torch.zeros(1, NUM_DIM),
        reject_outside_prior=False,
        show_progress_bars=False,
    )
    assert samples.shape == (5, NUM_DIM)
    assert torch.isfinite(samples).all()
    assert samples.abs().mean() > 10.0


# --------------------------------------------------------------------------
# score-estimator loss composition hook (theta -> z at the top of loss)
# --------------------------------------------------------------------------


def test_score_estimator_loss_compose_hook():
    """Building a compose NPSE and calling loss() exercises the score-estimator
    composition hook (unit-stats check + theta->z standardization of the input)."""
    est = _compose_estimator(estimator_type="score", sde_type="ve")
    theta = 100.0 + 5.0 * torch.randn(16, NUM_DIM)
    x = torch.randn(16, NUM_DIM)
    loss = est.loss(theta, x)
    assert torch.isfinite(loss).all()


# --------------------------------------------------------------------------
# compose-OFF else branch of the single-obs log_prob potential
# --------------------------------------------------------------------------


def test_single_obs_log_prob_compose_off_else_branch():
    """compose OFF: the potential takes the identity (else) branch of the
    single-obs log_prob path (no to_z, no Jacobian). Runs and returns finite."""
    batch_x, batch_y = _batches()
    est = build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=False,
    )
    prior = _wide_prior()
    potential = VectorFieldBasedPotential(est, prior=prior, x_o=None, device="cpu")
    potential.set_x(torch.zeros(1, NUM_DIM), x_is_iid=False)  # builds flow, no training
    out = potential(torch.zeros(1, NUM_DIM))
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# R4 backstop — directly-forced iid under compose raises in __call__
# --------------------------------------------------------------------------


def test_iid_log_prob_compose_backstop_raises_on_direct_force():
    """R4 backstop: forcing ``_x_is_iid=True`` under compose (bypassing the set_x
    guard) makes the potential ``__call__`` raise ``NotImplementedError`` instead of
    returning a partially-incorrect log-prob (affine Jacobian applied, prior not yet
    in z). This is exactly the branch the set_x guard cannot structurally reach."""
    est = _compose_estimator()
    potential = VectorFieldBasedPotential(
        est, prior=_wide_prior(), x_o=None, device="cpu"
    )
    # Reach the iid+compose branch WITHOUT going through set_x (which guards it):
    potential._x_o = torch.randn(3, NUM_DIM)
    potential._x_is_iid = True
    potential.flows = potential.rebuild_flows_for_batch()
    with pytest.raises(NotImplementedError, match="iid"):
        potential(torch.zeros(1, NUM_DIM))
