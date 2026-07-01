"""Regression tests for scale-equivariance of FMPE / NPSE posteriors (#1680).

Companion to ``tests/linearGaussian_vector_field_test.py``. A calibrated amortized
posterior must be invariant under an exact reparameterization ``theta -> c * theta``
(the simulator unscales, so the x-distribution is identical). FMPE and NPSE default
posteriors are NOT equivariant when parameters are far from O(1) scale (the flow/SDE
integrate against a unit base), so calibration collapses despite the default
``z_score_theta="independent"``.

The fix validated here is the OPT-IN ``compose_standardization`` flag: the estimator
trains and samples in standardized z-space with an invertible PER-DIM affine
``theta = shift + scale * z`` composed at the boundaries (loss input standardized;
samples unstandardized; log_prob corrected by the affine Jacobian). Per-dim handles
HETEROGENEOUS scales (mixed O(1) and O(1e-6) dims).

Scope: single observation. iid / guided sampling are guarded (NotImplementedError)
and not tested here. Runtime: marked slow.
"""

import contextlib

import numpy as np
import pytest
import torch
from torch.distributions import Independent, Normal

from sbi.inference import FMPE, NPSE
from sbi.neural_nets.factory import posterior_flow_nn, posterior_score_nn
from sbi.utils import BoxUniform

N_TRAIN, N_POST, MAX_EPOCHS = 2000, 2000, 150


def _linear_gaussian(s, seed=0):
    """Homogeneous linear-Gaussian: theta ~ N(0, s^2 I_2), x = sum(theta) + s * noise.
    Posterior coupling rho = -0.5, tightness ~0.816 (both scale-free)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    prior = Independent(Normal(torch.zeros(2), s * torch.ones(2)), 1)
    theta = prior.sample((N_TRAIN,))
    x = theta.sum(1, keepdim=True) + s * torch.randn(N_TRAIN, 1)
    return prior, theta, x


def _heterogeneous(seed=0):
    """Mixed-scale prior: two O(1) dims and two O(1e-6) dims (cfg62-like). A scalar
    noise-schedule rescale cannot bracket both scales; per-dim composed
    standardization can. The simulator's second output pins (theta2 - theta3) at the
    small scale, so at the observation x=[*, 0] the posterior must place
    theta2 ~ theta3 -> STRONG POSITIVE coupling of the small-scale block."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    scale = torch.tensor([1.0, 1.0, 1e-6, 1e-6])
    prior = Independent(Normal(torch.zeros(4), scale), 1)
    theta = prior.sample((N_TRAIN,))
    x = torch.stack(
        [theta[:, 0] + theta[:, 1], (theta[:, 2] - theta[:, 3]) / 1e-6], dim=-1
    ) + 0.01 * torch.randn(N_TRAIN, 2)
    return prior, theta, x, scale


def _fit(kind, prior, theta, x, compose):
    if kind == "FMPE":
        de = posterior_flow_nn(compose_standardization=True) if compose else "mlp"
        tr = FMPE(prior=prior, vf_estimator=de)
    else:
        de = (
            posterior_score_nn(sde_type="ve", compose_standardization=True)
            if compose
            else "mlp"
        )
        tr = NPSE(prior=prior, sde_type="ve", vf_estimator=de)
    tr.append_simulations(theta, x).train(
        max_num_epochs=MAX_EPOCHS, show_train_summary=False
    )
    return tr.build_posterior()


def _coupling_tightness(post, s, x_o):
    a = post.sample((N_POST,), x=x_o, show_progress_bars=False).numpy()
    rho = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    tight = float(a[:, 0].std()) / s
    return rho, tight


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["FMPE", "NPSE"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compose_standardization_homogeneous_equivariant(kind, seed):
    """GREEN (multi-seed): composed standardization recovers calibration at small
    scale and matches the unit-scale reference (coupling + tightness), for both
    FMPE and NPSE."""
    prior_u, th_u, x_u = _linear_gaussian(1.0, seed=seed)
    rho_u, _ = _coupling_tightness(
        _fit(kind, prior_u, th_u, x_u, True), 1.0, torch.zeros(1)
    )

    s = 1e-5
    prior_i, th_i, x_i = _linear_gaussian(s, seed=seed)
    rho_i, tight_i = _coupling_tightness(
        _fit(kind, prior_i, th_i, x_i, True), s, torch.zeros(1)
    )
    assert tight_i < 3.0, f"[seed={seed}] ill-scale tightness inflated: {tight_i}"
    assert rho_i < -0.15, f"[seed={seed}] ill-scale coupling collapsed: {rho_i}"
    assert abs(rho_i - rho_u) < 0.3, (
        f"[seed={seed}] ill vs unit mismatch: {rho_i} vs {rho_u}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["FMPE", "NPSE"])
def test_default_collapses_without_compose(kind):
    """RED guard: with the flag OFF (default) the posterior collapses at small scale
    -- documents the bug #1681 left unfixed and that the fix is opt-in (no behavior
    change)."""
    s = 1e-5
    prior_i, th_i, x_i = _linear_gaussian(s)
    rho_i, tight_i = _coupling_tightness(
        _fit(kind, prior_i, th_i, x_i, False), s, torch.zeros(1)
    )
    assert tight_i > 3.0 or rho_i > -0.15, (
        f"expected collapse without compose: rho={rho_i}, tight={tight_i}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["FMPE", "NPSE"])
def test_compose_standardization_heterogeneous_recovers(kind):
    """GREEN + comparator: on a mixed-scale (O(1)+O(1e-6)) problem, composed
    standardization (per-dim) recovers the SIGN-CORRECT strong POSITIVE coupling of
    the small-scale block (x pins theta2 ~ theta3), where the default collapses it
    (rho ~ 0)."""
    prior, theta, x, scale = _heterogeneous()
    x_o = torch.tensor([[0.0, 0.0]])

    a_fix = (
        _fit(kind, prior, theta, x, True)
        .sample((N_POST,), x=x_o, show_progress_bars=False)
        .numpy()
    )
    rho_fix = float(np.corrcoef(a_fix[:, 2], a_fix[:, 3])[0, 1])
    tight_fix = float(a_fix[:, 2].std()) / float(scale[2])

    a_def = (
        _fit(kind, prior, theta, x, False)
        .sample((N_POST,), x=x_o, show_progress_bars=False, reject_outside_prior=False)
        .numpy()
    )
    rho_def = float(np.corrcoef(a_def[:, 2], a_def[:, 3])[0, 1])

    # Recovery: strong POSITIVE coupling (sign-correct), contracted small-scale block.
    assert tight_fix < 5.0, f"small-scale block not contracted: {tight_fix}"
    assert rho_fix > 0.7, f"composed std failed to recover positive coupling: {rho_fix}"
    # Comparator: default leaves the small-scale block far less coupled.
    assert rho_fix - rho_def > 0.3, (
        f"compose did not beat default on coupling: fix={rho_fix} def={rho_def}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["FMPE", "NPSE"])
def test_compose_log_prob_jacobian_scaling(kind):
    """Affine-Jacobian correctness: an equivariant posterior satisfies
    p_s(theta) = s^{-d} p_1(theta/s), so at the (symmetric) mode theta=0,
    log p_s(0) - log p_1(0) = -d * log(s). This checks the log_prob Jacobian
    correction (-sum log scale) is applied with the right magnitude/sign (d=2)."""
    d, s = 2, 1e-3
    prior_u, th_u, x_u = _linear_gaussian(1.0)
    lp_u = float(
        _fit(kind, prior_u, th_u, x_u, True)
        .log_prob(torch.zeros(1, d), x=torch.zeros(1))
        .item()
    )
    prior_s, th_s, x_s = _linear_gaussian(s)
    lp_s = float(
        _fit(kind, prior_s, th_s, x_s, True)
        .log_prob(torch.zeros(1, d), x=torch.zeros(1))
        .item()
    )

    expected = -d * np.log(s)  # ~ +13.8
    assert np.isfinite(lp_u) and np.isfinite(lp_s), (
        f"non-finite log_prob: {lp_u}, {lp_s}"
    )
    assert abs((lp_s - lp_u) - expected) < 4.0, (
        f"Jacobian scaling off: lp_s-lp_u={lp_s - lp_u:.2f} vs expected {expected:.2f}"
    )


class _RecordingSupport:
    """Wraps a distribution's ``support`` and records every tensor passed to
    ``check`` (the call the VFPE rejection path makes via ``within_support``)."""

    def __init__(self, support, sink):
        self._support = support
        self._sink = sink

    def check(self, value):
        self._sink.append(value.detach().clone())
        return self._support.check(value)


class _RecordingPrior:
    """Thin proxy around a prior that records the coordinates handed to the
    prior-support check during rejection sampling. ``within_support`` reaches the
    prior via ``prior.support.check(theta)``; we intercept exactly that so the test
    can assert the recorded coordinate is in THETA space (post ``from_z``), not
    z-space. ``log_prob`` is also recorded as a fallback path."""

    def __init__(self, prior, sink):
        self._prior = prior
        self._sink = sink
        self.support = _RecordingSupport(prior.support, sink)

    def log_prob(self, value):
        self._sink.append(value.detach().clone())
        return self._prior.log_prob(value)

    def __getattr__(self, name):
        return getattr(self._prior, name)


@pytest.mark.slow
def test_compose_boxuniform_rejection_in_original_theta_space():
    """Direct coordinate oracle: the prior-support check during rejection must
    receive THETA-space coordinates (post ``from_z``), not z-space ones.

    The other sample tests use Normal priors, whose ``within_support`` is always
    True, so they cannot detect whether prior rejection happens in z-space or in
    original-theta space. Under composed standardization the SDE/ODE proposal
    samples in z-space (~O(1)); the sampler must map z -> theta (``from_z``) BEFORE
    rejecting against the prior support.

    We choose a BoxUniform support FAR from unit scale (|theta| <= 3e-4) so that
    theta-space coordinates (O(1e-4)) and z-space coordinates (O(1)) are numerically
    disjoint. A recording proxy on the prior captures the exact tensor passed to the
    support check; we assert its magnitude is consistent with the THETA box, not z.
    A regression that rejected in z-space (``from_z`` removed before rejection)
    would feed O(1) values here and fail by a clear assertion (not a timeout).

    Tiny train (few sims/epochs); marked slow.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    bound = 3e-4
    low = -bound * torch.ones(2)
    high = bound * torch.ones(2)
    prior = BoxUniform(low=low, high=high)

    n_train = 500
    theta = prior.sample((n_train,))
    x = theta.sum(1, keepdim=True) + bound * torch.randn(n_train, 1)

    de = posterior_flow_nn(compose_standardization=True)
    tr = FMPE(prior=prior, vf_estimator=de)
    tr.append_simulations(theta, x).train(max_num_epochs=20, show_train_summary=False)
    post = tr.build_posterior()

    # Spy on the coordinates handed to the prior-support check during rejection.
    recorded: list = []
    post.prior = _RecordingPrior(prior, recorded)

    # The proxy records the coordinate on the FIRST rejection batch, so the oracle
    # below does not depend on collecting any accepted samples. A z-space regression
    # would reject almost everything (O(1) proposals vs the |theta|<=3e-4 box) and
    # could otherwise time out instead of asserting; cap the time and tolerate a
    # raise so detection is the DIRECT coordinate assertion, never a timeout.
    samples = None
    # A z-space regression rejects almost everything and may raise on timeout; the
    # recorded coordinate is already captured, so suppress and let the assertion judge.
    with contextlib.suppress(RuntimeError):
        samples = post.sample(
            (8,),
            x=torch.zeros(1),
            show_progress_bars=False,
            max_sampling_time=20.0,
            return_partial_on_timeout=True,
        )

    # The oracle: the rejection path must have queried the prior at least once,
    # and the queried coordinate must be in the THETA box, NOT at z scale.
    assert recorded, "prior-support check was never invoked during rejection"
    max_recorded = max(float(t.abs().max()) for t in recorded)
    # Theta-space magnitude is ~bound (here ~5e-4); z-space would be ~O(1) (>> bound).
    # Allow slack above `bound` for the proposal's pre-rejection tails, but stay far
    # below the O(1) z scale so a z-space regression fails by THIS assertion.
    z_scale_floor = 1e-1
    assert max_recorded < z_scale_floor, (
        "prior-support check received z-space coordinates "
        f"(max|recorded|={max_recorded:.3e} >= {z_scale_floor:g}); "
        "from_z must be applied BEFORE prior rejection"
    )

    # And any returned (theta-space) samples lie within the ORIGINAL BoxUniform box.
    if samples is not None and samples.numel() > 0:
        assert torch.all(samples >= low), (
            "sample below BoxUniform low (z-space rejection?): "
            f"min={samples.min().item()}"
        )
        assert torch.all(samples <= high), (
            "sample above BoxUniform high (z-space rejection?): "
            f"max={samples.max().item()}"
        )
