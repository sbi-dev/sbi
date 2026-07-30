# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Scale-equivariance regression tests for composed FMPE/NPSE standardization.

Without composition, parameters at scale 1e-5 show inflated posterior width or
weakened coupling.
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
    """Return a homogeneous linear-Gaussian problem."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    prior = Independent(Normal(torch.zeros(2), s * torch.ones(2)), 1)
    theta = prior.sample((N_TRAIN,))
    x = theta.sum(1, keepdim=True) + s * torch.randn(N_TRAIN, 1)
    return prior, theta, x


def _heterogeneous(seed=0):
    """Return a problem mixing unit-scale and 1e-6-scale parameters."""
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
        de = posterior_flow_nn(compose_standardization=compose)
        tr = FMPE(prior=prior, vf_estimator=de)
    else:
        de = posterior_score_nn(sde_type="ve", compose_standardization=compose)
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
def test_compose_standardization_homogeneous_equivariant(kind):
    """Composition preserves posterior coupling under homogeneous rescaling."""
    prior_u, th_u, x_u = _linear_gaussian(1.0)
    rho_u, _ = _coupling_tightness(
        _fit(kind, prior_u, th_u, x_u, True), 1.0, torch.zeros(1)
    )

    s = 1e-5
    prior_i, th_i, x_i = _linear_gaussian(s)
    rho_i, tight_i = _coupling_tightness(
        _fit(kind, prior_i, th_i, x_i, True), s, torch.zeros(1)
    )
    assert tight_i < 3.0, f"ill-scale tightness inflated: {tight_i}"
    assert rho_i < -0.15, f"ill-scale coupling collapsed: {rho_i}"
    assert abs(rho_i - rho_u) < 0.3, f"ill vs unit mismatch: {rho_i} vs {rho_u}"


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["FMPE", "NPSE"])
def test_compose_standardization_heterogeneous_recovers(kind):
    """Composition recovers coupling in a heterogeneous-scale posterior."""
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

    assert tight_fix < 5.0, f"small-scale block not contracted: {tight_fix}"
    assert rho_fix > 0.7, f"composed std failed to recover positive coupling: {rho_fix}"
    assert rho_fix - rho_def > 0.3, (
        f"compose did not beat default on coupling: fix={rho_fix} def={rho_def}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["FMPE", "NPSE"])
def test_compose_log_prob_jacobian_scaling(kind):
    """The log density changes by the affine Jacobian under rescaling."""
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
    """Record values passed to a distribution support check."""

    def __init__(self, support, sink):
        self._support = support
        self._sink = sink

    def check(self, value):
        self._sink.append(value.detach().clone())
        return self._support.check(value)


class _RecordingPrior:
    """Record coordinates evaluated against prior support."""

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
    """Prior rejection receives theta-space rather than standardized samples."""
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

    recorded: list = []
    post.prior = _RecordingPrior(prior, recorded)

    samples = None
    with contextlib.suppress(RuntimeError):
        samples = post.sample(
            (8,),
            x=torch.zeros(1),
            show_progress_bars=False,
            max_sampling_time=20.0,
            return_partial_on_timeout=True,
        )

    assert recorded, "prior-support check was never invoked during rejection"
    max_recorded = max(float(t.abs().max()) for t in recorded)
    z_scale_floor = 1e-1
    assert max_recorded < z_scale_floor, (
        "prior-support check received z-space coordinates "
        f"(max|recorded|={max_recorded:.3e} >= {z_scale_floor:g}); "
        "from_z must be applied BEFORE prior rejection"
    )

    if samples is not None and samples.numel() > 0:
        assert torch.all(samples >= low), (
            "sample below BoxUniform low (z-space rejection?): "
            f"min={samples.min().item()}"
        )
        assert torch.all(samples <= high), (
            "sample above BoxUniform high (z-space rejection?): "
            f"max={samples.max().item()}"
        )
