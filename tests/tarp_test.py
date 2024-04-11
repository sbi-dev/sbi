import numpy as np
import pytest
from sbi.diagnostics.tarp import TARP, l1, l2
from torch import Tensor, allclose, exp, eye, normal, ones, sqrt, sum, zeros
from torch.distributions import MultivariateNormal as mvn
from torch.distributions import Normal, Uniform
from torch.nn import L1Loss


def generate_toy_gaussian(nsamples=100, nsims=100, ndims=5, covfactor=1.0):
    """adopted from the tarp paper page 7, section 4.1 Gaussian Toy Model correct case"""

    base_mean = Uniform(-5, 5)
    base_log_var = Uniform(-5, -1)

    locs = base_mean.sample((nsims, ndims))
    scales = exp(base_log_var.sample((nsims, ndims)))

    spdf = Normal(loc=locs, scale=covfactor * scales)
    tpdf = Normal(loc=locs, scale=scales)

    samples = spdf.sample((nsamples,))
    theta_prime = tpdf.sample()

    return theta_prime, samples


@pytest.fixture
def onsamples():

    nsamples = 100  # samples per simulation
    nsims = 100
    ndims = 5

    return generate_toy_gaussian(nsamples, nsims, ndims)


@pytest.fixture
def undersamples():
    # taken from the paper page 7, section 4.1 Gaussian Toy Model underconfident case

    nsamples = 100  # samples per simulation
    nsims = 100
    ndims = 5

    return generate_toy_gaussian(nsamples, nsims, ndims, covfactor=0.5)


@pytest.fixture
def oversamples():
    # taken from the paper page 7, section 4.1 Gaussian Toy Model overconfident case

    nsamples = 100  # samples per simulation
    nsims = 100
    ndims = 5

    return generate_toy_gaussian(nsamples, nsims, ndims, covfactor=2.0)


def test_onsamples(onsamples):

    theta, samples = onsamples

    assert theta.shape == (100, 5) or theta.shape == (1, 100, 5)
    assert samples.shape == (100, 100, 5)


def test_undersamples(undersamples):

    theta, samples = undersamples

    assert theta.shape == (100, 5) or theta.shape == (1, 100, 5)
    assert samples.shape == (100, 100, 5)


def test_distances(onsamples):

    theta, samples = onsamples

    obs = l2(theta, samples)

    assert obs.shape == (100, 100)

    obs = l1(theta, samples)

    assert obs.shape == (100, 100)

    # difference in reductions
    l1loss = L1Loss(reduction="sum")  # sum across last axis AND batch
    broadcasted_theta = theta.expand(samples.shape[0], -1, -1)
    exp = l1loss(broadcasted_theta, samples)  # sum across last axis

    assert obs.shape != exp.shape  # gives the wrong shape

    # results including expansion
    theta_ = theta.expand(samples.shape[0], -1, -1)
    obs_ = l1(theta_, samples)

    assert allclose(obs, obs_)


def test_tarp_constructs():

    tarp_ = TARP()
    assert isinstance(tarp_, TARP)


def test_tarp_runs(onsamples):

    theta, samples = onsamples
    tarp_ = TARP()

    assert theta.shape != samples.shape

    ecp, alpha = tarp_.run(samples, theta)
    assert ecp.shape == alpha.shape


def test_tarp_correct(onsamples):

    theta, samples = onsamples

    tarp = TARP(num_alpha_bins=30)
    ecp, alpha = tarp.run(samples, theta)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)

    tarp = TARP(num_alpha_bins=30, metric="l1")
    ecp, alpha = tarp.run(samples, theta)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_tarp_correct_using_norm(onsamples):

    theta, samples = onsamples

    tarp = TARP(num_alpha_bins=30, norm=True)
    ecp, alpha = tarp.run(samples, theta)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (
        ecp - alpha
    ).abs().sum() < 1.0  # integral of residuals should vanish, fig.2 in paper

    tarp_ = TARP(num_alpha_bins=30, norm=True, metric="l1")
    ecp, alpha = tarp_.run(samples, theta)

    # TARP detects that this is a correct representation of the posterior
    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_tarp_detect_overdispersed(oversamples):

    theta, samples = oversamples

    tarp = TARP(num_alpha_bins=30, norm=True)
    ecp, alpha = tarp.run(samples, theta)

    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper

    tarp_ = TARP(num_alpha_bins=30, norm=True, metric="l1")
    ecp, alpha = tarp_.run(samples, theta)

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_tarp_detect_underdispersed(undersamples):

    theta, samples = undersamples

    tarp = TARP(num_alpha_bins=30, norm=True)
    ecp, alpha = tarp.run(samples, theta)

    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper

    tarp_ = TARP(num_alpha_bins=30, norm=True, metric="l1")
    ecp, alpha = tarp_.run(samples, theta)

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
