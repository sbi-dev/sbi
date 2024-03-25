import pytest
from sbi.diagnostics.tarp import TARP, l1, l2
from torch import Tensor, allclose, exp, eye, ones, sqrt, sum, zeros
from torch.distributions import MultivariateNormal as mvn
from torch.distributions import Normal, Uniform
from torch.nn import L1Loss


@pytest.fixture
def onsamples():
    # taken from the paper page 7, section 4.1 Gaussian Toy Model correct case

    nsamples = 100  # samples per simulation
    nsims = 100
    ndims = 5

    base_mean = Uniform(-5, 5)
    base_log_var = Uniform(-5, -1)
    thmu = base_mean.sample((nsims, ndims))
    thsigma = exp(base_log_var.sample((nsims, ndims)))

    theta_pdf = Normal(loc=thmu, scale=thsigma)

    samples = theta_pdf.sample((nsamples,))
    theta = theta_pdf.sample((1,))

    return theta, samples


# @pytest.fixture
# def offsamples():
#     base_pdf = mvn(zeros(3), eye(3))
#     offset = 0.5
#     theta = base_pdf.sample((50,))
#     samples = base_pdf.sample((150,))
#     samples = samples.unsqueeze(0).reshape(3, -1, 3)
#     return theta, samples + offset


def test_onsamples(onsamples):

    theta, samples = onsamples

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
    exp = l1loss(theta, samples)  # sum across last axis

    print(obs.shape, exp.shape, exp)
    assert obs.shape != exp.shape

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

    tarp_ = TARP(num_alpha_bins=30)

    ecp, alpha = tarp_.run(samples, theta)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
