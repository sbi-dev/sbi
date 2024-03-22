import pytest
from sbi.diagnostics.tarp import TARP
from torch import allclose, eye, ones, zeros
from torch.distributions import MultivariateNormal as mvn
from torch.distributions import Uniform


@pytest.fixture
def onsamples():
    # taken from the paper page 7, section 4.1 Gaussian Toy Model correct case

    prior = Uniform(-5, 5)
    post_log_var = Uniform(-5, -1)

    theta = prior.sample((50, 3))

    samples = base_pdf.sample((150,))
    samples = samples.unsqueeze(0).reshape(3, -1, 3)
    return theta, samples


@pytest.fixture
def offsamples():
    base_pdf = mvn(zeros(3), eye(3))
    offset = 0.5
    theta = base_pdf.sample((50,))
    samples = base_pdf.sample((150,))
    samples = samples.unsqueeze(0).reshape(3, -1, 3)
    return theta, samples + offset


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

    tarp_ = TARP(num_alpha_bins=10)

    true_ecp, true_alpha = tarp_.run(samples, theta)

    print(true_ecp)
