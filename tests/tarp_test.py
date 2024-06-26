import pytest
from scipy.stats import uniform
from torch import Tensor, allclose, exp, eye, ones
from torch.distributions import Normal, Uniform
from torch.nn import L1Loss

from sbi.diagnostics.tarp import (
    _infer_posterior_on_batch,
    _prepare_estimates,
    _run_tarp,
    check_tarp,
    run_tarp,
)
from sbi.inference import SNPE, simulate_for_sbi
from sbi.simulators import linear_gaussian
from sbi.utils import BoxUniform
from sbi.utils.metrics import l1, l2


def generate_toy_gaussian(nsamples=100, nsims=100, ndims=5, covfactor=1.0):
    """adopted from the tarp paper page 7, section 4.1 Gaussian Toy Model
    correct case"""

    base_mean = Uniform(-5, 5)
    base_log_var = Uniform(-5, -1)

    locs = base_mean.sample((nsims, ndims))
    scales = exp(base_log_var.sample((nsims, ndims)))

    spdf = Normal(loc=locs, scale=covfactor * scales)
    tpdf = Normal(loc=locs, scale=scales)

    samples = spdf.sample((nsamples,))
    theta_prime = tpdf.sample()

    return theta_prime, samples


def biased_toy_gaussian(nsamples=100, nsims=100, ndims=5, covfactor=1.0):
    """adopted from the tarp paper page 7, section 4.1 Gaussian Toy Model
    correct case"""

    base_mean = Uniform(-5, 5)
    base_mean_ = uniform(-5, 5)
    base_log_var = Uniform(-5, -1)

    locs_ = base_mean.sample((nsims, ndims))
    scales = exp(base_log_var.sample((nsims, ndims)))
    locs = locs_ - locs_.sign() * base_mean_.isf(locs_) * scales

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

    return generate_toy_gaussian(nsamples, nsims, ndims, covfactor=0.25)


@pytest.fixture
def oversamples():
    # taken from the paper page 7, section 4.1 Gaussian Toy Model overconfident case

    nsamples = 100  # samples per simulation
    nsims = 100
    ndims = 5

    return generate_toy_gaussian(nsamples, nsims, ndims, covfactor=4.0)


@pytest.fixture
def biased():
    nsamples = 100  # samples per simulation
    nsims = 100
    ndims = 5

    return biased_toy_gaussian(nsamples, nsims, ndims, covfactor=2.0)


def test_onsamples(onsamples):
    theta, samples = onsamples

    assert theta.shape == (100, 5) or theta.shape == (1, 100, 5)
    assert samples.shape == (100, 100, 5)


def test_undersamples(undersamples):
    theta, samples = undersamples

    assert theta.shape == (100, 5) or theta.shape == (1, 100, 5)
    assert samples.shape == (100, 100, 5)


def test_biased(biased):
    theta, samples = biased

    assert theta.shape == (100, 5) or theta.shape == (1, 100, 5)
    assert samples.shape == (100, 100, 5)


######################################################################
## test TARP library


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


######################################################################
## Reproduce Toy Examples in paper, see Section 4.1


def test_run_tarp_correct(onsamples):
    theta, samples = onsamples

    ecp, alpha = _run_tarp(samples, theta, num_bins=30)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)

    ecp, alpha = _run_tarp(samples, theta, distance=l1, num_bins=30)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_run_tarp_correct_using_norm(onsamples):
    theta, samples = onsamples

    ecp, alpha = _run_tarp(samples, theta, num_bins=30, do_norm=False)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (
        ecp - alpha
    ).abs().sum() < 1.0  # integral of residuals should vanish, fig.2 in paper

    ecp, alpha = _run_tarp(samples, theta, num_bins=30, do_norm=False, distance=l1)

    # TARP detects that this is a correct representation of the posterior
    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_run_tarp_detect_overdispersed(oversamples):
    theta, samples = oversamples

    ecp, alpha = _run_tarp(samples, theta, num_bins=30, do_norm=True)

    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper

    ecp, alpha = _run_tarp(samples, theta, num_bins=30, do_norm=True, distance=l1)

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_run_tarp_detect_underdispersed(undersamples):
    theta, samples = undersamples

    ecp, alpha = _run_tarp(samples, theta, num_bins=30, do_norm=True)

    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper

    ecp, alpha = _run_tarp(samples, theta, num_bins=30, do_norm=True, distance=l1)

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_run_tarp_detect_bias(biased):
    theta, samples = biased

    ecp, alpha = _run_tarp(samples, theta, num_bins=30, do_norm=True)

    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper

    ecp, alpha = _run_tarp(samples, theta, num_bins=30, do_norm=True, distance=l1)

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_check_tarp_correct(onsamples):
    theta, samples = onsamples

    ecp, alpha = _run_tarp(samples, theta, num_bins=30, do_norm=False)
    print("onsamples")
    print("tarp results", ecp, alpha)
    atc, kspvals = check_tarp(ecp, alpha)

    print("tarp checks", atc, kspvals)
    assert atc != 0.0
    assert atc < 1.0

    assert kspvals > 0.05  # samples are likely from the same PDF


def test_check_tarp_underdispersed(undersamples):
    theta, samples = undersamples

    ecp, alpha = _run_tarp(samples, theta, num_bins=30, do_norm=False)
    print("underdispersed")
    print("tarp results", ecp, alpha)
    atc, kspvals = check_tarp(ecp, alpha)

    print("tarp checks", atc, kspvals)

    assert atc != 0.0
    assert atc < -2.0
    # assert atc < -1.0 # TODO: need to check why this breaks

    # TODO: need to check why this breaks
    assert kspvals < 0.2  # samples are unlikely from the same PDF


def test_check_tarp_overdispersed(oversamples):
    theta, samples = oversamples

    ecp, alpha = _run_tarp(samples, theta, num_bins=50, do_norm=False)
    print("overdispersed")
    print("tarp results", ecp, alpha)
    atc, kspvals = check_tarp(ecp, alpha)

    print("tarp checks", atc, kspvals)

    assert atc != 0.0
    assert atc > 2.0

    assert kspvals < 0.05  # samples are unlikely from the same PDF


def test_check_tarp_detect_bias(biased):
    theta, samples = biased

    ecp, alpha = _run_tarp(samples, theta, num_bins=30, do_norm=True)
    print("biased")
    print("tarp results", ecp, alpha)
    atc, kspvals = check_tarp(ecp, alpha)

    print("tarp checks", atc, kspvals)
    assert atc != 0.0
    assert atc > 1.0

    assert kspvals < 0.05  # samples are unlikely from the same PDF


######################################################################
## Check TARP with SBI


@pytest.mark.slow
@pytest.mark.parametrize("method", [SNPE])
def test_batched_prepare_estimates(method, model="mdn"):
    """Tests running inference and checking samples with tarp."""

    num_dim = 2
    prior = BoxUniform(-ones(num_dim), ones(num_dim))

    num_simulations = 1000
    max_num_epochs = 20
    num_tarp_runs = 100

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inferer = method(prior, show_progress_bars=False, density_estimator=model)

    theta, x = simulate_for_sbi(simulator, prior, num_simulations)

    _ = inferer.append_simulations(theta, x).train(
        training_batch_size=100, max_num_epochs=max_num_epochs
    )

    posterior = inferer.build_posterior()
    num_posterior_samples = 256
    thetas = prior.sample((num_tarp_runs,))
    xs = simulator(thetas)

    samples = _infer_posterior_on_batch(xs, posterior, num_posterior_samples)

    assert samples.shape != thetas.shape
    assert samples.shape[1:] == thetas.shape
    assert samples.shape[0] == num_posterior_samples

    samples_ = _prepare_estimates(
        xs, posterior, num_posterior_samples, infer_batch_size=32
    )

    assert samples_.shape != thetas.shape
    assert samples_.shape[1:] == thetas.shape
    assert samples_.shape[0] == num_posterior_samples
    assert samples_.shape == samples.shape


@pytest.mark.slow
@pytest.mark.parametrize("method", [SNPE])
def test_consistent_run_tarp_results_with_posterior(method, model="mdn"):
    """Tests running inference and checking samples with tarp."""

    num_dim = 2
    prior = BoxUniform(-ones(num_dim), ones(num_dim))

    num_simulations = 6000
    num_tarp_sims = 1000
    num_posterior_samples = 1000

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inferer = method(prior, show_progress_bars=True, density_estimator=model)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    _ = inferer.append_simulations(theta, x).train(training_batch_size=1000)

    posterior = inferer.build_posterior()

    thetas = prior.sample((num_tarp_sims,))
    xs = simulator(thetas)

    ecp, alpha = run_tarp(
        thetas,
        xs,
        posterior=posterior,
        num_posterior_samples=num_posterior_samples,
        num_bins=30,
        do_norm=True,
        rng_seed=41,
    )

    atc, kspvals = check_tarp(ecp, alpha)
    print(atc, kspvals)
    assert -0.5 < atc < 0.5
    assert kspvals > 0.05
