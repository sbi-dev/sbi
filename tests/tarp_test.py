import pytest
from sbi.diagnostics.tarp import (TARP, infer_posterior_on_batch, l1, l2,
                                  prepare_estimates, run_tarp)
from sbi.inference import SNPE, simulate_for_sbi
from sbi.simulators import linear_gaussian
from sbi.utils import BoxUniform
from scipy.stats import uniform
from torch import Tensor, allclose, exp, eye, ones
from torch.distributions import Normal, Uniform
from torch.nn import L1Loss


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

    return generate_toy_gaussian(nsamples, nsims, ndims, covfactor=0.5)


@pytest.fixture
def oversamples():
    # taken from the paper page 7, section 4.1 Gaussian Toy Model overconfident case

    nsamples = 100  # samples per simulation
    nsims = 100
    ndims = 5

    return generate_toy_gaussian(nsamples, nsims, ndims, covfactor=2.0)


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


def test_tarp_constructs():
    tarp_ = TARP()
    assert isinstance(tarp_, TARP)


def test_tarp_runs(onsamples):
    theta, samples = onsamples
    tarp_ = TARP()

    assert theta.shape != samples.shape

    ecp, alpha = tarp_.check(samples, theta)
    assert ecp.shape == alpha.shape


def test_run_tarp_function(onsamples):
    theta, samples = onsamples
    tarp_ = TARP()

    assert theta.shape != samples.shape

    ecp, alpha = run_tarp(samples, theta)
    assert ecp.shape == alpha.shape


######################################################################
## Reproduce Toy Examples in paper, see Section 4.1


def test_tarp_correct(onsamples):
    theta, samples = onsamples

    tarp = TARP(num_alpha_bins=30)
    ecp, alpha = tarp.check(samples, theta)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)

    tarp = TARP(num_alpha_bins=30, metric="l1")
    ecp, alpha = tarp.check(samples, theta)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_run_tarp_function_correct(onsamples):
    theta, samples = onsamples

    ecp, alpha = run_tarp(samples, theta, num_bins=30)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)

    ecp, alpha = run_tarp(samples, theta, distance=l1, num_bins=30)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_tarp_correct_using_norm(onsamples):
    theta, samples = onsamples

    tarp = TARP(num_alpha_bins=30, norm=True)
    ecp, alpha = tarp.check(samples, theta)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (
        ecp - alpha
    ).abs().sum() < 1.0  # integral of residuals should vanish, fig.2 in paper

    tarp_ = TARP(num_alpha_bins=30, norm=True, metric="l1")
    ecp, alpha = tarp_.check(samples, theta)

    # TARP detects that this is a correct representation of the posterior
    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_run_tarp_function_correct_using_norm(onsamples):
    theta, samples = onsamples

    # tarp = TARP(num_alpha_bins=30, norm=True)
    ecp, alpha = run_tarp(samples, theta, num_bins=30, do_norm=False)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (
        ecp - alpha
    ).abs().sum() < 1.0  # integral of residuals should vanish, fig.2 in paper

    # tarp_ = TARP(num_alpha_bins=30, norm=True, metric="l1")
    ecp, alpha = run_tarp(samples, theta, num_bins=30, do_norm=False, distance=l1)

    # TARP detects that this is a correct representation of the posterior
    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_tarp_detect_overdispersed(oversamples):
    theta, samples = oversamples

    tarp = TARP(num_alpha_bins=30, norm=True)
    ecp, alpha = tarp.check(samples, theta)

    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper

    tarp_ = TARP(num_alpha_bins=30, norm=True, metric="l1")
    ecp, alpha = tarp_.check(samples, theta)

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_run_tarp_function_detect_overdispersed(oversamples):
    theta, samples = oversamples

    # tarp = TARP(num_alpha_bins=30, norm=True)
    ecp, alpha = run_tarp(samples, theta, num_bins=30, do_norm=True)

    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper

    # tarp_ = TARP(num_alpha_bins=30, norm=True, metric="l1")
    ecp, alpha = run_tarp(samples, theta, num_bins=30, do_norm=True, distance=l1)

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_tarp_detect_underdispersed(undersamples):
    theta, samples = undersamples

    tarp = TARP(num_alpha_bins=30, norm=True)
    ecp, alpha = tarp.check(samples, theta)

    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper

    tarp_ = TARP(num_alpha_bins=30, norm=True, metric="l1")
    ecp, alpha = tarp_.check(samples, theta)

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_run_tarp_function_detect_underdispersed(undersamples):
    theta, samples = undersamples

    # tarp = TARP(num_alpha_bins=30, norm=True)
    ecp, alpha = run_tarp(samples, theta, num_bins=30, do_norm=True)

    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper

    # tarp_ = TARP(num_alpha_bins=30, norm=True, metric="l1")
    ecp, alpha = run_tarp(samples, theta, num_bins=30, do_norm=True, distance=l1)

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_tarp_detect_bias(biased):
    theta, samples = biased

    tarp = TARP(num_alpha_bins=30, norm=True)
    ecp, alpha = tarp.check(samples, theta)

    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper

    tarp_ = TARP(num_alpha_bins=30, norm=True, metric="l1")
    ecp, alpha = tarp_.check(samples, theta)

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


def test_run_tarp_function_detect_bias(biased):
    theta, samples = biased

    # tarp = TARP(num_alpha_bins=30, norm=True)
    ecp, alpha = run_tarp(samples, theta, num_bins=30, do_norm=True)

    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper

    # tarp_ = TARP(num_alpha_bins=30, norm=True, metric="l1")
    ecp, alpha = run_tarp(samples, theta, num_bins=30, do_norm=True, distance=l1)

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)


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

    samples = infer_posterior_on_batch(xs, posterior, num_posterior_samples)

    assert samples.shape != thetas.shape
    assert samples.shape[1:] == thetas.shape
    assert samples.shape[0] == num_posterior_samples

    # tarp = TARP(num_alpha_bins=30, norm=True, metric="l2")
    samples_ = prepare_estimates(
        xs, posterior, num_posterior_samples, infer_batch_size=32
    )

    assert samples_.shape != thetas.shape
    assert samples_.shape[1:] == thetas.shape
    assert samples_.shape[0] == num_posterior_samples
    assert samples_.shape == samples.shape


@pytest.mark.slow
@pytest.mark.parametrize("method", [SNPE])
def test_consistent_tarp_results_with_posterior(method, model="mdn"):
    """Tests running inference and checking samples with tarp."""

    num_dim = 2
    prior = BoxUniform(-ones(num_dim), ones(num_dim))

    num_simulations = 1000
    max_num_epochs = 20
    num_tarp_sims = 100

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
    thetas = prior.sample((num_tarp_sims,))
    xs = simulator(thetas)

    tarp = TARP(num_alpha_bins=30, norm=True, metric="l2")

    samples = tarp.run(xs, posterior, num_posterior_samples)

    ecp, alpha = tarp.check(samples, thetas)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() < 1.0


@pytest.mark.slow
@pytest.mark.parametrize("method", [SNPE])
def test_consistent_run_tarp_function_results_with_posterior(method, model="mdn"):
    """Tests running inference and checking samples with tarp."""

    num_dim = 2
    prior = BoxUniform(-ones(num_dim), ones(num_dim))

    num_simulations = 1000
    max_num_epochs = 20
    num_tarp_sims = 100

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

    thetas = prior.sample((num_tarp_sims,))
    xs = simulator(thetas)

    # tarp = TARP(num_alpha_bins=30, norm=True, metric="l2")

    samples = prepare_estimates(xs, posterior, num_posterior_samples)

    ecp, alpha = run_tarp(samples, thetas, num_bins=30, do_norm=True)

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() < 1.0
