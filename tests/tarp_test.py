import pytest
from scipy.stats import uniform
from torch import Tensor, allclose, exp, eye, ones
from torch.distributions import Normal, Uniform
from torch.nn import L1Loss

from sbi.analysis.plot import plot_tarp
from sbi.diagnostics.tarp import _run_tarp, check_tarp, get_tarp_references, run_tarp
from sbi.inference import NPE
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
def accurate_samples():
    nsamples = 100  # samples per simulation
    nsims = 100
    ndims = 5

    return generate_toy_gaussian(nsamples, nsims, ndims)


@pytest.fixture
def underdispersed_samples():
    # taken from the paper page 7, section 4.1 Gaussian Toy Model underconfident case

    nsamples = 100  # samples per simulation
    nsims = 100
    ndims = 5

    return generate_toy_gaussian(nsamples, nsims, ndims, covfactor=0.25)


@pytest.fixture
def overdispersed_samples():
    # taken from the paper page 7, section 4.1 Gaussian Toy Model overconfident case

    nsamples = 100  # samples per simulation
    nsims = 100
    ndims = 5

    return generate_toy_gaussian(nsamples, nsims, ndims, covfactor=4.0)


@pytest.fixture
def biased_samples():
    nsamples = 100  # samples per simulation
    nsims = 100
    ndims = 5

    return biased_toy_gaussian(nsamples, nsims, ndims, covfactor=2.0)


######################################################################
## test TARP library


def test_distances(accurate_samples):
    theta, samples = accurate_samples

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


@pytest.mark.parametrize("distance", (l1, l2))
@pytest.mark.parametrize("z_score_theta", (True, False))
def test_run_tarp_correct(distance, z_score_theta, accurate_samples):
    theta, samples = accurate_samples

    references = get_tarp_references(theta)

    ecp, alpha = _run_tarp(
        samples,
        theta,
        references,
        distance=distance,
        z_score_theta=z_score_theta,
        num_bins=30,
    )

    assert allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (
        ecp - alpha
    ).abs().sum() < 1.0  # integral of residuals should vanish, fig.2 in paper


@pytest.mark.parametrize("distance", (l1, l2))
def test_run_tarp_detect_overdispersed(distance, overdispersed_samples):
    theta, samples = overdispersed_samples
    references = get_tarp_references(theta)

    ecp, alpha = _run_tarp(
        samples, theta, references, num_bins=30, z_score_theta=True, distance=distance
    )

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper


@pytest.mark.parametrize("distance", (l1, l2))
def test_run_tarp_detect_underdispersed(distance, underdispersed_samples):
    theta, samples = underdispersed_samples
    references = get_tarp_references(theta)

    ecp, alpha = _run_tarp(
        samples, theta, references, num_bins=30, z_score_theta=True, distance=distance
    )

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper


@pytest.mark.parametrize("distance", (l1, l2))
def test_run_tarp_detect_bias(distance, biased_samples):
    theta, samples = biased_samples
    references = get_tarp_references(theta)

    ecp, alpha = _run_tarp(
        samples, theta, references, distance=distance, num_bins=30, z_score_theta=True
    )

    # TARP detects that this is NOT a correct representation of the posterior
    # hence we test for not allclose
    assert not allclose((ecp - alpha).abs().max(), Tensor([0.0]), atol=1e-1)
    assert (ecp - alpha).abs().sum() > 3.0  # integral is nonzero, fig.2 in paper


def test_check_tarp_correct(accurate_samples):
    """Test TARP on correct samples."""
    theta, samples = accurate_samples
    references = get_tarp_references(theta)

    ecp, alpha = _run_tarp(samples, theta, references)
    atc, kspvals = check_tarp(ecp, alpha)

    assert -0.75 < atc < 0.75, "TARP should not detect bias"
    assert kspvals > 0.5  # samples are likely from the same PDF


def test_check_tarp_underdispersed(underdispersed_samples):
    """Test TARP on underdispersed samples."""
    theta, samples = underdispersed_samples
    references = get_tarp_references(theta)

    ecp, alpha = _run_tarp(samples, theta, references, num_bins=30)
    atc, kspvals = check_tarp(ecp, alpha)

    # TARP should detect that the posterior is underdispersed (atc < -1.0)
    assert atc < -2.0
    # and p-values should be relatively small
    assert kspvals < 0.05


def test_check_tarp_overdispersed(overdispersed_samples):
    theta, samples = overdispersed_samples
    references = get_tarp_references(theta)

    ecp, alpha = _run_tarp(samples, theta, references, num_bins=50, z_score_theta=False)
    atc, kspvals = check_tarp(ecp, alpha)

    assert atc != 0.0
    assert atc > 2.0

    assert kspvals < 0.05  # samples are unlikely from the same PDF


def test_check_tarp_detect_bias(biased_samples):
    theta, samples = biased_samples
    references = get_tarp_references(theta)

    ecp, alpha = _run_tarp(samples, theta, references, num_bins=30, z_score_theta=True)
    atc, kspvals = check_tarp(ecp, alpha)

    assert atc != 0.0
    assert atc > 1.0

    assert kspvals < 0.05  # samples are unlikely from the same PDF


######################################################################
## Check TARP with SBI


@pytest.mark.slow
@pytest.mark.parametrize("method", [NPE])
def test_consistent_run_tarp_results_with_posterior(method):
    """Tests running inference and checking samples with tarp."""

    num_dim = 2
    prior = BoxUniform(-ones(num_dim), ones(num_dim))

    num_simulations = 1000
    num_tarp_sims = 500
    num_posterior_samples = 1000

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inferer = method(prior, show_progress_bars=True, density_estimator="maf")

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    inferer.append_simulations(theta, x).train(training_batch_size=200)
    posterior = inferer.build_posterior()

    thetas = prior.sample((num_tarp_sims,))
    xs = simulator(thetas)

    ecp, alpha = run_tarp(
        thetas,
        xs,
        posterior=posterior,
        num_posterior_samples=num_posterior_samples,
    )

    atc, kspvals = check_tarp(ecp, alpha)
    assert -0.5 < atc < 0.5
    assert kspvals > 0.05


# Test tarp plotting
@pytest.mark.parametrize("title", ["Correct", None])
def test_tarp_plotting(title: str, accurate_samples):
    theta, samples = accurate_samples
    references = get_tarp_references(theta)

    ecp, alpha = _run_tarp(samples, theta, references)

    plot_tarp(ecp, alpha, title=title)
