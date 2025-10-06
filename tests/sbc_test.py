# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Callable, Dict, Optional

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal, Uniform

from sbi.analysis import sbc_rank_plot
from sbi.diagnostics import check_sbc, get_nltp, run_sbc
from sbi.inference import NLE, NPE, NPSE
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.posterior_parameters import (
    MCMCPosteriorParameters,
    VIPosteriorParameters,
)
from sbi.simulators.linear_gaussian import linear_gaussian
from sbi.utils import BoxUniform, MultipleIndependent
from tests.test_utils import PosteriorPotential, TractablePosterior


@pytest.fixture
def gaussian_setup():
    """Fixture for common Gaussian test setup."""
    num_dim = 2
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    return {
        "num_dim": num_dim,
        "prior": prior,
        "simulator": simulator,
        "likelihood_shift": likelihood_shift,
        "likelihood_cov": likelihood_cov,
    }


def train_inference_method(
    method_cls: Callable,
    prior: torch.distributions.Distribution,
    simulator: Callable,
    num_simulations: int = 100,
    max_num_epochs: int = 1,
    **kwargs,
) -> NeuralPosterior:
    """Helper function to train an inference method and return its posterior."""
    inferer = method_cls(prior, show_progress_bars=False)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    inferer.append_simulations(theta, x).train(max_num_epochs=max_num_epochs)
    posterior = inferer.build_posterior(**kwargs)

    return posterior


@pytest.mark.parametrize("reduce_fn_str", ("marginals", "posterior_log_prob"))
@pytest.mark.parametrize("prior_type", ("boxuniform", "independent"))
@pytest.mark.parametrize(
    "method, sampler",
    (
        (NPE, None),
        pytest.param(NLE, "mcmc", marks=pytest.mark.mcmc),
        pytest.param(NLE, "vi", marks=pytest.mark.mcmc),
        (NPSE, None),
    ),
)
def test_running_sbc(
    method,
    prior_type: str,
    reduce_fn_str: str,
    sampler: Optional[str],
    mcmc_params_fast: MCMCPosteriorParameters,
):
    """Test running inference and then SBC and obtaining nltp with different methods."""
    # Setup
    num_dim = 2
    if prior_type == "boxuniform":
        prior = BoxUniform(-torch.ones(num_dim), torch.ones(num_dim))
    else:
        prior = MultipleIndependent([
            Uniform(-torch.ones(1), torch.ones(1)) for _ in range(num_dim)
        ])

    # Test parameters
    num_simulations = 100
    max_num_epochs = 1
    num_sbc_runs = 2
    num_posterior_samples = 20

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    # Helper function to simulate data
    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    # Build posterior
    posterior_kwargs = {}
    if method == NLE:
        posterior_kwargs = {
            "posterior_parameters": mcmc_params_fast
            if sampler == "mcmc"
            else VIPosteriorParameters()
        }

    posterior = train_inference_method(
        method,
        prior,
        simulator,
        num_simulations=num_simulations,
        max_num_epochs=max_num_epochs,
        **posterior_kwargs,
    )

    # Generate test data for SBC
    thetas = prior.sample((num_sbc_runs,))
    xs = simulator(thetas)

    # Run SBC
    reduce_fn = "marginals" if reduce_fn_str == "marginals" else posterior.potential
    ranks, _ = run_sbc(
        thetas,
        xs,
        posterior,
        num_posterior_samples=num_posterior_samples,
        reduce_fns=reduce_fn,
    )

    # Basic shape check
    target_rank_dim = num_dim if reduce_fn_str == "marginals" else 1
    assert ranks.shape == (num_sbc_runs, target_rank_dim), "Ranks shape is incorrect"

    # Check nltp calculation (only for normalized posteriors)
    if method in [NPE, NPSE]:
        nltp = get_nltp(thetas, xs, posterior)
        assert nltp.shape == (num_sbc_runs,), "NLTP shape is incorrect"


@pytest.mark.slow
@pytest.mark.parametrize("density_estimator", ["mdn", "maf"])
@pytest.mark.parametrize("cov_method", ("sbc", "coverage"))
def test_consistent_sbc_results(
    density_estimator: str, cov_method: str, gaussian_setup: Dict
):
    """Test consistent SBC results on well-trained NPE."""
    # Extract setup from fixture
    prior = gaussian_setup["prior"]
    simulator = gaussian_setup["simulator"]

    # Test parameters
    num_simulations = 4000
    num_posterior_samples = 1000
    num_sbc_runs = 100

    # Create and train inference
    inference = NPE(prior=prior, density_estimator=density_estimator)
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()

    # Generate test data
    thetas = prior.sample((num_sbc_runs,))
    xs = simulator(thetas)

    # Run SBC
    ranks, dap_samples = run_sbc(
        thetas,
        xs,
        posterior,
        num_workers=1,
        num_posterior_samples=num_posterior_samples,
        # Switch between SBC and expected coverage
        reduce_fns="marginals" if cov_method == "sbc" else posterior.log_prob,
    )

    # Check results
    checks = check_sbc(
        ranks,
        prior.sample((num_sbc_runs,)),
        dap_samples,
        num_posterior_samples=num_posterior_samples,
    )

    # Statistical tests
    assert (checks["ks_pvals"] > 0.05).all(), (
        f"KS p-values too small: {checks['ks_pvals']}"
    )
    assert (checks["c2st_ranks"] < 0.6).all(), (
        f"C2ST ranks too large: {checks['c2st_ranks']}"
    )
    assert (checks["c2st_dap"] < 0.6).all(), f"C2ST DAP too large: {checks['c2st_dap']}"


def test_sbc_accuracy():
    """Test SBC with prior as posterior (perfect calibration case)."""
    num_dim = 2
    # Gaussian toy problem, set posterior = prior
    simulator = lambda theta: torch.randn_like(theta) + theta
    prior = BoxUniform(-ones(num_dim), ones(num_dim))
    posterior_dist = prior

    # Create tractable posterior for testing
    potential = PosteriorPotential(posterior=posterior_dist, prior=prior)
    posterior = TractablePosterior(potential_fn=potential)

    # Run SBC
    N = L = 1000
    thetas = prior.sample((N,))
    xs = simulator(thetas)

    ranks, daps = run_sbc(
        thetas,
        xs,
        posterior,
        num_workers=1,
        num_posterior_samples=L,
    )

    # Check results
    checks = check_sbc(ranks, prior.sample((N,)), daps, num_posterior_samples=L)
    pvals, c2st_ranks, _ = checks.values()

    # With perfect calibration, ranks should be uniform
    assert (c2st_ranks <= 0.6).all(), "posterior ranks must be close to uniform."
    assert (pvals > 0.05).all(), "posterior ranks uniformity test p-values too small."


@pytest.mark.slow
def test_sbc_checks():
    """Test the uniformity checks for SBC with artificial uniform ranks."""
    num_dim = 2
    num_posterior_samples = 1500

    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    # Data averaged posterior samples should be distributed as prior
    daps = prior.sample((num_posterior_samples,))

    # Create perfectly uniform ranks for testing
    ranks = torch.distributions.Uniform(
        zeros(num_dim), num_posterior_samples * ones(num_dim)
    ).sample((num_posterior_samples,))

    # Run checks
    checks = check_sbc(
        ranks,
        prior.sample((num_posterior_samples,)),
        daps,
        num_posterior_samples=num_posterior_samples,
    )

    # With artificial uniform ranks, test statistics should indicate uniformity
    assert (checks["ks_pvals"] > 0.05).all(), "KS test failed on uniform ranks"
    assert (checks["c2st_ranks"] < 0.55).all(), "C2ST failed on uniform ranks"
    assert (checks["c2st_dap"] < 0.55).all(), (
        "C2ST failed on prior-distributed DAP samples"
    )


@pytest.mark.parametrize("num_bins", (None, 30))
@pytest.mark.parametrize("plot_type", ("cdf", "hist"))
@pytest.mark.parametrize("legend_kwargs", (None, {"loc": "upper left"}))
@pytest.mark.parametrize("num_rank_sets", (1, 2))
def test_sbc_plotting(
    num_bins: Optional[int],
    plot_type: str,
    legend_kwargs: Optional[Dict],
    num_rank_sets: int,
):
    """Test SBC plotting functionality with various options."""
    num_dim = 2
    num_posterior_samples = 1000

    # Generate artificial uniform ranks for testing visualization
    ranks = [
        torch.distributions.Uniform(
            zeros(num_dim), num_posterior_samples * ones(num_dim)
        ).sample((num_posterior_samples,))
    ] * num_rank_sets

    # Test that plotting function runs without errors
    fig = sbc_rank_plot(
        ranks,
        num_posterior_samples,
        num_bins=num_bins,
        plot_type=plot_type,
        legend_kwargs=legend_kwargs,
    )

    # Basic check that figure was created
    assert fig is not None, "Plot function should return a figure"


@pytest.mark.parametrize("num_workers", [1, 2])
def test_sbc_parallelization(num_workers: int, gaussian_setup: Dict):
    """Test that SBC produces consistent results with different worker counts."""
    prior = gaussian_setup["prior"]
    simulator = gaussian_setup["simulator"]

    # Parameters
    num_simulations = 200
    num_sbc_runs = 10
    num_posterior_samples = 50

    # Train model
    posterior = train_inference_method(
        NPE, prior, simulator, num_simulations=num_simulations, max_num_epochs=1
    )

    # Generate test data
    thetas = prior.sample((num_sbc_runs,))
    xs = simulator(thetas)

    # Run SBC with specified number of workers
    ranks, _ = run_sbc(
        thetas,
        xs,
        posterior,
        num_posterior_samples=num_posterior_samples,
        num_workers=num_workers,
    )

    # Check shape
    assert ranks.shape == (num_sbc_runs, gaussian_setup["num_dim"]), (
        f"Ranks shape incorrect for {num_workers} workers"
    )


@pytest.mark.parametrize("batch_sampling", [True, False])
def test_sbc_batch_sampling(batch_sampling: bool, gaussian_setup: Dict):
    """Test that SBC works with both batched and non-batched sampling."""
    prior = gaussian_setup["prior"]
    simulator = gaussian_setup["simulator"]

    # Parameters
    num_simulations = 200
    num_sbc_runs = 5
    num_posterior_samples = 50

    # Train model
    posterior = train_inference_method(
        NPE, prior, simulator, num_simulations=num_simulations, max_num_epochs=1
    )

    # Generate test data
    thetas = prior.sample((num_sbc_runs,))
    xs = simulator(thetas)

    # Run SBC with specified batch sampling setting
    ranks, _ = run_sbc(
        thetas,
        xs,
        posterior,
        num_posterior_samples=num_posterior_samples,
        use_batched_sampling=batch_sampling,
    )

    # Check shape
    assert ranks.shape == (num_sbc_runs, gaussian_setup["num_dim"]), (
        f"Ranks shape incorrect with batched_sampling={batch_sampling}"
    )
