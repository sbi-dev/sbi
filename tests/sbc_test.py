# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Union

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal, Uniform

from sbi.analysis import sbc_rank_plot
from sbi.diagnostics import check_sbc, get_nltp, run_sbc
from sbi.inference import NLE, NPE, NPSE
from sbi.simulators.linear_gaussian import linear_gaussian
from sbi.utils import BoxUniform, MultipleIndependent
from tests.test_utils import PosteriorPotential, TractablePosterior


@pytest.mark.parametrize("reduce_fn_str", ("marginals", "posterior_log_prob"))
@pytest.mark.parametrize("prior", ("boxuniform", "independent"))
@pytest.mark.parametrize(
    "method, sampler",
    (
        (NPE, None),
        pytest.param(NLE, "mcmc", marks=pytest.mark.mcmc),
        pytest.param(NLE, "vi", marks=pytest.mark.mcmc),
        (NPSE, None),
    ),
)
def test_running_sbc(method, prior, reduce_fn_str, sampler, mcmc_params_accurate: dict):
    """Tests running inference and then SBC and obtaining nltp."""

    num_dim = 2
    if prior == "boxuniform":
        prior = BoxUniform(-torch.ones(num_dim), torch.ones(num_dim))
    else:
        prior = MultipleIndependent([
            Uniform(-torch.ones(1), torch.ones(1)) for _ in range(num_dim)
        ])

    # Fast dummy settings.
    num_simulations = 100
    max_num_epochs = 1
    num_sbc_runs = 2
    num_posterior_samples = 20

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inferer = method(prior, show_progress_bars=False)

    inferer.append_simulations(theta, x).train(max_num_epochs=max_num_epochs)
    if method == NLE:
        posterior_kwargs = {
            "sample_with": "mcmc" if sampler == "mcmc" else "vi",
            "mcmc_method": "slice_np_vectorized",
            "mcmc_parameters": mcmc_params_accurate,
        }
    else:
        posterior_kwargs = {}

    posterior = inferer.build_posterior(**posterior_kwargs)

    thetas = prior.sample((num_sbc_runs,))
    xs = linear_gaussian(thetas, likelihood_shift, likelihood_cov)

    reduce_fn = "marginals" if reduce_fn_str == "marginals" else posterior.log_prob
    run_sbc(
        thetas,
        xs,
        posterior,
        num_posterior_samples=num_posterior_samples,
        reduce_fns=reduce_fn,
    )

    # Check nltp
    get_nltp(thetas, xs, posterior)


@pytest.mark.slow
@pytest.mark.parametrize("density_estimator", ["mdn", "maf"])
@pytest.mark.parametrize("cov_method", ("sbc", "coverage"))
def test_consistent_sbc_results(density_estimator, cov_method):
    """Test consistent SBC results on well-trained NPE."""

    num_dim = 2

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    num_simulations = 4000
    num_posterior_samples = 1000
    num_sbc_runs = 100

    # Create inference object. Here, NPE is used.
    inference = NPE(prior=prior, density_estimator=density_estimator)

    # generate simulations and pass to the inference object
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()

    thetas = prior.sample((num_sbc_runs,))
    xs = simulator(thetas)

    ranks, dap_samples = run_sbc(
        thetas,
        xs,
        posterior,
        num_workers=1,
        num_posterior_samples=num_posterior_samples,
        # switch between SBC and expected coverage.
        reduce_fns="marginals" if cov_method == "sbc" else posterior.log_prob,
    )
    checks = check_sbc(
        ranks,
        prior.sample((num_sbc_runs,)),
        dap_samples,
        num_posterior_samples=num_posterior_samples,
    )

    assert (
        checks["ks_pvals"] > 0.05
    ).all(), f"KS p-values too small: {checks['ks_pvals']}"
    assert (
        checks["c2st_ranks"] < 0.6
    ).all(), f"C2ST ranks too large: {checks['c2st_ranks']}"
    assert (checks["c2st_dap"] < 0.6).all(), f"C2ST DAP too large: {checks['c2st_dap']}"


def test_sbc_accuracy():
    """Test SBC with prior as posterior."""
    num_dim = 2
    # Gaussian toy problem, set posterior = prior
    simulator = lambda theta: torch.randn_like(theta) + theta
    prior = BoxUniform(-ones(num_dim), ones(num_dim))
    posterior_dist = prior

    potential = PosteriorPotential(posterior=posterior_dist, prior=prior)

    posterior = TractablePosterior(potential_fn=potential)

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

    pvals, c2st_ranks, _ = check_sbc(
        ranks, prior.sample((N,)), daps, num_posterior_samples=L
    ).values()
    assert (c2st_ranks <= 0.6).all(), "posterior ranks must be close to uniform."
    assert (pvals > 0.05).all(), "posterior ranks uniformity test p-values too small."


@pytest.mark.slow
def test_sbc_checks():
    """Test the uniformity checks for SBC."""

    num_dim = 2
    num_posterior_samples = 1500

    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    # Data averaged posterior samples should be distributed as prior.
    daps = prior.sample((num_posterior_samples,))
    # Ranks should be distributed uniformly in [0, num_posterior_samples]
    ranks = torch.distributions.Uniform(
        zeros(num_dim), num_posterior_samples * ones(num_dim)
    ).sample((num_posterior_samples,))

    checks = check_sbc(
        ranks,
        prior.sample((num_posterior_samples,)),
        daps,
        num_posterior_samples=num_posterior_samples,
    )
    assert (checks["ks_pvals"] > 0.05).all()
    assert (checks["c2st_ranks"] < 0.55).all()
    assert (checks["c2st_dap"] < 0.55).all()


# add test for sbc plotting
@pytest.mark.parametrize("num_bins", (None, 30))
@pytest.mark.parametrize("plot_type", ("cdf", "hist"))
@pytest.mark.parametrize("legend_kwargs", (None, {"loc": "upper left"}))
@pytest.mark.parametrize("num_rank_sets", (1, 2))
def test_sbc_plotting(
    num_bins: int, plot_type: str, legend_kwargs: Union[None, dict], num_rank_sets: int
):
    """Test the uniformity checks for SBC."""

    num_dim = 2
    num_posterior_samples = 1000

    # Ranks should be distributed uniformly in [0, num_posterior_samples]
    ranks = [
        torch.distributions.Uniform(
            zeros(num_dim), num_posterior_samples * ones(num_dim)
        ).sample((num_posterior_samples,))
    ] * num_rank_sets

    sbc_rank_plot(
        ranks,
        num_posterior_samples,
        num_bins=num_bins,
        plot_type=plot_type,
        legend_kwargs=legend_kwargs,
    )
