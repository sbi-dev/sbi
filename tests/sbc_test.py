# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal, Uniform

from sbi.diagnostics import check_sbc, get_nltp, run_sbc
from sbi.inference import SNLE, SNPE, simulate_for_sbi
from sbi.simulators import linear_gaussian
from sbi.utils import BoxUniform, MultipleIndependent
from tests.test_utils import PosteriorPotential, TractablePosterior


@pytest.mark.parametrize("reduce_fn_str", ("marginals", "posterior_log_prob"))
@pytest.mark.parametrize("prior", ("boxuniform", "independent"))
@pytest.mark.parametrize(
    "method, sampler",
    (
        (SNPE, None),
        pytest.param(SNLE, "mcmc", marks=pytest.mark.mcmc),
        pytest.param(SNLE, "vi", marks=pytest.mark.mcmc),
    ),
)
def test_running_sbc(
    method, prior, reduce_fn_str, sampler, mcmc_params_accurate: dict, model="mdn"
):
    """Tests running inference and then SBC and obtaining nltp."""

    num_dim = 2
    if prior == "boxuniform":
        prior = BoxUniform(-torch.ones(num_dim), torch.ones(num_dim))
    else:
        prior = MultipleIndependent([
            Uniform(-torch.ones(1), torch.ones(1)) for _ in range(num_dim)
        ])

    num_simulations = 100
    max_num_epochs = 1
    num_sbc_runs = 2

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inferer = method(prior, show_progress_bars=False, density_estimator=model)

    theta, x = simulate_for_sbi(simulator, prior, num_simulations)

    _ = inferer.append_simulations(theta, x).train(
        training_batch_size=100, max_num_epochs=max_num_epochs
    )
    if method == SNLE:
        posterior_kwargs = {
            "sample_with": "mcmc" if sampler == "mcmc" else "vi",
            "mcmc_method": "slice_np_vectorized",
            "mcmc_parameters": mcmc_params_accurate,
        }
    else:
        posterior_kwargs = {}

    posterior = inferer.build_posterior(**posterior_kwargs)

    thetas = prior.sample((num_sbc_runs,))
    xs = simulator(thetas)

    reduce_fn = "marginals" if reduce_fn_str == "marginals" else posterior.log_prob
    run_sbc(
        thetas,
        xs,
        posterior,
        num_workers=1,
        num_posterior_samples=10,
        reduce_fns=reduce_fn,
    )

    # Check nltp
    get_nltp(thetas, xs, posterior)


@pytest.mark.slow
@pytest.mark.parametrize("method", [SNPE])
def test_consistent_sbc_results(method, model="mdn"):
    """Tests running inference and then SBC and obtaining nltp."""

    num_dim = 2
    prior = BoxUniform(-torch.ones(num_dim), torch.ones(num_dim))

    num_simulations = 1000
    max_num_epochs = 20
    num_sbc_runs = 100

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
    num_posterior_samples = 1000
    thetas = prior.sample((num_sbc_runs,))
    xs = simulator(thetas)

    mranks, mdaps = run_sbc(
        thetas,
        xs,
        posterior,
        num_workers=1,
        num_posterior_samples=num_posterior_samples,
    )
    mstats = check_sbc(
        mranks, thetas, mdaps, num_posterior_samples=num_posterior_samples
    )
    lranks, ldaps = run_sbc(
        thetas,
        xs,
        posterior,
        num_workers=1,
        num_posterior_samples=num_posterior_samples,
        reduce_fns=posterior.log_prob,
    )
    lstats = check_sbc(
        lranks, thetas, ldaps, num_posterior_samples=num_posterior_samples
    )

    assert lstats["ks_pvals"] > 0.05
    assert (mstats["ks_pvals"] > 0.05).all()

    assert lstats["c2st_ranks"] < 0.75
    assert (mstats["c2st_ranks"] < 0.75).all()


def test_sbc_accuracy():
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
        reduce_fns="marginals",
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
