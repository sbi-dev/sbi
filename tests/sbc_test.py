# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal, Uniform

from sbi.analysis import check_sbc, get_nltp, run_sbc
from sbi.inference import SNLE, SNPE, simulate_for_sbi
from sbi.simulators import linear_gaussian
from sbi.utils import BoxUniform, MultipleIndependent
from tests.test_utils import PosteriorPotential, TractablePosterior


@pytest.mark.parametrize("reduce_fn_str", ("marginals", "posterior_log_prob"))
@pytest.mark.parametrize("prior", ("boxuniform", "independent"))
@pytest.mark.parametrize("method", (SNPE, SNLE))
def test_running_sbc(method, prior, reduce_fn_str, model="mdn"):
    """Tests running inference and then SBC and obtaining nltp."""

    num_dim = 2
    if prior == "boxuniform":
        prior = BoxUniform(-torch.ones(num_dim), torch.ones(num_dim))
    else:
        prior = MultipleIndependent(
            [Uniform(-torch.ones(1), torch.ones(1)) for _ in range(num_dim)]
        )

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
    posterior = inferer.build_posterior()

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
    N = 10000
    L = 1000

    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    # Daps and ranks from prior for testing.
    daps = prior.sample((N,))
    ranks = torch.distributions.Uniform(zeros(num_dim), L * ones(num_dim)).sample((N,))

    checks = check_sbc(ranks, prior.sample((N,)), daps, num_posterior_samples=L)
    assert (checks["ks_pvals"] > 0.05).all()
    assert (checks["c2st_ranks"] < 0.55).all()
    assert (checks["c2st_dap"] < 0.55).all()
