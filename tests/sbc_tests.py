# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.diagnostics import check_sbc, run_sbc
from sbi.inference import SNPE_C, simulate_for_sbi
from sbi.simulators import linear_gaussian


@pytest.mark.slow
@pytest.mark.parametrize(
    "method, model",
    [
        (SNPE_C, "mdn"),
    ],
)
def test_running_sbc(method, model):

    num_dim = 2
    num_simulations = 10000
    max_num_epochs = 5

    x_o = zeros(1, num_dim)
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inferer = method(prior, show_progress_bars=False, density_estimator=model)

    theta, x = simulate_for_sbi(simulator, prior, num_simulations)

    _ = inferer.append_simulations(theta, x).train(
        training_batch_size=100, max_num_epochs=max_num_epochs
    )
    posterior = inferer.build_posterior().set_default_x(x_o)

    run_sbc(prior, simulator, posterior, num_workers=5)


def test_sbc_checks():
    """Test the uniformity checks for SBC."""

    num_dim = 2
    N = 10000
    L = 1000

    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    # Dummy log prob true params.
    log_probs = prior.log_prob(prior.sample((N,)))
    # Daps and ranks from prior for testing.
    daps = prior.sample((N,))
    ranks = torch.distributions.Uniform(zeros(num_dim), L * ones(num_dim)).sample((N,))

    checks = check_sbc(ranks, log_probs, prior.sample((N,)), daps, num_ranks=L)
    assert (checks["ks_pvals"] > 0.05).all()
    assert (checks["c2st_ranks"] < 0.55).all()
    assert (checks["c2st_dap"] < 0.55).all()
