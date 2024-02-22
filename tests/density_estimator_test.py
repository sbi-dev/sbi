# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.neural_nets.density_estimators.flow import NFlowsNSF
from sbi.neural_nets.flow import build_nsf


@pytest.mark.parametrize("density_estimator", (NFlowsNSF,))
@pytest.mark.parametrize(
    "input_dim",
    (
        1,
        2,
    ),
)
@pytest.mark.parametrize(
    "context_dim",
    (
        1,
        2,
    ),
)
def test_api_density_estimator(density_estimator, input_dim, context_dim):
    """Runs SNL and checks that posterior_sampler is correctly set.

    Args:
        mcmc_method: which mcmc method to use for sampling
        set_seed: fixture for manual seeding

    """
    nsamples = 10
    nsamples_test = 5

    input_mvn = MultivariateNormal(
        loc=zeros(input_dim), covariance_matrix=eye(input_dim)
    )
    batch_input = input_mvn.sample((nsamples,))
    context_mvn = MultivariateNormal(
        loc=zeros(context_dim), covariance_matrix=eye(context_dim)
    )
    batch_context = context_mvn.sample((nsamples,))

    net = build_nsf(batch_input, batch_context, hidden_features=10, num_transforms=2)
    estimator = density_estimator(net)

    log_probs = estimator.log_prob(batch_input, batch_context)
    assert log_probs.shape == (nsamples,), "log_prob shape is not correct"

    loss = estimator.loss(batch_input, batch_context)
    loss_item = loss.item()  # test loss is scalar

    samples = estimator.sample((nsamples_test,), batch_context[0]).reshape(
        nsamples_test, -1
    )
    assert samples.shape == (nsamples_test, input_dim), "samples shape is not correct"
