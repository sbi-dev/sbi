# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.neural_nets.density_estimators.flow import NFlowsFlow
from sbi.neural_nets.flow import build_nsf


@pytest.mark.parametrize("density_estimator", (NFlowsFlow,))
@pytest.mark.parametrize("input_dim", (1, 2))
@pytest.mark.parametrize("context_dim", (1, 2))
def test_api_density_estimator(density_estimator, input_dim, context_dim):
    r"""Checks whether we can evaluate and sample from density estimators correctly.

    Args:
        density_estimator: DensityEstimator subclass.
        input_dim: Dimensionality of the input.
        context_dim: Dimensionality of the context.
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

    # Loss is only required to work for batched inputs and contexts
    loss = estimator.loss(batch_input, batch_context)
    assert loss.shape == (
        nsamples,
    ), f"Loss shape is not correct. It is of shape {loss.shape}, but should be {(nsamples, )}"

    # Same for log_prob
    log_probs = estimator.log_prob(batch_input, batch_context)
    assert log_probs.shape == (
        nsamples,
    ), f"log_prob shape is not correct. It is of shape {log_probs.shape}, but should be {(nsamples, )}"

    # Sample and log_prob should work for batched and unbatched contexts

    # Unbatched context
    samples = estimator.sample((nsamples_test,), batch_context[0])
    assert samples.shape == (
        nsamples_test,
        input_dim,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should be {(nsamples_test, input_dim)}"

    samples = estimator.sample((1, nsamples_test), batch_context[0])
    assert samples.shape == (
        1,
        nsamples_test,
        input_dim,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should be {(1, nsamples_test, input_dim)}"

    samples = estimator.sample((2, nsamples_test), batch_context[0])
    assert samples.shape == (
        2,
        nsamples_test,
        input_dim,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should be {(batch_context.shape[0], nsamples_test, input_dim)}"

    # Batched context
    samples = estimator.sample((nsamples_test,), batch_context)
    assert samples.shape == (
        batch_context.shape[0],
        nsamples_test,
        input_dim,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should be {(batch_context.shape[0], nsamples_test, input_dim)}"

    samples = estimator.sample((nsamples_test,), batch_context[0].unsqueeze(0))
    assert samples.shape == (
        1,
        nsamples_test,
        input_dim,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should be {(batch_context.shape[0], nsamples_test, input_dim)}"

    # Both batched
    samples = estimator.sample((2, nsamples_test), batch_context.unsqueeze(0))
    assert samples.shape == (
        1,
        batch_context.shape[0],
        2,
        nsamples_test,
        input_dim,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should be {(1,batch_context.shape[0],2,nsamples_test,input_dim)}"

    # Sample and log_prob work for batched and unbatched contexts
    samples, log_probs = estimator.sample_and_log_prob(
        (nsamples_test,), batch_context[0]
    )
    assert samples.shape == (
        nsamples_test,
        input_dim,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should be {(nsamples_test, input_dim)}"
    assert log_probs.shape == (
        nsamples_test,
    ), f"log_prob shape is not correct. It is of shape {log_probs.shape}, but should be {(nsamples_test, )}"

    samples, log_probs = estimator.sample_and_log_prob((nsamples_test,), batch_context)

    assert samples.shape == (
        batch_context.shape[0],
        nsamples_test,
        input_dim,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should be {(batch_context.shape[0], nsamples_test, input_dim)}"
    assert log_probs.shape == (
        batch_context.shape[0],
        nsamples_test,
    ), f"log_prob shape is not correct. It is of shape {log_probs.shape}, but should be {(batch_context.shape[0], nsamples_test)}"

    samples, log_probs = estimator.sample_and_log_prob(
        (
            2,
            nsamples_test,
        ),
        batch_context.unsqueeze(0),
    )
    assert samples.shape == (
        1,
        batch_context.shape[0],
        2,
        nsamples_test,
        input_dim,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should be {(1,batch_context.shape[0],2,nsamples_test,input_dim)}"
    assert log_probs.shape == (
        1,
        batch_context.shape[0],
        2,
        nsamples_test,
    ), f"log_prob shape is not correct. It is of shape {log_probs.shape}, but should be {(1,batch_context.shape[0],2,nsamples_test)}"
