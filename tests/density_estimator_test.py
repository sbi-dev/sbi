# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
from sbi.neural_nets.density_estimators.nflows_flow import NFlowsFlow
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.neural_nets.density_estimators.zuko_flow import ZukoFlow
from sbi.neural_nets.density_estimators.zuko_flow_estimator import ZukoFlowMatchingEstimator
from sbi.neural_nets.flow import build_nsf, build_zuko_maf, build_zuko_flow_matching


@pytest.mark.parametrize("density_estimator", (NFlowsFlow, ZukoFlow, ZukoFlowMatchingEstimator))
@pytest.mark.parametrize("input_dims", (1, 2))
@pytest.mark.parametrize(
    "condition_shape", ((1,), (2,), (1, 1), (2, 2), (1, 1, 1), (2, 2, 2))
)
def test_api_density_estimator(density_estimator, input_dims, condition_shape):
    r"""Checks whether we can evaluate and sample from density estimators correctly.

    Args:
        nsamples (int): The number of samples to generate.
        input_dims (int): The dimensionality of the input samples.

    Returns:
        torch.Tensor: A tensor of shape (nsamples, input_dims)
        containing the generated samples.
    """
    input_mvn = MultivariateNormal(
        loc=zeros(input_dims), covariance_matrix=eye(input_dims)
    )
    return input_mvn.sample((nsamples,))


def get_batch_context(nsamples: int, condition_shape: tuple[int, ...]) -> torch.Tensor:
    r"""Generate a batch of context samples from a multivariate normal distribution.

    Args:
        nsamples (int): The number of context samples to generate.
        condition_shape (tuple[int, ...]): The shape of the condition for each sample.

    Returns:
        torch.Tensor: A tensor containing the generated context samples.
    """
    context_mvn = MultivariateNormal(
        loc=zeros(*condition_shape), covariance_matrix=eye(condition_shape[-1])
    )
    return context_mvn.sample((nsamples,))

    class EmbeddingNet(torch.nn.Module):
        def forward(self, x):
            for _ in range(len(condition_shape) - 1):
                x = torch.sum(x, dim=-1)
            return x

    if density_estimator == NFlowsFlow:
        estimator = build_nsf(
            batch_input,
            batch_context,
            hidden_features=10,
            num_transforms=2,
            embedding_net=EmbeddingNet(),
        )
    elif density_estimator == ZukoFlow:
        estimator = build_zuko_maf(
            batch_input,
            batch_context,
            hidden_features=10,
            num_transforms=2,
            embedding_net=EmbeddingNet(),
        )
    elif density_estimator == ZukoFlowMatchingEstimator:
        estimator = build_zuko_flow_matching(
            batch_input,
            batch_context,
            hidden_features=10,
            num_transforms=2
        )

    # Loss is only required to work for batched inputs and contexts
    loss = estimator.loss(batch_input, batch_context)
    print(density_estimator, loss)
    assert loss.shape == (
        nsamples,
    ), f"Loss shape is not correct. It is of shape {loss.shape}, but should \
        be {(nsamples,)}"

    # Sample and log_prob should work for batched and unbatched contexts

    # Unbatched context
    samples = estimator.sample((nsamples_test,), batch_context[0])
    assert samples.shape == (
        nsamples_test,
        input_dims,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should \
        be {(nsamples_test, input_dims)}"
    log_probs = estimator.log_prob(samples, batch_context[0])
    assert log_probs.shape == (
        nsamples_test,
    ), f"log_prob shape is not correct. It is of shape {log_probs.shape}, but should \
        be {(nsamples_test,)}"

    samples = estimator.sample((1, nsamples_test), batch_context[0])
    assert samples.shape == (
        1,
        nsamples_test,
        input_dims,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should \
        be {(1, nsamples_test, input_dims)}"
    log_probs = estimator.log_prob(samples, batch_context[0])
    assert log_probs.shape == (
        1,
        nsamples_test,
    ), f"log_prob shape is not correct. It is of shape {log_probs.shape}, but should \
        be {(1, nsamples_test)}"

    samples = estimator.sample((2, nsamples_test), batch_context[0])
    assert samples.shape == (
        2,
        nsamples_test,
        input_dims,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should \
        be {(batch_context.shape[0], nsamples_test, input_dims)}"
    log_probs = estimator.log_prob(samples, batch_context[0])
    assert log_probs.shape == (
        2,
        nsamples_test,
    ), f"log_prob shape is not correct. It is of shape {log_probs.shape}, but should \
        be {(batch_context.shape[0], nsamples_test)}"

    # Batched context
    samples = estimator.sample((nsamples_test,), batch_context)
    assert samples.shape == (
        batch_context.shape[0],
        nsamples_test,
        input_dims,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should \
        be {(batch_context.shape[0], nsamples_test, input_dims)}"
    try:
        log_probs = estimator.log_prob(samples, batch_context)
    except RuntimeError:
        # Shapes (10,) and (5,) are not broadcastable, so we expect a ValueError
        pass
    except Exception as err:
        raise AssertionError(f"Expected RuntimeError as shapes {batch_context.shape} \
                             and {samples.shape} are not broadcastable, but got a \
                             different/no error.") from err

    samples = estimator.sample((nsamples_test,), batch_context[0].unsqueeze(0))
    assert samples.shape == (
        1,
        nsamples_test,
        input_dims,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should be \
        {(batch_context.shape[0], nsamples_test, input_dims)}"
    log_probs = estimator.log_prob(samples, batch_context[0].unsqueeze(0))
    assert log_probs.shape == (
        1,
        nsamples_test,
    ), f"log_prob shape is not correct. It is of shape {log_probs.shape}, but should \
        be {(batch_context.shape[0], nsamples_test)}"

    # Both batched
    samples = estimator.sample((2, nsamples_test), batch_context.unsqueeze(0))
    assert samples.shape == (
        1,
        batch_context.shape[0],
        2,
        nsamples_test,
        input_dims,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should \
        be {(1, batch_context.shape[0], 2, nsamples_test, input_dims)}"
    try:
        log_probs = estimator.log_prob(samples, batch_context.unsqueeze(0))
    except RuntimeError:
        # Shapes (10,) and (5,) are not broadcastable, so we expect a ValueError
        pass
    except Exception as err:
        raise AssertionError(f"Expected RuntimeError as shapes {batch_context.shape} \
                            and {samples.shape} are not broadcastable, but got a \
                            different/no error.") from err

    # Sample and log_prob work for batched and unbatched contexts
    samples, log_probs = estimator.sample_and_log_prob(
        (nsamples_test,), batch_context[0]
    )
    assert samples.shape == (
        nsamples_test,
        input_dims,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should \
        be {(nsamples_test, input_dims)}"
    assert log_probs.shape == (
        nsamples_test,
    ), f"log_prob shape is not correct. It is of shape {log_probs.shape}, but should \
        be {(nsamples_test,)}"

    samples, log_probs = estimator.sample_and_log_prob((nsamples_test,), batch_context)

    assert samples.shape == (
        batch_context.shape[0],
        nsamples_test,
        input_dims,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should \
        be {(batch_context.shape[0], nsamples_test, input_dims)}"
    assert log_probs.shape == (
        batch_context.shape[0],
        nsamples_test,
    ), f"log_prob shape is not correct. It is of shape {log_probs.shape}, but should \
        be {(batch_context.shape[0], nsamples_test)}"

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
        input_dims,
    ), f"Samples shape is not correct. It is of shape {samples.shape}, but should \
        be {(1, batch_context.shape[0], 2, nsamples_test, input_dims)}"
    assert log_probs.shape == (
        1,
        batch_context.shape[0],
        2,
        nsamples_test,
    ), f"log_prob shape is not correct. It is of shape {log_probs.shape}, but should \
        be {(1, batch_context.shape[0], 2, nsamples_test)}"
