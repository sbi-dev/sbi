# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Tuple

import pytest
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.neural_nets.density_estimators.shape_handling import reshape_to_iid_batch_event
from sbi.neural_nets.flow import (
    build_maf,
    build_maf_rqs,
    build_nsf,
    build_zuko_bpf,
    build_zuko_gf,
    build_zuko_maf,
    build_zuko_naf,
    build_zuko_ncsf,
    build_zuko_nice,
    build_zuko_nsf,
    build_zuko_sospf,
    build_zuko_unaf,
)


def get_batch_input(nsamples: int, input_dims: int) -> torch.Tensor:
    r"""Generate a batch of input samples from a multivariate normal distribution.

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


@pytest.mark.parametrize(
    "build_density_estimator",
    (
        build_maf,
        build_maf_rqs,
        build_nsf,
        build_zuko_nice,
        build_zuko_maf,
        build_zuko_nsf,
        build_zuko_ncsf,
        build_zuko_sospf,
        build_zuko_naf,
        build_zuko_unaf,
        build_zuko_gf,
        build_zuko_bpf,
    ),
)
@pytest.mark.parametrize("input_dims", (1, 2))
@pytest.mark.parametrize(
    "condition_shape", ((1,), (2,), (1, 1), (2, 2), (1, 1, 1), (2, 2, 2))
)
def test_api_density_estimator(build_density_estimator, input_dims, condition_shape):
    r"""Checks whether we can evaluate and sample from density estimators correctly.

    Args:
        build_density_estimator: function that creates a DensityEstimator subclass.
        input_dim: Dimensionality of the input.
        context_shape: Dimensionality of the context.
    """

    nsamples = 10
    nsamples_test = 5

    batch_input = get_batch_input(nsamples, input_dims)
    batch_context = get_batch_context(nsamples, condition_shape)

    class EmbeddingNet(torch.nn.Module):
        def forward(self, x):
            for _ in range(len(condition_shape) - 1):
                x = torch.sum(x, dim=-1)
            return x

    estimator = build_density_estimator(
        batch_input,
        batch_context,
        hidden_features=10,
        num_transforms=2,
        embedding_net=EmbeddingNet(),
    )

    # Loss is only required to work for batched inputs and contexts
    loss = estimator.loss(batch_input, batch_context)
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
        raise AssertionError(
            f"Expected RuntimeError as shapes {batch_context.shape} \
                             and {samples.shape} are not broadcastable, but got a \
                             different/no error."
        ) from err

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
        raise AssertionError(
            f"Expected RuntimeError as shapes {batch_context.shape} \
                            and {samples.shape} are not broadcastable, but got a \
                            different/no error."
        ) from err

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


@pytest.mark.parametrize(
    "theta_or_x_shape, target_shape, event_shape, leading_is_iid",
    (
        ((3,), (1, 1, 3), (3,), False),
        ((3,), (1, 1, 3), (3,), True),
        ((1, 3), (1, 1, 3), (3,), False),
        ((1, 3), (1, 1, 3), (3,), True),
        ((2, 3), (1, 2, 3), (3,), False),
        ((2, 3), (2, 1, 3), (3,), True),
        ((1, 2, 3), (1, 2, 3), (3,), True),
        ((1, 2, 3), (1, 2, 3), (3,), False),
        ((3, 5), (1, 1, 3, 5), (3, 5), False),
        ((3, 5), (1, 1, 3, 5), (3, 5), True),
        ((1, 3, 5), (1, 1, 3, 5), (3, 5), False),
        ((1, 3, 5), (1, 1, 3, 5), (3, 5), True),
        ((2, 3, 5), (1, 2, 3, 5), (3, 5), False),
        ((2, 3, 5), (2, 1, 3, 5), (3, 5), True),
        ((1, 2, 3, 5), (1, 2, 3, 5), (3, 5), True),
        ((1, 2, 3, 5), (1, 2, 3, 5), (3, 5), False),
        pytest.param((1, 2, 3, 5), (1, 2, 3, 5), (5), False, marks=pytest.mark.xfail),
        pytest.param((1, 2, 3, 5), (1, 2, 3, 5), (3), False, marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), (1, 2, 3), (1, 5), False, marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), (1, 2, 3), (1, 3), False, marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), (2, 1, 3), (3), False, marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), (2, 1, 3), (3), True, marks=pytest.mark.xfail),
    ),
)
def test_shape_handling_utility_for_density_estimator(
    theta_or_x_shape: Tuple,
    target_shape: Tuple,
    event_shape: Tuple,
    leading_is_iid: bool,
):
    """Test whether `reshape_to_batch_iid_event` results in expected outputs."""
    input = torch.randn(theta_or_x_shape)
    output = reshape_to_iid_batch_event(
        input, event_shape=event_shape, leading_is_iid=leading_is_iid
    )
    assert output.shape == target_shape, (
        f"Shapes of Output ({output.shape}) and target shape ({target_shape}) do not "
        f"match."
    )
