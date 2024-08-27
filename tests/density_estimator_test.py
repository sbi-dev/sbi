# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Callable, Tuple

import pytest
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_sample_batch_event,
)
from sbi.neural_nets.net_builders import (
    build_categoricalmassestimator,
    build_maf,
    build_maf_rqs,
    build_mdn,
    build_mlp_flowmatcher,
    build_mnle,
    build_nsf,
    build_resnet_flowmatcher,
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

# List of all density estimator builders for testing.
model_builders = [
    build_mdn,
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
]

flowmatching_build_functions = [
    build_mlp_flowmatcher,
    build_resnet_flowmatcher,
]


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
    "theta_or_x_shape, target_shape, event_shape, leading_is_sample",
    (
        ((3,), (1, 1, 3), (3,), False),
        ((3,), (1, 1, 3), (3,), True),
        ((1, 3), (1, 1, 3), (3,), False),
        ((1, 3), (1, 1, 3), (3,), True),
        ((2, 3), (1, 2, 3), (3,), False),
        ((2, 3), (2, 1, 3), (3,), True),
        ((1, 2, 3), (1, 2, 3), (3,), True),
        ((1, 2, 3), (2, 1, 3), (3,), False),
        ((3, 5), (1, 1, 3, 5), (3, 5), False),
        ((3, 5), (1, 1, 3, 5), (3, 5), True),
        ((1, 3, 5), (1, 1, 3, 5), (3, 5), False),
        ((1, 3, 5), (1, 1, 3, 5), (3, 5), True),
        ((2, 3, 5), (1, 2, 3, 5), (3, 5), False),
        ((2, 3, 5), (2, 1, 3, 5), (3, 5), True),
        ((1, 2, 3, 5), (1, 2, 3, 5), (3, 5), True),
        ((1, 2, 3, 5), (2, 1, 3, 5), (3, 5), False),
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
    leading_is_sample: bool,
):
    """Test whether `reshape_to_batch_sample_event` results in expected outputs."""
    input = torch.randn(theta_or_x_shape)
    output = reshape_to_sample_batch_event(
        input, event_shape=event_shape, leading_is_sample=leading_is_sample
    )
    assert output.shape == target_shape, (
        f"Shapes of Output ({output.shape}) and target shape ({target_shape}) do not "
        f"match."
    )


@pytest.mark.parametrize(
    "density_estimator_build_fn", model_builders + flowmatching_build_functions
)
@pytest.mark.parametrize("input_sample_dim", (1, 2))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_density_estimator_loss_shapes(
    density_estimator_build_fn,
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
):
    """Test whether `loss` of DensityEstimators follow the shape convention."""
    density_estimator, inputs, conditions, _ = _build_density_estimator_and_tensors(
        density_estimator_build_fn,
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
    )

    losses = density_estimator.loss(inputs[0], condition=conditions)
    assert losses.shape == (batch_dim,)


@pytest.mark.parametrize("density_estimator_build_fn", model_builders)
@pytest.mark.parametrize("input_sample_dim", (1, 2))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
# This mimics the shape of a 2D image and is used for testing CNN embeddings.
@pytest.mark.parametrize("condition_event_shape", ((2, 1), (2, 7), (7, 2), (7, 7)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_density_estimator_log_prob_shapes_with_embedding(
    density_estimator_build_fn,
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
):
    """Test whether `loss` of DensityEstimators follow the shape convention."""
    density_estimator, inputs, conditions, _ = _build_density_estimator_and_tensors(
        density_estimator_build_fn,
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
    )

    log_probs = density_estimator.log_prob(inputs, condition=conditions)
    assert log_probs.shape == (input_sample_dim, batch_dim)


@pytest.mark.parametrize("density_estimator_build_fn", model_builders)
@pytest.mark.parametrize("sample_shape", ((), (1,), (2, 3)))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_density_estimator_sample_shapes(
    density_estimator_build_fn,
    sample_shape,
    input_event_shape,
    condition_event_shape,
    batch_dim,
):
    """Test whether `sample` of DensityEstimators follow the shape
    convention."""
    density_estimator, _, conditions, input_event_shape = (
        _build_density_estimator_and_tensors(
            density_estimator_build_fn,
            input_event_shape,
            condition_event_shape,
            batch_dim,
        )
    )
    samples = density_estimator.sample(sample_shape, condition=conditions)
    if density_estimator_build_fn == build_categoricalmassestimator:
        # Our categorical is always 1D and does not return `input_event_shape`.
        input_event_shape = ()
    assert samples.shape == (*sample_shape, batch_dim, *input_event_shape)


@pytest.mark.parametrize("density_estimator_build_fn", model_builders)
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_correctness_of_density_estimator_log_prob(
    density_estimator_build_fn,
    input_event_shape,
    condition_event_shape,
    batch_dim,
):
    """Test whether identical inputs lead to identical log_prob values."""
    input_sample_dim = 2
    density_estimator, inputs, condition, _ = _build_density_estimator_and_tensors(
        density_estimator_build_fn,
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
    )
    log_probs = density_estimator.log_prob(inputs, condition=condition)
    assert torch.allclose(log_probs[0, :], log_probs[1, :], rtol=1e-4)


@pytest.mark.parametrize(
    "density_estimator_build_fn", model_builders + flowmatching_build_functions
)
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("sample_shape", ((1000,), (500, 2)))
def test_correctness_of_batched_vs_seperate_sample_and_log_prob(
    density_estimator_build_fn: Callable,
    input_event_shape,
    condition_event_shape,
    sample_shape,
):
    input_sample_dim = 2
    batch_dim = 2
    density_estimator, inputs, condition, input_event_shape = (
        _build_density_estimator_and_tensors(
            density_estimator_build_fn,
            input_event_shape,
            condition_event_shape,
            batch_dim,
            input_sample_dim,
        )
    )
    # Batched vs separate sampling
    samples = density_estimator.sample(sample_shape, condition=condition)
    samples = samples.reshape(-1, batch_dim, *input_event_shape)  # Flat for comp.

    samples_separate1 = density_estimator.sample(
        (1000,), condition=condition[0][None, ...]
    )
    samples_separate2 = density_estimator.sample(
        (1000,), condition=condition[1][None, ...]
    )

    # Check if means are approx. same
    samples_m = torch.mean(samples, dim=0, dtype=torch.float32)
    samples_separate1_m = torch.mean(samples_separate1, dim=0, dtype=torch.float32)
    samples_separate2_m = torch.mean(samples_separate2, dim=0, dtype=torch.float32)
    samples_sep_m = torch.cat([samples_separate1_m, samples_separate2_m], dim=0)

    assert torch.allclose(
        samples_m, samples_sep_m, atol=0.5, rtol=0.5
    ), "Batched sampling is not consistent with separate sampling."

    # Batched vs separate log_prob
    log_probs = density_estimator.log_prob(inputs, condition=condition)

    log_probs_separate1 = density_estimator.log_prob(
        inputs[:, :1], condition=condition[0][None, ...]
    )
    log_probs_separate2 = density_estimator.log_prob(
        inputs[:, 1:], condition=condition[1][None, ...]
    )
    log_probs_sep = torch.hstack([log_probs_separate1, log_probs_separate2])

    assert torch.allclose(
        log_probs, log_probs_sep, atol=1e-2, rtol=1e-2
    ), "Batched log_prob is not consistent with separate log_prob."


def _build_density_estimator_and_tensors(
    density_estimator_build_fn: Callable,
    input_event_shape: Tuple[int],
    condition_event_shape: Tuple[int],
    batch_dim: int,
    input_sample_dim: int = 1,
):
    """Helper function for all tests that deal with shapes of density
    estimators."""

    # Use positive random values for continuous dims (log transform)
    batch_input = torch.rand((1000, *input_event_shape), dtype=torch.float32) * 10.0
    # make last dim discrete for mixed density estimators
    batch_input[:, -1] = torch.randint(0, 4, (1000,))
    batch_condition = torch.randn((1000, *condition_event_shape))
    if len(condition_event_shape) > 1:
        embedding_net = CNNEmbedding(condition_event_shape, kernel_size=1)
        z_score_y = "structured"
    else:
        embedding_net = torch.nn.Identity()
        z_score_y = "independent"

    if density_estimator_build_fn in [build_mnle, build_categoricalmassestimator]:
        density_estimator = density_estimator_build_fn(
            batch_x=batch_input,
            batch_y=batch_condition,
            embedding_net=embedding_net,
            z_score_y=z_score_y,
        )
    else:
        density_estimator = density_estimator_build_fn(
            torch.randn_like(batch_input),
            torch.randn_like(batch_condition),
            embedding_net=embedding_net,
            z_score_y=z_score_y,
        )

    inputs = batch_input[:batch_dim]
    condition = batch_condition[:batch_dim]

    inputs = inputs.unsqueeze(0)
    inputs = inputs.expand(input_sample_dim, -1, -1)
    return density_estimator, inputs, condition, input_event_shape


# separate test cases for mixed density estimation
@pytest.mark.parametrize(
    (
        "density_estimator_build_fn",
        "input_sample_dim",
        "input_event_shape",
        "condition_event_shape",
        "batch_dim",
    ),
    (
        (build_mnle, 1, (2,), (7,), 1),
        (build_mnle, 1, (2,), (7,), 10),
        # Test with 2D condition with embedding net.
        [build_mnle, 1, (2,), (7, 7), 10],
        [build_mnle, 1, (2,), (2, 7), 10],
        [build_mnle, 1, (2,), (7, 2), 10],
        [build_categoricalmassestimator, 1, (1,), (7, 7), 10],
        [build_categoricalmassestimator, 2, (1,), (7, 7), 10],
        pytest.param(
            build_mnle,
            2,
            (1,),
            (7,),
            10,
            marks=pytest.mark.xfail(
                reason="Sample dim > 1 not supported for Mixed Density Estimation"
            ),
        ),
        pytest.param(
            build_categoricalmassestimator,
            1,
            (4,),
            (7,),
            10,
            marks=pytest.mark.xfail(
                reason="Event shape > 1 not supported for categorical"
            ),
        ),
    ),
)
@pytest.mark.parametrize("sample_shape", ((10,), (1,)))
def test_mixed_density_estimator(
    density_estimator_build_fn,
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
    sample_shape,
):
    density_estimator, inputs, conditions, input_event_shape = (
        _build_density_estimator_and_tensors(
            density_estimator_build_fn,
            input_event_shape,
            condition_event_shape,
            batch_dim,
            input_sample_dim,
        )
    )
    # Test losses
    losses = density_estimator.loss(inputs[0], condition=conditions)
    assert losses.shape == (batch_dim,)

    # Test log_prob
    log_probs = density_estimator.log_prob(inputs, condition=conditions)
    assert log_probs.shape == (input_sample_dim, batch_dim)

    # Test samples
    samples = density_estimator.sample(sample_shape, condition=conditions)
    if density_estimator_build_fn == build_categoricalmassestimator:
        # Our categorical is always 1D and does not return `input_event_shape`.
        input_event_shape = ()
    assert samples.shape == (*sample_shape, batch_dim, *input_event_shape)
