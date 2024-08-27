# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets.net_builders import build_score_estimator


@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize("input_sample_dim", (1, 2))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("score_net", ["mlp", "ada_mlp"])
def test_score_estimator_loss_shapes(
    sde_type,
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
    score_net,
):
    """Test whether `loss` of DensityEstimators follow the shape convention."""
    score_estimator, inputs, conditions = _build_score_estimator_and_tensors(
        sde_type,
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        score_net=score_net,
    )

    losses = score_estimator.loss(inputs[0], condition=conditions)
    assert losses.shape == (batch_dim,)


@pytest.mark.gpu
@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_score_estimator_on_device(sde_type, device):
    """Test whether DensityEstimators can be moved to the device."""
    score_estimator = build_score_estimator(
        torch.randn(100, 1), torch.randn(100, 1), sde_type=sde_type
    )
    score_estimator.to(device)

    # Test forward
    inputs = torch.randn(100, 1, device=device)
    condition = torch.randn(100, 1, device=device)
    time = torch.randn(1, device=device)
    out = score_estimator(inputs, condition, time)

    assert str(out.device).split(":")[0] == device, "Output device mismatch."

    # Test loss
    loss = score_estimator.loss(inputs, condition)
    assert str(loss.device).split(":")[0] == device, "Loss device mismatch."


@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize("input_sample_dim", (1, 2))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("score_net", ["mlp", "ada_mlp"])
def test_score_estimator_forward_shapes(
    sde_type,
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
    score_net,
):
    """Test whether `forward` of DensityEstimators follow the shape convention."""
    score_estimator, inputs, conditions = _build_score_estimator_and_tensors(
        sde_type,
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        score_net=score_net,
    )
    # Batched times
    times = torch.rand((batch_dim,))
    outputs = score_estimator(inputs[0], condition=conditions, time=times)
    assert outputs.shape == (batch_dim, *input_event_shape), "Output shape mismatch."

    # Single time
    time = torch.rand(())
    outputs = score_estimator(inputs[0], condition=conditions, time=time)
    assert outputs.shape == (batch_dim, *input_event_shape), "Output shape mismatch."


def _build_score_estimator_and_tensors(
    sde_type: str,
    input_event_shape: Tuple[int],
    condition_event_shape: Tuple[int],
    batch_dim: int,
    input_sample_dim: int = 1,
    **kwargs,
):
    """Helper function for all tests that deal with shapes of density estimators."""

    # Use discrete thetas such that categorical density esitmators can also use them.
    building_thetas = torch.randint(
        0, 4, (1000, *input_event_shape), dtype=torch.float32
    )
    building_xs = torch.randn((1000, *condition_event_shape))

    if len(condition_event_shape) > 1:
        embedding_net_y = CNNEmbedding(condition_event_shape, kernel_size=1)
    else:
        embedding_net_y = torch.nn.Identity()

    if len(input_event_shape) > 1:
        embedding_net_x = CNNEmbedding(input_event_shape, kernel_size=1)
    else:
        embedding_net_x = torch.nn.Identity()

    score_estimator = build_score_estimator(
        torch.randn_like(building_thetas),
        torch.randn_like(building_xs),
        sde_type=sde_type,
        embedding_net_x=embedding_net_x,
        embedding_net_y=embedding_net_y,
        **kwargs,
    )

    inputs = building_thetas[:batch_dim]
    condition = building_xs[:batch_dim]

    inputs = inputs.unsqueeze(0)
    inputs = inputs.expand(
        [
            input_sample_dim,
        ]
        + [-1] * (1 + len(input_event_shape))
    )
    condition = condition
    return score_estimator, inputs, condition
