# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Optional, Tuple

import pytest
import torch

from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets.net_builders import (
    build_flow_matching_estimator,
    build_score_matching_estimator,
)


@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,), (3, 3)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize(
    "estimator_type,sde_type",
    [
        ("flow", None),  # Flow matching doesn't use sde_type
        ("score", "vp"),
        ("score", "subvp"),
        ("score", "ve"),
    ],
)
@pytest.mark.parametrize("net", ["mlp"])
def test_vector_field_estimator_loss_shapes(
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
    estimator_type,
    sde_type,
    net,
):
    """Test whether `loss` of vector field estimators follow the shape convention."""
    estimator, inputs, conditions = _build_vector_field_estimator_and_tensors(
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        estimator_type=estimator_type,
        sde_type=sde_type,
        net=net,
    )

    losses = estimator.loss(inputs[0], condition=conditions)
    assert losses.shape == (batch_dim,)


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "estimator_type,sde_type",
    [
        ("flow", None),  # Flow matching doesn't use sde_type
        ("score", "vp"),
        ("score", "subvp"),
        ("score", "ve"),
    ],
)
def test_vector_field_estimator_on_device(device, estimator_type, sde_type):
    """Test whether vector field estimators can be moved to the device."""
    if estimator_type == "flow":
        estimator = build_flow_matching_estimator(
            torch.randn(100, 1), torch.randn(100, 1)
        )
    else:
        estimator = build_score_matching_estimator(
            torch.randn(100, 1), torch.randn(100, 1), sde_type=sde_type
        )
    estimator.to(device)

    # Test forward
    inputs = torch.randn(100, 1, device=device)
    condition = torch.randn(100, 1, device=device)
    time = torch.randn(1, device=device)
    out = estimator(inputs, condition, time)

    assert str(out.device).split(":")[0] == device, "Output device mismatch."

    # Test loss
    loss = estimator.loss(inputs, condition)
    assert str(loss.device).split(":")[0] == device, "Loss device mismatch."


@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,), (3, 3)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize(
    "estimator_type,sde_type",
    [
        ("flow", None),  # Flow matching doesn't use sde_type
        ("score", "vp"),
        ("score", "subvp"),
        ("score", "ve"),
    ],
)
@pytest.mark.parametrize("net", ["mlp"])
def test_vector_field_estimator_forward_shapes(
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
    estimator_type,
    sde_type,
    net,
):
    """Test whether `forward` of vector field estimators follow the shape convention."""
    estimator, inputs, conditions = _build_vector_field_estimator_and_tensors(
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        estimator_type=estimator_type,
        sde_type=sde_type,
        net=net,
    )
    # Batched times
    times = torch.rand((batch_dim,))
    outputs = estimator(inputs[0], condition=conditions, time=times)
    assert outputs.shape == (batch_dim, *input_event_shape), "Output shape mismatch."

    # Single time
    time = torch.rand(())
    outputs = estimator(inputs[0], condition=conditions, time=time)
    assert outputs.shape == (batch_dim, *input_event_shape), "Output shape mismatch."


def _build_vector_field_estimator_and_tensors(
    input_event_shape: Tuple[int],
    condition_event_shape: Tuple[int],
    batch_dim: int,
    input_sample_dim: int = 1,
    estimator_type: str = "flow",
    sde_type: Optional[str] = None,
    **kwargs,
):
    """
    Helper function for all tests that deal with shapes of vector field estimators.
    """
    # Use discrete thetas such that categorical density estimators can also use them.
    building_thetas = torch.randint(
        0, 4, (100, *input_event_shape), dtype=torch.float32
    )
    building_xs = torch.randn((100, *condition_event_shape))

    if len(condition_event_shape) > 1:
        embedding_net = CNNEmbedding(condition_event_shape, kernel_size=1)
    else:
        embedding_net = torch.nn.Identity()

    # Build the appropriate estimator based on type
    if estimator_type == "flow":
        estimator = build_flow_matching_estimator(
            torch.randn_like(building_thetas),
            torch.randn_like(building_xs),
            embedding_net=embedding_net,
            **kwargs,
        )
    else:
        estimator = build_score_matching_estimator(
            torch.randn_like(building_thetas),
            torch.randn_like(building_xs),
            embedding_net=embedding_net,
            sde_type=sde_type,
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
    return estimator, inputs, condition
