# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Optional, Tuple

import pytest
import torch

from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets.net_builders import (
    build_flow_matching_estimator,
    build_masked_score_matching_estimator,
    build_score_matching_estimator,
)


@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,), (3, 3)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize(
    "estimator_type,sde_type,net",
    [
        ("flow", None, "mlp"),  # Flow matching doesn't use sde_type
        ("score", "vp", "mlp"),
        ("score", "subvp", "mlp"),
        ("score", "ve", "mlp"),
        ("masked-score", "ve", "simformer"),
    ],
)
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
    (
        estimator,
        inputs,
        conditions,
        condition_masks,
        edge_masks,
    ) = _build_vector_field_estimator_and_tensors(
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        estimator_type=estimator_type,
        sde_type=sde_type,
        net=net,
    )

    if estimator_type == "masked-score":
        losses = estimator.loss(
            inputs[0], condition_mask=condition_masks, edge_mask=edge_masks
        )
        losses = losses
    else:
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
        ("masked-score", "ve"),
    ],
)
def test_vector_field_estimator_on_device(device, estimator_type, sde_type):
    """Test whether vector field estimators can be moved to the device."""
    if estimator_type == "flow":
        estimator = build_flow_matching_estimator(
            torch.randn(100, 1), torch.randn(100, 1)
        )
    elif estimator_type == "score":
        estimator = build_score_matching_estimator(
            torch.randn(100, 1), torch.randn(100, 1), sde_type=sde_type
        )
    elif estimator_type == "masked-score":
        estimator = build_masked_score_matching_estimator(
            torch.randn(100, 5, 1),
            torch.randn(100, 5, 1),
            sde_type=sde_type,
            net="simformer",
        )

    estimator.to(device)

    time = torch.randn(1, device=device)

    if estimator_type == "masked-score":
        inputs = torch.randn(100, 5, 1, device=device)
        condition_masks = torch.ones(100, 5, device=device)
        edge_masks = torch.ones(100, 5, 5, device=device)
        out = estimator(inputs, time, condition_masks, edge_masks)
        loss = estimator.loss(inputs, condition_masks, edge_masks)
    else:
        # Test forward
        inputs = torch.randn(100, 1, device=device)
        condition = torch.randn(100, 1, device=device)
        out = estimator(inputs, condition, time)
        # Test loss
        loss = estimator.loss(inputs, condition)

    assert str(out.device).split(":")[0] == device, "Output device mismatch."
    assert str(loss.device).split(":")[0] == device, "Loss device mismatch."


@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,), (3, 3)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize(
    "estimator_type,sde_type,net",
    [
        ("flow", None, "mlp"),  # Flow matching doesn't use sde_type
        ("score", "vp", "mlp"),
        ("score", "subvp", "mlp"),
        ("score", "ve", "mlp"),
        ("masked-score", "ve", "simformer"),
    ],
)
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
    (
        estimator,
        inputs,
        conditions,
        condition_masks,
        edge_masks,
    ) = _build_vector_field_estimator_and_tensors(
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        estimator_type=estimator_type,
        sde_type=sde_type,
        net=net,
    )
    inputs_for_forward = inputs[0]

    # Batched times
    times = torch.rand((batch_dim,))
    if estimator_type == "masked-score":
        outputs = estimator(
            inputs_for_forward,
            time=times,
            condition_mask=condition_masks,
            edge_mask=edge_masks,
        )
        assert outputs.shape == inputs_for_forward.shape, "Output shape mismatch."
    else:
        outputs = estimator(inputs_for_forward, condition=conditions, time=times)
        assert outputs.shape == (
            batch_dim,
            *input_event_shape,
        ), "Output shape mismatch."

    # Single time
    time = torch.rand(())
    if estimator_type == "masked-score":
        outputs = estimator(
            inputs_for_forward,
            time=time,
            condition_mask=condition_masks,
            edge_mask=edge_masks,
        )
        assert outputs.shape == inputs_for_forward.shape, "Output shape mismatch."
    else:
        outputs = estimator(inputs_for_forward, condition=conditions, time=time)
        assert outputs.shape == (
            batch_dim,
            *input_event_shape,
        ), "Output shape mismatch."


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
    # Use discrete thetas such that categorical density esitmators can also use them.
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
    elif estimator_type == "score":
        estimator = build_score_matching_estimator(
            torch.randn_like(building_thetas),
            torch.randn_like(building_xs),
            embedding_net=embedding_net,
            sde_type=sde_type,
            **kwargs,
        )
    elif estimator_type == "masked-score":
        # For masked estimators, the input is different.
        # The builder expects batch_x to have the shape of the combined inputs
        # Let's assume num_features = 1 for simplicity.
        num_nodes = input_event_shape[0]
        num_features = 1
        building_inputs = building_thetas.reshape(100, num_nodes, num_features)

        estimator = build_masked_score_matching_estimator(
            building_inputs,
            building_inputs,  # y is not used for simformer
            embedding_net=embedding_net,
            sde_type=sde_type,
            **kwargs,
        )

    if estimator_type == "masked-score":
        num_nodes = input_event_shape[0]
        num_features = 1
        inputs = building_thetas[:batch_dim].reshape(batch_dim, num_nodes, num_features)
        condition_masks = torch.bernoulli(torch.rand(batch_dim, num_nodes))
        edge_masks = torch.ones(batch_dim, num_nodes, num_nodes)
        condition = None
        inputs = inputs.unsqueeze(0).expand(
            [input_sample_dim] + [-1] * (len(inputs.shape))
        )
        return estimator, inputs, condition, condition_masks, edge_masks
    else:
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
        return estimator, inputs, condition, None, None
