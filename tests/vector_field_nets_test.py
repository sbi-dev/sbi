# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn as nn

from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets.net_builders.vector_field_nets import (
    VectorFieldMLP,
    VectorFieldTransformer,
    build_flow_matching_estimator,
    build_standard_mlp_network,
    build_score_matching_estimator,
    build_transformer_network,
)


@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_mlp_network_shapes(input_event_shape, condition_event_shape, batch_dim):
    """Test whether MLP vector field networks produce correct output shapes."""
    network, inputs, conditions = _build_vector_field_components(
        "mlp", input_event_shape, condition_event_shape, batch_dim
    )

    # Create time parameter
    t = torch.rand((batch_dim,))

    # Test forward pass
    outputs = network(inputs, conditions, t)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"


@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_transformer_network_shapes(
    input_event_shape, condition_event_shape, batch_dim
):
    """Test whether transformer vector field networks produce correct output shapes."""
    network, inputs, conditions = _build_vector_field_components(
        "transformer", input_event_shape, condition_event_shape, batch_dim
    )

    # Create time parameter
    t = torch.rand((batch_dim,))

    # Test forward pass
    outputs = network(inputs, conditions, t)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"


@pytest.mark.parametrize("net_type", ["mlp", "transformer"])
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_flow_matching_estimator_shapes(
    net_type, input_event_shape, condition_event_shape, batch_dim
):
    """Test whether flow matching estimator produces correct output shapes."""
    # Build the estimator
    flow_matching_estimator, inputs, conditions = _build_estimator_and_tensors(
        "flow", input_event_shape, condition_event_shape, batch_dim, net=net_type
    )

    # Create time parameter
    t = torch.rand((batch_dim,))

    # Test forward pass
    outputs = flow_matching_estimator(inputs, conditions, t)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"

    # Test loss
    losses = flow_matching_estimator.loss(inputs, condition=conditions)
    assert losses.shape == (batch_dim,), "Loss shape mismatch"


@pytest.mark.parametrize("net_type", ["mlp", "transformer"])
@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_score_matching_estimator_shapes(
    net_type, sde_type, input_event_shape, condition_event_shape, batch_dim
):
    """Test whether score matching estimator produces correct output shapes."""
    # Build the estimator
    score_matching_estimator, inputs, conditions = _build_estimator_and_tensors(
        "score",
        input_event_shape,
        condition_event_shape,
        batch_dim,
        net=net_type,
        sde_type=sde_type,
    )

    # Create time parameter
    t = torch.rand((batch_dim,))

    # Test forward pass
    outputs = score_matching_estimator(inputs, conditions, t)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"

    # Test loss
    losses = score_matching_estimator.loss(inputs, condition=conditions)
    assert losses.shape == (batch_dim,), "Loss shape mismatch"


@pytest.mark.gpu
@pytest.mark.parametrize("net_type", ["mlp", "transformer"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_flow_matching_estimator_on_device(net_type, device):
    """Test whether flow matching estimator can be moved to the device."""
    flow_estimator = build_flow_matching_estimator(
        batch_x=torch.randn(100, 1), batch_y=torch.randn(100, 1), net=net_type
    )
    flow_estimator.to(device)

    # Test forward
    inputs = torch.randn(100, 1, device=device)
    condition = torch.randn(100, 1, device=device)
    t = torch.randn(100, device=device)
    out = flow_estimator(inputs, condition, t)

    assert str(out.device).split(":")[0] == device, "Output device mismatch."

    # Test loss
    loss = flow_estimator.loss(inputs, condition)
    assert str(loss.device).split(":")[0] == device, "Loss device mismatch."


@pytest.mark.gpu
@pytest.mark.parametrize("net_type", ["mlp", "transformer"])
@pytest.mark.parametrize("sde_type", ["vp"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_score_matching_estimator_on_device(net_type, sde_type, device):
    """Test whether score matching estimator can be moved to the device."""
    score_estimator = build_score_matching_estimator(
        batch_x=torch.randn(100, 1),
        batch_y=torch.randn(100, 1),
        net=net_type,
        sde_type=sde_type,
    )
    score_estimator.to(device)

    # Test forward
    inputs = torch.randn(100, 1, device=device)
    condition = torch.randn(100, 1, device=device)
    t = torch.randn(100, device=device)
    out = score_estimator(inputs, condition, t)

    assert str(out.device).split(":")[0] == device, "Output device mismatch."

    # Test loss
    loss = score_estimator.loss(inputs, condition)
    assert str(loss.device).split(":")[0] == device, "Loss device mismatch."


@pytest.mark.parametrize("hidden_features", [50, 100, [50, 100]])
@pytest.mark.parametrize("num_layers", [2, 5])
def test_mlp_network_parameters(hidden_features, num_layers):
    """Test whether MLP vector field networks can be built with different parameters."""
    batch_x = torch.randn(100, 5)
    batch_y = torch.randn(100, 3)

    network = build_standard_mlp_network(
        batch_x=batch_x,
        batch_y=batch_y,
        hidden_features=hidden_features,
        num_layers=num_layers,
        time_embedding_dim=32,
    )

    # Verify it's the correct type
    assert isinstance(network, VectorFieldMLP), "Should be a VectorFieldMLP instance"

    # Test a forward pass to ensure it works
    inputs = torch.randn(10, 5)
    conditions = torch.randn(10, 3)
    times = torch.rand(10)

    outputs = network(inputs, conditions, times)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"


@pytest.mark.parametrize("hidden_features", [64, 128])
@pytest.mark.parametrize("num_blocks", [2, 4])
@pytest.mark.parametrize("num_heads", [4, 8])
def test_transformer_network_parameters(hidden_features, num_blocks, num_heads):
    """Test whether transformer vector field networks can be built with different
    parameters."""
    batch_x = torch.randn(100, 5)
    batch_y = torch.randn(100, 3)

    network = build_transformer_network(
        batch_x=batch_x,
        batch_y=batch_y,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        num_heads=num_heads,
        time_embedding_dim=32,
    )

    # Verify it's the correct type
    assert isinstance(network, VectorFieldTransformer), (
        "Should be a VectorFieldTransformer instance"
    )

    # Test a forward pass to ensure it works
    inputs = torch.randn(10, 5)
    conditions = torch.randn(10, 3)
    times = torch.rand(10)

    outputs = network(inputs, conditions, times)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"


def _build_vector_field_components(
    net_type: str,
    input_event_shape: Tuple[int],
    condition_event_shape: Tuple[int],
    batch_dim: int,
):
    """Helper function to build vector field components for testing."""
    # Create example inputs and conditions
    inputs = torch.randn(batch_dim, *input_event_shape)
    conditions = torch.randn(batch_dim, *condition_event_shape)

    # Create batches for building networks
    batch_x = torch.randn(100, *input_event_shape)
    batch_y = torch.randn(100, *condition_event_shape)

    # Setup embedding net if needed
    if len(condition_event_shape) > 1:
        embedding_net = CNNEmbedding(condition_event_shape, kernel_size=1)
    else:
        embedding_net = nn.Identity()

    # Build the appropriate network
    if net_type == "mlp":
        network = build_standard_mlp_network(
            batch_x=batch_x,
            batch_y=batch_y,
            hidden_features=64,
            time_embedding_dim=32,
            embedding_net=embedding_net,
        )
    elif net_type == "transformer":
        network = build_transformer_network(
            batch_x=batch_x,
            batch_y=batch_y,
            hidden_features=64,
            num_blocks=3,
            num_heads=4,
            time_embedding_dim=32,
            embedding_net=embedding_net,
        )
    else:
        raise ValueError(f"Unknown network type: {net_type}")

    return network, inputs, conditions


def _build_estimator_and_tensors(
    estimator_type: str,
    input_event_shape: Tuple[int],
    condition_event_shape: Tuple[int],
    batch_dim: int,
    **kwargs,
):
    """Helper function to build estimators for testing."""
    # Create example inputs and conditions
    inputs = torch.randn(batch_dim, *input_event_shape)
    conditions = torch.randn(batch_dim, *condition_event_shape)

    # Create batches for building estimators
    batch_x = torch.randn(100, *input_event_shape)
    batch_y = torch.randn(100, *condition_event_shape)

    # Setup embedding net if needed
    if len(condition_event_shape) > 1:
        embedding_net = CNNEmbedding(condition_event_shape, kernel_size=1)
    else:
        embedding_net = nn.Identity()

    # Build the appropriate estimator
    if estimator_type == "flow":
        estimator = build_flow_matching_estimator(
            batch_x=batch_x, batch_y=batch_y, embedding_net=embedding_net, **kwargs
        )
    elif estimator_type == "score":
        estimator = build_score_matching_estimator(
            batch_x=batch_x, batch_y=batch_y, embedding_net=embedding_net, **kwargs
        )
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")

    return estimator, inputs, conditions
