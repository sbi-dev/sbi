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
    build_adamlp_network,
    build_standard_mlp_network,
    build_transformer_network,
)


@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,), (3, 3)))
@pytest.mark.parametrize("batch_dim", (1, 5))
@pytest.mark.parametrize(
    "net_type", ["mlp", "ada_mlp", "transformer", "transformer_cross"]
)
@pytest.mark.parametrize("time_emb_type", ["sinusoidal", "random_fourier"])
def test_network_shapes(
    input_event_shape,
    condition_event_shape,
    batch_dim,
    net_type,
    time_emb_type,
):
    """Test whether vector field networks produce correct output shapes."""
    if net_type == "transformer_cross" and len(condition_event_shape) < 2:
        condition_event_shape = condition_event_shape + (1,)

    network, inputs, conditions, embedding_net = _build_vector_field_components(
        net_type,
        input_event_shape,
        condition_event_shape,
        batch_dim,
        time_emb_type=time_emb_type,
    )
    # Nets do not use the embedding net, it is used on the estimator level.
    conditions = embedding_net(conditions)

    # Create time parameter
    t = torch.rand((batch_dim,))

    # Test forward pass
    outputs = network(inputs, conditions, t)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"


@pytest.mark.parametrize("hidden_features", [50, [50, 100]])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("layer_norm", [True, False])
@pytest.mark.parametrize("skip_connections", [True, False])
@pytest.mark.parametrize("time_emb_type", ["sinusoidal", "random_fourier"])
@pytest.mark.parametrize("time_embedding_dim", [16, 32])
@pytest.mark.parametrize("batch_dim", [1, 5])
@pytest.mark.parametrize("input_dim", [1, 3])
@pytest.mark.parametrize("condition_dim", [1, 3])
def test_mlp_network_parameters(
    hidden_features,
    num_layers,
    layer_norm,
    skip_connections,
    time_emb_type,
    time_embedding_dim,
    batch_dim,
    input_dim,
    condition_dim,
):
    """Test whether MLP vector field networks can be built with different parameters."""
    batch_x = torch.randn(10, input_dim)
    batch_y = torch.randn(10, condition_dim)

    network = build_standard_mlp_network(
        batch_x=batch_x,
        batch_y=batch_y,
        hidden_features=hidden_features,
        num_layers=num_layers,
        time_embedding_dim=time_embedding_dim,
        layer_norm=layer_norm,
        skip_connections=skip_connections,
        time_emb_type=time_emb_type,
    )

    # Verify it's the correct type
    assert isinstance(network, VectorFieldMLP), "Should be a VectorFieldMLP instance"

    # Test a forward pass to ensure it works
    inputs = torch.randn(batch_dim, input_dim)
    conditions = torch.randn(batch_dim, condition_dim)
    times = torch.rand(batch_dim)

    outputs = network(inputs, conditions, times)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"


@pytest.mark.parametrize("hidden_features", [8, 32])
@pytest.mark.parametrize("num_layers", [2, 4])
@pytest.mark.parametrize("mlp_ratio", [2, 4])
@pytest.mark.parametrize("num_intermediate_mlp_layers", [0, 2])
@pytest.mark.parametrize("time_emb_type", ["sinusoidal", "random_fourier"])
@pytest.mark.parametrize("batch_dim", [1, 5])
@pytest.mark.parametrize("input_dim", [1, 3])
@pytest.mark.parametrize("condition_dim", [1, 3])
def test_adamlp_network_parameters(
    hidden_features,
    num_layers,
    mlp_ratio,
    num_intermediate_mlp_layers,
    time_emb_type,
    batch_dim,
    input_dim,
    condition_dim,
):
    """Test whether AdaMLP vector field networks can be built with different
    parameters."""
    batch_x = torch.randn(10, input_dim)
    batch_y = torch.randn(10, condition_dim)

    network = build_adamlp_network(
        batch_x=batch_x,
        batch_y=batch_y,
        hidden_features=hidden_features,
        num_layers=num_layers,
        time_embedding_dim=32,
        mlp_ratio=mlp_ratio,
        num_intermediate_mlp_layers=num_intermediate_mlp_layers,
        time_emb_type=time_emb_type,
    )

    # Test a forward pass to ensure it works
    inputs = torch.randn(batch_dim, input_dim)
    conditions = torch.randn(batch_dim, condition_dim)
    times = torch.rand(batch_dim)

    outputs = network(inputs, conditions, times)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"


@pytest.mark.parametrize("hidden_features", [64, 128])
@pytest.mark.parametrize("num_blocks", [2, 4])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("mlp_ratio", [2, 4, 8])
@pytest.mark.parametrize("time_emb_type", ["sinusoidal", "random_fourier"])
@pytest.mark.parametrize("time_embedding_dim", [16, 32])
@pytest.mark.parametrize("batch_dim", [1, 5])
@pytest.mark.parametrize("input_dim", [1, 3])
@pytest.mark.parametrize("condition_dim", [1, 3])
def test_transformer_network_parameters(
    hidden_features,
    num_blocks,
    num_heads,
    mlp_ratio,
    time_emb_type,
    time_embedding_dim,
    batch_dim,
    input_dim,
    condition_dim,
):
    """Test whether transformer vector field networks can be built with different
    parameters."""
    batch_x = torch.randn(100, input_dim)
    batch_y = torch.randn(100, condition_dim)

    network = build_transformer_network(
        batch_x=batch_x,
        batch_y=batch_y,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        time_embedding_dim=time_embedding_dim,
        time_emb_type=time_emb_type,
    )

    # Verify it's the correct type
    assert isinstance(network, VectorFieldTransformer), (
        "Should be a VectorFieldTransformer instance"
    )

    # Test a forward pass to ensure it works
    inputs = torch.randn(batch_dim, input_dim)
    conditions = torch.randn(batch_dim, condition_dim)
    times = torch.rand(batch_dim)

    outputs = network(inputs, conditions, times)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"


@pytest.mark.parametrize("hidden_features", [64, 128])
@pytest.mark.parametrize("num_blocks", [2, 4])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("mlp_ratio", [2, 4, 8])
@pytest.mark.parametrize("time_emb_type", ["sinusoidal", "random_fourier"])
@pytest.mark.parametrize("time_embedding_dim", [16, 32])
@pytest.mark.parametrize("batch_dim", [1, 5])
@pytest.mark.parametrize("input_dim", [1, 3])
@pytest.mark.parametrize("condition_dim", [1, 3])
def test_transformer_cross_attention_parameters(
    hidden_features,
    num_blocks,
    num_heads,
    mlp_ratio,
    time_emb_type,
    time_embedding_dim,
    batch_dim,
    input_dim,
    condition_dim,
):
    """Test whether transformer with cross attention can be built with different
    parameters."""
    batch_x = torch.randn(100, input_dim)
    batch_y = torch.randn(100, condition_dim)

    network = build_transformer_network(
        batch_x=batch_x,
        batch_y=batch_y,
        hidden_features=hidden_features,
        num_layers=num_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        time_embedding_dim=time_embedding_dim,
        time_emb_type=time_emb_type,
        is_x_emb_seq=True,
    )

    # Verify it's the correct type
    assert isinstance(network, VectorFieldTransformer), (
        "Should be a VectorFieldTransformer instance"
    )

    # Test a forward pass to ensure it works
    inputs = torch.randn(batch_dim, input_dim)
    conditions = torch.randn(batch_dim, condition_dim)
    times = torch.rand(batch_dim)

    outputs = network(inputs, conditions, times)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"


def _build_vector_field_components(
    net_type: str,
    input_event_shape: Tuple[int],
    condition_event_shape: Tuple[int],
    batch_dim: int,
    time_emb_type: str = "sinusoidal",
    activation: nn.Module = nn.GELU,
):
    """Helper function to build vector field components for testing."""
    # Create example inputs and conditions
    inputs = torch.randn(batch_dim, *input_event_shape)
    conditions = torch.randn(batch_dim, *condition_event_shape)


    # Setup embedding net if needed
    if len(condition_event_shape) > 1 and net_type != "transformer_cross":
        embedding_net = CNNEmbedding(condition_event_shape, kernel_size=1)
    else:
        embedding_net = nn.Identity()

    # Build the appropriate network
    if net_type == "mlp":
        network = build_standard_mlp_network(
            batch_x=inputs,
            batch_y=conditions,
            hidden_features=64,
            time_embedding_dim=32,
            embedding_net=embedding_net,
            time_emb_type=time_emb_type,
            activation=activation,
        )
    elif net_type == "ada_mlp":
        network = build_adamlp_network(
            batch_x=inputs,
            batch_y=conditions,
            hidden_features=64,
            time_embedding_dim=32,
            embedding_net=embedding_net,
            time_emb_type=time_emb_type,
        )
    elif net_type == "transformer":
        network = build_transformer_network(
            batch_x=inputs,
            batch_y=conditions,
            hidden_features=64,
            num_layers=3,
            num_heads=4,
            time_embedding_dim=32,
            embedding_net=embedding_net,
            time_emb_type=time_emb_type,
        )
    elif net_type == "transformer_cross":
        # NOTE: This needs a sequence of conditions
        network = build_transformer_network(
            batch_x=inputs,
            batch_y=conditions,
            hidden_features=64,
            num_layers=3,
            num_heads=4,
            time_embedding_dim=32,
            embedding_net=embedding_net,
            time_emb_type=time_emb_type,
            is_x_emb_seq=True,
        )
    else:
        raise ValueError(f"Unknown network type: {net_type}")

    return network, inputs, conditions, embedding_net
