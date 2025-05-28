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
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("net_type", ["mlp", "ada_mlp", "transformer"])
@pytest.mark.parametrize("time_emb_type", ["sinusoidal", "random_fourier"])
@pytest.mark.parametrize("activation", [nn.GELU, nn.ReLU, nn.SiLU])
def test_network_shapes(
    input_event_shape,
    condition_event_shape,
    batch_dim,
    net_type,
    time_emb_type,
    activation,
):
    """Test whether vector field networks produce correct output shapes."""
    network, inputs, conditions = _build_vector_field_components(
        net_type,
        input_event_shape,
        condition_event_shape,
        batch_dim,
        time_emb_type=time_emb_type,
        activation=activation,
    )

    # Create time parameter
    t = torch.rand((batch_dim,))

    # Test forward pass
    outputs = network(inputs, conditions, t)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"


@pytest.mark.parametrize("hidden_features", [50, 100, [50, 100]])
@pytest.mark.parametrize("num_layers", [2, 5])
@pytest.mark.parametrize("layer_norm", [True, False])
@pytest.mark.parametrize("skip_connections", [True, False])
@pytest.mark.parametrize("time_emb_type", ["sinusoidal", "random_fourier"])
def test_mlp_network_parameters(
    hidden_features, num_layers, layer_norm, skip_connections, time_emb_type
):
    """Test whether MLP vector field networks can be built with different parameters."""
    batch_x = torch.randn(100, 5)
    batch_y = torch.randn(100, 3)

    network = build_standard_mlp_network(
        batch_x=batch_x,
        batch_y=batch_y,
        hidden_features=hidden_features,
        num_layers=num_layers,
        time_embedding_dim=32,
        layer_norm=layer_norm,
        skip_connections=skip_connections,
        time_emb_type=time_emb_type,
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
@pytest.mark.parametrize("num_layers", [2, 4])
@pytest.mark.parametrize("mlp_ratio", [2, 4])
@pytest.mark.parametrize("num_intermediate_mlp_layers", [0, 2])
@pytest.mark.parametrize("time_emb_type", ["sinusoidal", "random_fourier"])
def test_adamlp_network_parameters(
    hidden_features, num_layers, mlp_ratio, num_intermediate_mlp_layers, time_emb_type
):
    """Test whether AdaMLP vector field networks can be built with different
    parameters."""
    batch_x = torch.randn(100, 5)
    batch_y = torch.randn(100, 3)

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
    inputs = torch.randn(10, 5)
    conditions = torch.randn(10, 3)
    times = torch.rand(10)

    outputs = network(inputs, conditions, times)
    assert outputs.shape == inputs.shape, "Output shape should match input shape"


@pytest.mark.parametrize("hidden_features", [64, 128])
@pytest.mark.parametrize("num_blocks", [2, 4])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("mlp_ratio", [2, 4, 8])
@pytest.mark.parametrize("time_emb_type", ["sinusoidal", "random_fourier"])
def test_transformer_network_parameters(
    hidden_features, num_blocks, num_heads, mlp_ratio, time_emb_type
):
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
        mlp_ratio=mlp_ratio,
        time_embedding_dim=32,
        time_emb_type=time_emb_type,
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
    time_emb_type: str = "sinusoidal",
    activation: nn.Module = nn.GELU,
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
            time_emb_type=time_emb_type,
            activation=activation,
        )
    elif net_type == "ada_mlp":
        network = build_adamlp_network(
            batch_x=batch_x,
            batch_y=batch_y,
            hidden_features=64,
            time_embedding_dim=32,
            embedding_net=embedding_net,
            time_emb_type=time_emb_type,
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
            time_emb_type=time_emb_type,
        )
    else:
        raise ValueError(f"Unknown network type: {net_type}")

    return network, inputs, conditions
