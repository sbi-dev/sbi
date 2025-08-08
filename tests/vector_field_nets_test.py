# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import itertools as it

import pytest
import torch
import torch.nn as nn

from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets.net_builders.vector_field_nets import (
    build_adamlp_network,
    build_standard_mlp_network,
    build_transformer_network,
)


def _grid(**kw_lists):
    keys = list(kw_lists)
    for vals in it.product(*(kw_lists[k] for k in keys)):
        yield dict(zip(keys, vals, strict=False))


MLP_CASES = [
    pytest.param(
        build_standard_mlp_network,
        kw,
        id="mlp-" + "-".join(f"{k}={v}" for k, v in kw.items()),
    )
    for kw in _grid(
        hidden_features=[16, [16, 32]],
        num_layers=[1, 2],
        layer_norm=[True, False],
    )
]

ADAMLP_CASES = [
    pytest.param(
        build_adamlp_network,
        kw,
        id="adamlp-" + "-".join(f"{k}={v}" for k, v in kw.items()),
    )
    for kw in _grid(
        hidden_features=[8, 16],
        num_layers=[1, 2],
        mlp_ratio=[2, 4],
    )
]

TRANSFORMER_CASES = [
    pytest.param(
        build_transformer_network,
        kw,
        id="tr-" + "-".join(f"{k}={v}" for k, v in kw.items()),
    )
    for kw in _grid(
        hidden_features=[16, 32],
        num_layers=[1, 2],
        num_heads=[2, 4],
        mlp_ratio=[2, 4],
        is_x_emb_seq=[False],  # plain transformer
    )
]

TRANSFORMER_CROSS_CASES = [
    pytest.param(
        build_transformer_network,
        kw,
        id="trX-" + "-".join(f"{k}={v}" for k, v in kw.items()),
    )
    for kw in _grid(
        hidden_features=[16, 32],
        num_layers=[1, 2],
        num_heads=[2, 4],
        mlp_ratio=[2, 4],
        is_x_emb_seq=[True],  # cross-attention variant
    )
]

ALL_BUILDER_CASES = (
    MLP_CASES + ADAMLP_CASES + TRANSFORMER_CASES + TRANSFORMER_CROSS_CASES
)


@pytest.mark.parametrize("builder, builder_kwargs", ALL_BUILDER_CASES)
@pytest.mark.parametrize("time_emb_type", ["sinusoidal", "random_fourier"])
@pytest.mark.parametrize("time_embedding_dim", [8, 16])
@pytest.mark.parametrize("batch_dim", [1, 3])
@pytest.mark.parametrize("input_dim", [1, 2])
@pytest.mark.parametrize(
    "condition_event_shape",
    [(1,), (2,), (1, 1), (2, 2)],  # includes seq-like for cross attention
)
def test_vector_field_builders_shape_and_build(
    builder,
    builder_kwargs,
    time_emb_type,
    time_embedding_dim,
    batch_dim,
    input_dim,
    condition_event_shape,
):
    # Ensure cross-attention always gets sequence-like conditions.
    if builder_kwargs.get("is_x_emb_seq", False) and len(condition_event_shape) < 2:
        condition_event_shape += (1,)

    # Inputs / conditions
    x = torch.randn(10, input_dim)
    y = torch.randn(10, *condition_event_shape)
    embedding_net = (
        nn.Identity()
        if (
            builder_kwargs.get("is_x_emb_seq", False) or len(condition_event_shape) == 1
        )
        else CNNEmbedding(condition_event_shape, kernel_size=1)
    )

    net = builder(
        batch_x=x,
        batch_y=y,
        time_embedding_dim=time_embedding_dim,
        time_emb_type=time_emb_type,
        embedding_net=embedding_net,
        **builder_kwargs,
    )

    # Forward pass
    inputs = torch.randn(batch_dim, input_dim)
    conditions = torch.randn(batch_dim, *condition_event_shape)
    t = torch.rand(batch_dim)

    out = net(
        inputs,
        conditions if embedding_net is nn.Identity else embedding_net(conditions),
        t,
    )
    assert out.shape == inputs.shape
