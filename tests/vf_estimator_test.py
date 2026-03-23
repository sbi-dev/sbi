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


@pytest.mark.parametrize(
    "estimator_type,sde_type",
    [
        ("score", "vp"),
        ("score", "subvp"),
        ("score", "ve"),
        ("flow", None),
    ],
)
def test_train_schedule(estimator_type, sde_type):
    """Test on shapes and bounds for train and solve schedules
    of vector field estimators (flow or score)
    """
    embedding_net = torch.nn.Identity()
    t_min = torch.tensor([0.0])
    t_max = torch.tensor([1.0])

    if estimator_type == "flow":
        estimator = build_flow_matching_estimator(
            torch.randn(100, 1),
            torch.randn(100, 1),
            embedding_net=embedding_net,
        )

    else:
        estimator = build_score_matching_estimator(
            torch.randn(100, 1),
            torch.randn(100, 1),
            embedding_net=embedding_net,
            sde_type=sde_type,
        )
        # Train schedule only defined for score estimators
        # Schedule with default bounds
        train_schedule_default = estimator.train_schedule(300)
        assert train_schedule_default.shape == torch.Size((300,))
        assert train_schedule_default.max() <= estimator.t_max
        assert train_schedule_default.min() >= estimator.t_min

        # Schedule with given bounds
        train_schedule = estimator.train_schedule(300, t_min, t_max)
        assert train_schedule.shape == torch.Size((300,))
        assert train_schedule.max() <= t_max.item()
        assert train_schedule.min() >= t_min.item()

    # Solve schedule with default bounds
    solve_schedule_default = estimator.solve_schedule(
        300, t_max=estimator.t_max, t_min=estimator.t_min
    )
    assert torch.allclose(solve_schedule_default[0], torch.tensor([estimator.t_max]))
    assert torch.allclose(solve_schedule_default[-1], torch.tensor([estimator.t_min]))
    assert solve_schedule_default.shape == torch.Size((300,))
    assert torch.all(solve_schedule_default[:-1] - solve_schedule_default[1:] >= 0)

    # Solve schedule with given bounds
    solve_schedule = estimator.solve_schedule(
        300, t_max=t_max.item(), t_min=t_min.item()
    )
    assert torch.allclose(solve_schedule[0], t_max)
    assert torch.allclose(solve_schedule[-1], t_min)
    assert solve_schedule_default.shape == torch.Size((300,))
    assert torch.all(solve_schedule[:-1] - solve_schedule[1:] >= 0)


@pytest.mark.parametrize(
    "train_schedule,solve_schedule",
    [
        ("uniform", "uniform"),
        ("lognormal", "uniform"),
        ("uniform", "power_law"),
        ("lognormal", "power_law"),
    ],
)
def test_ve_edm_schedules(train_schedule, solve_schedule):
    """Test EDM-style schedules for VE estimator (Karras et al. 2022)."""
    estimator = build_score_matching_estimator(
        torch.randn(100, 1),
        torch.randn(100, 1),
        sde_type="ve",
        train_schedule=train_schedule,
        solve_schedule=solve_schedule,
    )

    # Test train schedule returns valid times without NaN.
    times_train = estimator.train_schedule(500)
    assert times_train.shape == (500,)
    assert torch.all(times_train >= estimator.t_min), "Train times below t_min"
    assert torch.all(times_train <= estimator.t_max), "Train times above t_max"
    assert not torch.any(torch.isnan(times_train)), "NaN in train schedule"

    # Test solve schedule returns monotonically decreasing times without NaN.
    times_solve = estimator.solve_schedule(100)
    assert times_solve.shape == (100,)
    assert torch.allclose(times_solve[0], torch.tensor(estimator.t_max)), (
        "First solve time != t_max"
    )
    assert torch.allclose(times_solve[-1], torch.tensor(estimator.t_min)), (
        "Last solve time != t_min"
    )
    assert torch.all(times_solve[:-1] >= times_solve[1:]), (
        "Solve schedule not monotonically decreasing"
    )
    assert not torch.any(torch.isnan(times_solve)), "NaN in solve schedule"


def test_ve_lognormal_no_nan_with_extreme_params():
    """Test that lognormal schedule doesn't produce NaN even with extreme params."""
    # Use parameters that could cause extreme sigma values.
    estimator = build_score_matching_estimator(
        torch.randn(100, 1),
        torch.randn(100, 1),
        sde_type="ve",
        train_schedule="lognormal",
        lognormal_mean=-3.0,  # Very low mean -> small sigmas
        lognormal_std=2.0,  # High variance -> some extreme samples
    )

    # Generate many samples to test edge cases.
    times = estimator.train_schedule(10000)
    assert not torch.any(torch.isnan(times)), (
        "NaN produced with extreme lognormal params"
    )
    assert not torch.any(torch.isinf(times)), (
        "Inf produced with extreme lognormal params"
    )
    assert torch.all(times >= estimator.t_min), "Times below t_min"
    assert torch.all(times <= estimator.t_max), "Times above t_max"
