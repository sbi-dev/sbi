# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from sbi.neural_nets import (
    FlowMatchingConfig,
    SubVPScoreConfig,
    VEScoreConfig,
    VPScoreConfig,
    VectorFieldConfigBase,
)
from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.utils.torchutils import process_device

CONFIGS = [FlowMatchingConfig, VPScoreConfig, SubVPScoreConfig, VEScoreConfig]


@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,), (3, 3)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("config_cls", CONFIGS)
def test_vector_field_estimator_loss_shapes(
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
    config_cls,
):
    """Test whether `loss` of vector field estimators follow the shape convention."""
    estimator, inputs, conditions = _build_estimator_and_tensors(
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        config_cls=config_cls,
    )

    losses = estimator.loss(inputs[0], condition=conditions)
    assert losses.shape == (batch_dim,)


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("config_cls", CONFIGS)
def test_vector_field_estimator_on_device(device, config_cls):
    """Test whether vector field estimators can be moved to the device."""
    device = process_device(device)
    estimator = config_cls().build(torch.randn(100, 1), torch.randn(100, 1))
    estimator.to(device)

    # Test forward
    inputs = torch.randn(100, 1, device=device)
    condition = torch.randn(100, 1, device=device)
    time = torch.randn(1, device=device)
    out = estimator(inputs, condition, time)

    assert out.device.type == device.split(":")[0], "Output device mismatch."

    # Test loss
    loss = estimator.loss(inputs, condition)
    assert loss.device.type == device.split(":")[0], "Loss device mismatch."


@pytest.mark.parametrize("input_sample_dim", (1, 2, 3))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,), (3, 3)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("config_cls", CONFIGS)
def test_vector_field_estimator_forward_shapes(
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
    config_cls,
):
    """Test whether `forward` of vector field estimators follow the shape convention."""
    estimator, inputs, conditions = _build_estimator_and_tensors(
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        config_cls=config_cls,
    )
    # Batched times
    times = torch.rand((batch_dim,))
    outputs = estimator(inputs[0], condition=conditions, time=times)
    assert outputs.shape == (batch_dim, *input_event_shape), "Output shape mismatch."

    # Single time
    time = torch.rand(())
    outputs = estimator(inputs[0], condition=conditions, time=time)
    assert outputs.shape == (batch_dim, *input_event_shape), "Output shape mismatch."


def _build_estimator_and_tensors(
    input_event_shape: Tuple[int],
    condition_event_shape: Tuple[int],
    batch_dim: int,
    input_sample_dim: int = 1,
    config_cls: type[VectorFieldConfigBase] = FlowMatchingConfig,
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

    estimator = config_cls(embedding_net=embedding_net).build(
        torch.randn_like(building_thetas), torch.randn_like(building_xs)
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


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_train_schedule(config_cls):
    """Test on shapes and bounds for train and solve schedules
    of vector field estimators (flow or score)
    """
    t_min = torch.tensor([0.0])
    t_max = torch.tensor([1.0])

    estimator = config_cls().build(torch.randn(100, 1), torch.randn(100, 1))

    if config_cls is not FlowMatchingConfig:
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
    estimator = VEScoreConfig(
        train_schedule=train_schedule,
        solve_schedule=solve_schedule,
    ).build(torch.randn(100, 1), torch.randn(100, 1))

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
    estimator = VEScoreConfig(
        train_schedule="lognormal",
        lognormal_mean=-3.0,  # Very low mean -> small sigmas
        lognormal_std=2.0,  # High variance -> some extreme samples
    ).build(torch.randn(100, 1), torch.randn(100, 1))

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


@pytest.mark.parametrize("num_dim", [1, 2])
def test_flow_matching_score_broadcasts_one_time_per_sample(num_dim):
    """One time per sample gives the same scores as one call per sample."""
    estimator = FlowMatchingConfig().build(
        torch.randn(100, num_dim), torch.randn(100, 3)
    )
    inputs, condition, times = torch.randn(4, num_dim), torch.randn(4, 3), torch.rand(4)

    scores = estimator.score(inputs, condition, times)

    expected = torch.cat([
        estimator.score(inputs[i : i + 1], condition[i : i + 1], times[i : i + 1])
        for i in range(4)
    ])
    assert scores.shape == (4, num_dim)
    torch.testing.assert_close(scores, expected)


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_forward_sde_matches_marginals(config_cls):
    """The forward SDE d theta = f theta dt + g dW reproduces the time marginals.

    With marginals N(m(t) theta_0, s(t)^2), this needs dm/dt = f m and
    d(s^2)/dt = 2 f s^2 + g^2. Checked with central differences in float64.
    """
    estimator = config_cls().build(torch.randn(100, 1), torch.randn(100, 1)).double()
    one = torch.ones(1, 1, dtype=torch.float64)
    h = 1e-5
    # Flow matching clamps time at effective_t_max: its SDE is frozen above.
    t_upper = getattr(estimator, "effective_t_max", estimator.t_max)
    times = torch.linspace(
        estimator.t_min + 0.01, t_upper - 0.01, 9, dtype=torch.float64
    )
    for t in times.reshape(-1, 1):
        m = estimator.mean_t_fn(t).reshape(-1)
        s = estimator.std_fn(t).reshape(-1)
        dm = (estimator.mean_t_fn(t + h) - estimator.mean_t_fn(t - h)).reshape(-1)
        ds2 = (estimator.std_fn(t + h) ** 2 - estimator.std_fn(t - h) ** 2).reshape(-1)
        f = estimator.drift_fn(one, t).reshape(-1)
        g = estimator.diffusion_fn(one, t).reshape(-1)
        torch.testing.assert_close(dm / (2 * h), f * m, rtol=1e-6, atol=0)
        torch.testing.assert_close(
            ds2 / (2 * h), 2 * f * s**2 + g**2, rtol=1e-6, atol=0
        )


def test_flow_matching_score_is_the_marginal_score():
    """With the exact velocity of a Gaussian, `score` is the exact marginal score."""
    dtype = torch.float64
    mean = torch.tensor([0.5, -1.0], dtype=dtype)
    cov = torch.tensor([[0.3, 0.1], [0.1, 0.2]], dtype=dtype)
    eye = torch.eye(2, dtype=dtype)
    estimator = FlowMatchingConfig().build(torch.randn(100, 2), torch.randn(100, 1))
    estimator = estimator.double()

    def marginal(time):
        # theta_t = a theta_0 + b eps with theta_0 ~ N(mean, cov), eps ~ N(0, I).
        a, b = 1 - time, time + estimator.noise_scale
        return a, b, a**2 * cov + b**2 * eye

    def exact_velocity(input, condition, time):
        # E[eps - theta_0 | theta_t], the target of the flow matching loss.
        a, b, marginal_cov = marginal(time)
        residual = torch.linalg.solve(marginal_cov, (input - a * mean).T).T
        return residual @ (b * eye - a * cov) - mean

    estimator.forward = exact_velocity
    inputs = torch.randn(5, 2, dtype=dtype)
    condition = torch.zeros(5, 1, dtype=dtype)
    # Above effective_t_max (0.99), all SDE functions use the marginal at 0.99; the
    # iid methods and guidance combine `score` with `mean_t_fn` and `std_fn`.
    for t in [0.05, 0.5, 0.95, 0.995, 1.0]:
        time = torch.tensor([t], dtype=dtype)
        m = estimator.mean_t_fn(time).reshape(())
        s = estimator.std_fn(time).reshape(())
        marginal_cov = m**2 * cov + s**2 * eye
        expected = -torch.linalg.solve(marginal_cov, (inputs - m * mean).T).T
        torch.testing.assert_close(estimator.score(inputs, condition, time), expected)
