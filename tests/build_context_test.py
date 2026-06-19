# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Tests for the builder-pattern foundation types.

Covers ``ZScoreConfig``, ``ZScoreStats``, ``BuildContext``,
``compute_z_score_stats``, the ``_EstimatorBuilderBase`` base class, and the
``ConditionalEstimatorBuildFn`` protocol rename.
"""

import pytest
import torch

from sbi.neural_nets.build_context import (
    BuildContext,
    ZScoreConfig,
    ZScoreStats,
    compute_z_score_stats,
)
from sbi.neural_nets.estimators.base import ConditionalEstimatorBuildFn
from sbi.neural_nets.net_builders.estimator_configs import ConditionalFlowConfig


def test_zscore_config_defaults():
    cfg = ZScoreConfig()
    assert cfg.theta == "independent"
    assert cfg.x == "independent"


@pytest.mark.parametrize(
    "theta, x",
    [
        ("none", "none"),
        ("independent", "structured"),
        ("structured", "independent"),
    ],
)
def test_zscore_config_valid_combinations(theta, x):
    cfg = ZScoreConfig(theta=theta, x=x)
    assert cfg.theta == theta
    assert cfg.x == x


@pytest.mark.parametrize("field", ["theta", "x"])
def test_zscore_config_invalid_value_raises(field):
    with pytest.raises(ValueError, match="must be one of"):
        ZScoreConfig(**{field: "invalid"})


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "theta_shape, x_shape",
    [
        ((100, 5), (100, 3)),
        ((50, 4), (50, 8, 8)),
    ],
)
def test_build_context_from_data(device, theta_shape, x_shape):
    """BuildContext.from_data infers shapes, device, and dtype correctly."""
    theta = torch.randn(*theta_shape, device=device)
    x = torch.randn(*x_shape, device=device)
    ctx = BuildContext.from_data(theta, x)
    assert ctx.theta_shape == torch.Size(theta_shape[1:])
    assert ctx.x_shape == torch.Size(x_shape[1:])
    assert ctx.device.type == device
    assert ctx.dtype == torch.float32


def test_build_context_from_data_with_stats():
    theta = torch.randn(100, 5)
    x = torch.randn(100, 3)
    stats = ZScoreStats(theta_mean=torch.zeros(5), theta_std=torch.ones(5))
    ctx = BuildContext.from_data(theta, x, z_score_stats=stats)
    assert ctx.z_score_stats is stats


@pytest.mark.gpu
def test_build_context_device_mismatch_raises():
    """Mismatched cpu/cuda devices raise ValueError."""
    theta_cpu = torch.randn(10, 3, device="cpu")
    x_cuda = torch.randn(10, 3, device="cuda")
    with pytest.raises(ValueError, match="same device"):
        BuildContext.from_data(theta_cpu, x_cuda)

    with pytest.raises(ValueError, match="same device"):
        BuildContext.from_data(x_cuda, theta_cpu)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "theta_mode, x_mode",
    [
        ("independent", "independent"),
        ("structured", "structured"),
        ("independent", "none"),
        ("none", "none"),
    ],
)
def test_compute_z_score_stats(device, theta_mode, x_mode):
    """Z-score stats have correct shape, type, and device for all mode combos."""
    theta = torch.randn(200, 5, device=device)
    x = torch.randn(200, 3, device=device)
    stats = compute_z_score_stats(theta, x, ZScoreConfig(theta=theta_mode, x=x_mode))

    if theta_mode == "none":
        assert stats.theta_mean is None
    elif theta_mode == "independent":
        assert stats.theta_mean.shape == (5,)
        assert stats.theta_mean.device.type == device
    else:
        assert stats.theta_mean.dim() == 0
        assert stats.theta_mean.device.type == device

    if x_mode == "none":
        assert stats.x_mean is None
    elif x_mode == "independent":
        assert stats.x_mean.shape == (3,)
        assert stats.x_mean.device.type == device
    else:
        assert stats.x_mean.dim() == 0
        assert stats.x_mean.device.type == device


def test_compute_z_score_stats_matches_z_standardization():
    """Output must match the existing z_standardization function."""
    from sbi.utils.sbiutils import z_standardization

    torch.manual_seed(0)
    theta = torch.randn(200, 4)
    x = torch.randn(200, 6)

    stats = compute_z_score_stats(theta, x, ZScoreConfig())
    expected_t_mean, expected_t_std = z_standardization(theta, False)
    expected_x_mean, expected_x_std = z_standardization(x, False)

    torch.testing.assert_close(stats.theta_mean, expected_t_mean)
    torch.testing.assert_close(stats.theta_std, expected_t_std)
    torch.testing.assert_close(stats.x_mean, expected_x_mean)
    torch.testing.assert_close(stats.x_std, expected_x_std)


def test_estimator_builder_base_build_raises():
    """Subclasses that don't override build() raise NotImplementedError."""
    cfg = ConditionalFlowConfig(hidden_features=64)
    theta = torch.randn(10, 5)
    x = torch.randn(10, 3)
    with pytest.raises(NotImplementedError, match="does not implement build"):
        cfg.build(theta, x)


def test_estimator_builder_base_from_kwargs():
    """The existing from_kwargs() + to_dict() contract is preserved."""
    cfg = ConditionalFlowConfig(hidden_features=64)
    d = cfg.to_dict()
    assert d == {"hidden_features": 64}


def test_estimator_builder_base_extra_warns():
    with pytest.warns(UserWarning, match="Unknown kwargs"):
        cfg = ConditionalFlowConfig.from_kwargs(
            hidden_features=64, some_zuko_param=True
        )
    d = cfg.to_dict()
    assert d == {"hidden_features": 64, "some_zuko_param": True}


def test_deprecated_alias_works_with_future_warning():
    with pytest.warns(FutureWarning, match="renamed to.*ConditionalEstimatorBuildFn"):
        from sbi.neural_nets.estimators.base import ConditionalEstimatorBuilder

    assert ConditionalEstimatorBuilder is ConditionalEstimatorBuildFn
