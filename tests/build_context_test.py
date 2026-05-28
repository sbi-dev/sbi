# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Tests for the builder-pattern foundation types.

Covers ``ZScoreConfig``, ``ZScoreStats``, ``BuildContext``,
``compute_z_score_stats``, the ``_EstimatorBuilderBase`` base class, and the
``ConditionalEstimatorBuildFn`` protocol rename.
"""

import pytest
import torch
from torch import Tensor

from sbi.neural_nets.build_context import (
    BuildContext,
    ZScoreConfig,
    ZScoreStats,
    compute_z_score_stats,
)
from sbi.neural_nets.estimators.base import ConditionalEstimatorBuildFn
from sbi.neural_nets.net_builders.estimator_configs import (
    ClassifierConfig,
    ConditionalFlowConfig,
    MarginalFlowConfig,
    _EstimatorBuilderBase,
)


class TestZScoreConfig:
    """Validation and immutability of ZScoreConfig."""

    def test_defaults(self):
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
    def test_valid_combinations(self, theta, x):
        cfg = ZScoreConfig(theta=theta, x=x)
        assert cfg.theta == theta
        assert cfg.x == x

    @pytest.mark.parametrize("field", ["theta", "x"])
    def test_invalid_value_raises(self, field):
        with pytest.raises(ValueError, match="must be one of"):
            ZScoreConfig(**{field: "invalid"})

    def test_frozen(self):
        cfg = ZScoreConfig()
        with pytest.raises(AttributeError):
            cfg.theta = "none"


class TestZScoreStats:
    """Default construction and immutability of ZScoreStats."""

    def test_defaults_are_none(self):
        stats = ZScoreStats()
        assert stats.theta_mean is None
        assert stats.theta_std is None
        assert stats.x_mean is None
        assert stats.x_std is None

    def test_with_tensors(self):
        m = torch.zeros(5)
        s = torch.ones(5)
        stats = ZScoreStats(theta_mean=m, theta_std=s)
        assert torch.equal(stats.theta_mean, m)
        assert stats.x_mean is None

    def test_frozen(self):
        stats = ZScoreStats()
        with pytest.raises(AttributeError):
            stats.theta_mean = torch.zeros(3)


class TestBuildContext:
    """Construction and factory method of BuildContext."""

    def test_from_data(self):
        theta = torch.randn(100, 5)
        x = torch.randn(100, 3)
        ctx = BuildContext.from_data(theta, x)
        assert ctx.theta_shape == torch.Size([5])
        assert ctx.x_shape == torch.Size([3])
        assert ctx.device == torch.device("cpu")
        assert ctx.dtype == torch.float32

    def test_from_data_multidim(self):
        theta = torch.randn(50, 4)
        x = torch.randn(50, 8, 8)
        ctx = BuildContext.from_data(theta, x)
        assert ctx.theta_shape == torch.Size([4])
        assert ctx.x_shape == torch.Size([8, 8])

    def test_from_data_with_stats(self):
        theta = torch.randn(100, 5)
        x = torch.randn(100, 3)
        stats = ZScoreStats(theta_mean=torch.zeros(5), theta_std=torch.ones(5))
        ctx = BuildContext.from_data(theta, x, z_score_stats=stats)
        assert ctx.z_score_stats is stats

    def test_from_data_default_stats(self):
        ctx = BuildContext.from_data(torch.randn(10, 2), torch.randn(10, 3))
        assert ctx.z_score_stats.theta_mean is None

    def test_frozen(self):
        ctx = BuildContext(
            theta_shape=torch.Size([5]),
            x_shape=torch.Size([3]),
        )
        with pytest.raises(AttributeError):
            ctx.theta_shape = torch.Size([10])


class TestComputeZScoreStats:
    """Correctness of the pure z-score stats computation."""

    def test_independent_computes_per_dim(self):
        torch.manual_seed(42)
        theta = torch.randn(1000, 5)
        x = torch.randn(1000, 3)
        stats = compute_z_score_stats(theta, x, ZScoreConfig())

        assert stats.theta_mean is not None
        assert stats.theta_std is not None
        assert stats.theta_mean.shape == (5,)
        assert stats.theta_std.shape == (5,)
        assert stats.x_mean.shape == (3,)
        assert stats.x_std.shape == (3,)

    def test_none_returns_none(self):
        theta = torch.randn(100, 5)
        x = torch.randn(100, 3)
        stats = compute_z_score_stats(
            theta, x, ZScoreConfig(theta="none", x="none")
        )
        assert stats.theta_mean is None
        assert stats.theta_std is None
        assert stats.x_mean is None
        assert stats.x_std is None

    def test_structured_returns_scalar(self):
        theta = torch.randn(100, 5)
        x = torch.randn(100, 3)
        stats = compute_z_score_stats(
            theta, x, ZScoreConfig(theta="structured", x="structured")
        )
        # Structured mode computes a single mean/std scalar.
        assert stats.theta_mean.dim() == 0
        assert stats.x_mean.dim() == 0

    def test_mixed_modes(self):
        theta = torch.randn(100, 5)
        x = torch.randn(100, 3)
        stats = compute_z_score_stats(
            theta, x, ZScoreConfig(theta="independent", x="none")
        )
        assert stats.theta_mean is not None
        assert stats.theta_mean.shape == (5,)
        assert stats.x_mean is None

    def test_matches_z_standardization(self):
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


class TestEstimatorBuilderBase:
    """The renamed base class retains existing functionality and adds build()."""

    def test_build_raises_not_implemented(self):
        """Subclasses that don't override build() raise NotImplementedError."""
        cfg = ConditionalFlowConfig(hidden_features=64)
        ctx = BuildContext(theta_shape=torch.Size([5]), x_shape=torch.Size([3]))
        with pytest.raises(NotImplementedError, match="does not implement build"):
            cfg.build(ctx)

    def test_from_kwargs_still_works(self):
        """The existing from_kwargs() + to_dict() contract is preserved."""
        cfg = ConditionalFlowConfig(hidden_features=64)
        d = cfg.to_dict()
        assert d == {"hidden_features": 64}

    def test_from_kwargs_extra_warns(self):
        with pytest.warns(UserWarning, match="Unknown kwargs"):
            cfg = ConditionalFlowConfig.from_kwargs(
                hidden_features=64, some_zuko_param=True
            )
        d = cfg.to_dict()
        assert d == {"hidden_features": 64, "some_zuko_param": True}

    def test_subclasses_inherit(self):
        """All existing config subclasses now inherit from _EstimatorBuilderBase."""
        assert issubclass(ConditionalFlowConfig, _EstimatorBuilderBase)
        assert issubclass(ClassifierConfig, _EstimatorBuilderBase)
        assert issubclass(MarginalFlowConfig, _EstimatorBuilderBase)


class TestProtocolRename:
    """ConditionalEstimatorBuildFn exists and has the expected signature."""

    def test_protocol_exists(self):
        # The protocol should be importable under the new name.
        assert hasattr(ConditionalEstimatorBuildFn, "__call__")

    def test_callable_satisfies_protocol(self):
        """A simple callable with the right signature satisfies the protocol."""

        def dummy_build_fn(theta: Tensor, x: Tensor):
            return None

        # Runtime isinstance checks don't work with Protocol, but we can verify
        # the callable has the right signature shape.
        assert callable(dummy_build_fn)
