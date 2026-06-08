# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Supporting types for the Builder pattern: configuration, data-derived state,
and z-score computation.

``ZScoreConfig`` captures the user's preprocessing choice (per-axis z-score mode).
``ZScoreStats`` holds the computed statistics. ``BuildContext`` bundles all
data-derived state needed by ``_EstimatorBuilderBase.build()``.

``compute_z_score_stats`` is a pure helper that bridges config + data → stats.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from torch import Tensor

from sbi.utils.sbiutils import z_standardization


@dataclass(frozen=True)
class ZScoreConfig:
    """User-facing choice for z-score preprocessing per axis.

    Attributes:
        theta: Z-score mode for parameters. One of ``"none"``,
            ``"independent"`` (per-dimension), ``"structured"`` (single
            mean/std across all dimensions).
        x: Z-score mode for simulation outputs, same options as ``theta``.
    """

    theta: Literal["none", "independent", "structured"] = "independent"
    x: Literal["none", "independent", "structured"] = "independent"

    def __post_init__(self):
        allowed = ("none", "independent", "structured")
        for name in ("theta", "x"):
            val = getattr(self, name)
            if val not in allowed:
                raise ValueError(
                    f"ZScoreConfig.{name} must be one of {allowed}, got '{val}'"
                )


@dataclass(frozen=True, eq=False)
class ZScoreStats:
    """Computed z-score statistics derived from training data.

    Fields are ``None`` when the corresponding axis was not z-scored.

    Note:
        Equality and hashing use object identity (``a == b`` iff ``a is b``).
        For *content* comparison of stats, compare each tensor field directly
        with ``torch.equal`` or use ``torch.testing.assert_close`` in tests.
        Custom value-equality may be added later if a concrete consumer
        (e.g., a build cache) requires it.
    """

    theta_mean: Optional[Tensor] = None
    theta_std: Optional[Tensor] = None
    x_mean: Optional[Tensor] = None
    x_std: Optional[Tensor] = None


@dataclass(frozen=True, eq=False)
class BuildContext:
    """All data-derived state needed to construct an estimator.

    Carries shape / device / dtype and computed z-score stats.  Does NOT carry
    trainer lifecycle state (round index, retrain flags, etc.) — that stays in
    the trainer.

    Attributes:
        theta_shape: Event shape of parameters (excluding batch dimension).
        x_shape: Event shape of simulation outputs (excluding batch dimension).
        z_score_stats: Pre-computed z-score statistics.
        device: Target device for the estimator.
        dtype: Target dtype for the estimator.
    """

    theta_shape: torch.Size
    x_shape: torch.Size
    z_score_stats: ZScoreStats = field(default_factory=ZScoreStats)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_data(
        cls,
        theta: Tensor,
        x: Tensor,
        z_score_stats: Optional[ZScoreStats] = None,
    ) -> "BuildContext":
        """Construct a ``BuildContext`` by inferring shapes from training data.

        Args:
            theta: Parameter batch of shape ``(batch_size, *event_shape)``.
            x: Simulation output batch of shape ``(batch_size, *event_shape)``.
            z_score_stats: Pre-computed z-score statistics, or ``None`` for
                defaults (no stats).

        Returns:
            A new ``BuildContext`` instance.

        Raises:
            ValueError: If ``theta`` and ``x`` are on different devices.
        """
        if theta.device != x.device:
            raise ValueError(
                f"theta and x must be on the same device, got "
                f"{theta.device} and {x.device}."
            )
        return cls(
            theta_shape=torch.Size(theta.shape[1:]),
            x_shape=torch.Size(x.shape[1:]),
            z_score_stats=z_score_stats or ZScoreStats(),
            device=theta.device,
            dtype=theta.dtype,
        )


def compute_z_score_stats(
    theta: Tensor,
    x: Tensor,
    config: ZScoreConfig,
) -> ZScoreStats:
    """Compute z-score statistics from training data.

    Delegates to the existing ``z_standardization`` function for the actual
    mean/std computation.

    Args:
        theta: Parameter batch of shape ``(batch_size, *event_shape)``.
        x: Simulation output batch of shape ``(batch_size, *event_shape)``.
        config: Per-axis z-score mode.

    Returns:
        A ``ZScoreStats`` instance with computed means and standard deviations
        for the requested axes.
    """
    theta_mean, theta_std = None, None
    x_mean, x_std = None, None

    if config.theta != "none":
        theta_mean, theta_std = z_standardization(
            theta, structured_dims=(config.theta == "structured")
        )

    if config.x != "none":
        x_mean, x_std = z_standardization(x, structured_dims=(config.x == "structured"))

    return ZScoreStats(
        theta_mean=theta_mean,
        theta_std=theta_std,
        x_mean=x_mean,
        x_std=x_std,
    )
