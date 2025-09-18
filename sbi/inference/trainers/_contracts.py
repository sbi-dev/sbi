# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""
Typed contracts shared by trainer implementations.

This module centralizes small, import-light dataclasses and type aliases that
express trainer-facing contracts. Keep runtime imports minimal to avoid cycles
and heavy dependencies; prefer forward-referenced annotations.

Notes
- Do not import torch/torch.distributions at runtime here.
- Only use typing/dataclasses; rely on from __future__ import annotations so
    forward references like "Tensor" stay as strings at runtime.
- Keep these structures stable and documented; they define cross-trainer
    expectations and enable LSP-friendly hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union

if TYPE_CHECKING:  # import-heavy deps only for type checkers
    from torch import Tensor
    from torch.distributions import Distribution

    from sbi.inference.posteriors.base_posterior import NeuralPosterior

# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StartIndexContext:
    """Inputs for computing the start index of training.

    Consolidates parameters that previously varied across subclasses, enabling a
    single base signature: `_get_start_index(ctx: StartIndexContext) -> int`.

    Fields are optional where method families differ; subclasses read only what
    they need.
    """

    # Common across methods (e.g., NLE/NRE);
    discard_prior_samples: bool

    # SNPE-specific knobs (optional, read if relevant):
    force_first_round_loss: Optional[bool] = None
    use_non_atomic_loss: Optional[bool] = None
    ran_final_round: Optional[bool] = None

    # Generic training state:
    resume_training: Optional[bool] = None


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Configuration for the core training path.

    This captures loop-level hyperparameters and toggles that are independent of
    any specific estimator family. Subclass `train(...kwargs)` wrappers translate
    user kwargs into this config and delegate to the base core.
    """

    # Data & optimization
    training_batch_size: int
    learning_rate: float

    # Loop controls
    validation_fraction: float
    stop_after_epochs: int
    max_num_epochs: int

    # Lifecycle
    resume_training: bool
    retrain_from_scratch: bool

    # UX
    show_train_summary: bool

    # Regularization / safety
    clip_max_norm: Optional[float] = None


# ---------------------------------------------------------------------------
# Typed loss arguments per estimator family
# ---------------------------------------------------------------------------

# To avoid runtime imports, we rely on forward references (strings) for types
# like "Tensor", "Distribution", and "NeuralPosterior".


@dataclass(frozen=True)
class LossArgsNRE:
    """Typed args for ratio-estimation losses (NRE family)."""

    num_atoms: int


@dataclass(frozen=True)
class LossArgsBNRE(LossArgsNRE):
    regularization_strength: float


@dataclass(frozen=True)
class LossArgsNRE_C(LossArgsNRE):
    gamma: float


@dataclass(frozen=True)
class LossArgsNPE:
    """Typed args for posterior-estimation losses (NPE family).

    proposal may be a torch.distributions.Distribution or a NeuralPosterior;
    calibration_kernel is callable and may return a Tensor or adjust sampling.
    """

    proposal: Optional[Union["Distribution", "NeuralPosterior"]] = None
    calibration_kernel: Optional[Callable[..., "Tensor"]] = None
    force_first_round_loss: bool = False


@dataclass(frozen=True)
class LossArgsVF:
    """Typed args for vector-field estimation losses (VF family)."""

    proposal: Optional[Union["Distribution", "NeuralPosterior"]] = None
    calibration_kernel: Optional[Callable[..., "Tensor"]] = None
    times: Optional["Tensor"] = None
    force_first_round_loss: bool = False


# Union/TypeVar helpers if generics are desired in core signatures
LossArgs = Union[LossArgsNRE, LossArgsNPE, LossArgsVF]
LossArgsT = TypeVar("LossArgsT", LossArgsNRE, LossArgsNPE, LossArgsVF)


__all__ = [
    "StartIndexContext",
    "TrainConfig",
    "LossArgsNRE",
    "LossArgsNPE",
    "LossArgsVF",
    "LossArgs",
    "LossArgsT",
]
