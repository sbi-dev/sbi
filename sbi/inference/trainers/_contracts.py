# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar, Union

from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils.typechecks import (
    validate_bool,
    validate_float_range,
    validate_optional,
    validate_positive_float,
    validate_positive_int,
)


@dataclass(frozen=True)
class StartIndexContext:
    """Inputs for computing the start index of training.

    Consolidates parameters that previously varied across subclasses, enabling a
    single base signature: `_get_start_index(context: StartIndexContext) -> int`.

    Fields are optional where method families differ; subclasses read only what
    they need.
    """

    # Common across methods (e.g., NLE/NRE);
    discard_prior_samples: bool

    # SNPE-specific knobs
    force_first_round_loss: Optional[bool] = None

    # Generic training state:
    resume_training: Optional[bool] = None

    def __post_init__(self):
        validate_bool(self.discard_prior_samples, "discard_prior_samples")
        validate_optional(self.force_first_round_loss, "force_first_round_loss", bool)
        validate_optional(self.resume_training, "resume_training", bool)


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

    def __post_init__(self):
        validate_positive_int(self.training_batch_size, "training_batch_size")
        validate_positive_float(self.learning_rate, "learning_rate")
        validate_float_range(
            self.validation_fraction,
            "validation_fraction",
            min_val=0,
            max_val=1,
            range_inclusive=False,
        )
        validate_positive_int(self.stop_after_epochs, "stop_after_epochs")
        validate_positive_int(self.max_num_epochs, "max_num_epochs")
        validate_bool(self.resume_training, "resume_training")
        validate_bool(self.retrain_from_scratch, "retrain_from_scratch")
        validate_bool(self.show_train_summary, "show_train_summary")
        if self.clip_max_norm is not None:
            validate_positive_float(self.clip_max_norm, "clip_max_norm")


@dataclass(frozen=True)
class LossArgsNRE:
    """
    Typed args for ratio-estimation losses (NRE family).

    Fields:
        num_atoms: Number of atoms to use for classification.
    """

    num_atoms: int = 10

    def __post_init__(self):
        validate_positive_int(self.num_atoms, "num_atoms")


@dataclass(frozen=True)
class LossArgsNRE_A(LossArgsNRE):
    """
    Typed args for NRE_A.

    Fields:
        num_atoms: Number of atoms to use for classification,
            AALR is defined for `num_atoms=2`.
    """

    num_atoms: int = field(init=False, default=2)

    def __post_init__(self):
        if self.num_atoms != 2:
            raise ValueError("In AARL / NRE-A, num_atoms must always be 2")


@dataclass(frozen=True, kw_only=True)
class LossArgsBNRE(LossArgsNRE_A):
    r"""
    Typed args for balanced neural ratio estimation losses (BNRE).

    Fields:
        regularization_strength: The multiplicative coefficient applied to the
            balancing regularizer ($\lambda$).
    """

    regularization_strength: float

    def __post_init__(self):
        validate_positive_float(self.regularization_strength, "regularization_strength")


@dataclass(frozen=True, kw_only=True)
class LossArgsNRE_C(LossArgsNRE):
    r"""
    Typed args for NRE_C losses.

    Fields:
       gamma: Determines the relative weight of the sum of all $K$ dependently
            drawn classes against the marginally drawn one. Specifically,
            $p(y=k) :=p_K$, $p(y=0) := p_0$, $p_0 = 1 - K p_K$, and finally
            $\gamma := K p_K / p_0$.
    """

    gamma: float

    def __post_init__(self):
        validate_positive_float(self.gamma, "gamma")


@dataclass(frozen=True)
class LossArgsNPE:
    """
    Typed args for posterior-estimation losses (NPE family).

    Fields:
        proposal may be a torch.distributions.Distribution or a NeuralPosterior
        calibration_kernel: A function to calibrate the loss with respect
            to the simulations `x` (optional). See Lueckmann, Gonçalves et al.,
            NeurIPS 2017. If `None`, no calibration is used.
        force_first_round_loss: If `True`, train with maximum likelihood,
            i.e., potentially ignoring the correction for using a proposal
            distribution different from the prior.
    """

    proposal: Optional[Union["Distribution", "NeuralPosterior"]] = None
    calibration_kernel: Optional[Callable[..., "Tensor"]] = None
    force_first_round_loss: bool = False

    def __post_init__(self):
        validate_optional(self.proposal, "proposal", Distribution, NeuralPosterior)
        validate_optional(self.calibration_kernel, "calibration_kernel", Callable)
        validate_bool(self.force_first_round_loss, "force_first_round_loss")


@dataclass(frozen=True)
class LossArgsVF:
    """
    Typed args for vector-field estimation losses (VF family).

    Fields:
        proposal: a torch.distributions.Distribution or a NeuralPosterior.
        calibration_kernel: A function to calibrate the loss with respect
            to the simulations `x` (optional). See Lueckmann, Gonçalves et al.,
            NeurIPS 2017. If `None`, no calibration is used.
        times: Time steps to compute the loss at.
        force_first_round_loss: If `True`, train with maximum likelihood,
            i.e., potentially ignoring the correction for using a proposal
            distribution different from the prior.
    """

    proposal: Optional[Union["Distribution", "NeuralPosterior"]] = None
    calibration_kernel: Optional[Callable[..., "Tensor"]] = None
    times: Optional["Tensor"] = None
    force_first_round_loss: bool = False

    def __post_init__(self):
        validate_optional(self.proposal, "proposal", Distribution, NeuralPosterior)
        validate_optional(self.calibration_kernel, "calibration_kernel", Callable)
        validate_optional(self.times, "times", Tensor)
        validate_bool(self.force_first_round_loss, "force_first_round_loss")


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
