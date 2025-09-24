# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union

if TYPE_CHECKING:  # import-heavy deps only for type checkers
    from torch import Tensor
    from torch.distributions import Distribution

    from sbi.inference.posteriors.base_posterior import NeuralPosterior


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
    use_non_atomic_loss: Optional[bool] = None
    ran_final_round: Optional[bool] = None

    # Generic training state:
    resume_training: Optional[bool] = None


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


@dataclass(frozen=True)
class LossArgsNRE:
    """
    Typed args for ratio-estimation losses (NRE family).

    Fields:
        num_atoms: Number of atoms to use for classification.
    """

    num_atoms: int = 10


@dataclass(frozen=True, kw_only=True)
class LossArgsBNRE(LossArgsNRE):
    r"""
    Typed args for balanced neural ratio estimation losses (BNRE).

    Fields:
        regularization_strength: The multiplicative coefficient applied to the
            balancing regularizer ($\lambda$).
    """

    regularization_strength: float


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


@dataclass(frozen=True)
class LossArgsVF:
    """
    Typed args for vector-field estimation losses (VF family).

    Fields:
        proposal: a torch.distributions.Distribution or a NeuralPosterior.
        calibration_kernel: A function to calibrate the loss with respect
            to the simulations `x` (optional). See Lueckmann, Gonçalves et al.,
            NeurIPS 2017. If `None`, no calibration is used.
        times: Times :math:`t`.
        force_first_round_loss: If `True`, train with maximum likelihood,
            i.e., potentially ignoring the correction for using a proposal
            distribution different from the prior.
    """

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
