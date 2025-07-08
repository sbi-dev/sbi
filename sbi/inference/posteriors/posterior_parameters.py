from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Union

import torch

from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.sbi_types import PyroTransformedDistribution, TorchTransform


@dataclass(frozen=True)
class DirectPosteriorParameters:
    """
    Parameters for initializing DirectPosterior.
    """

    max_sampling_batch_size: int = 10_000
    x_shape: Optional[torch.Size] = None
    enable_transform: bool = True


@dataclass(frozen=True)
class ImportanceSamplingPosteriorParameters:
    """
    Parameters for initializing ImportanceSamplingPosterior.
    """

    theta_transform: Optional[TorchTransform] = None
    method: Literal["sir", "importance"] = "sir"
    oversampling_factor: int = 32
    max_sampling_batch_size: int = 10_000
    x_shape: Optional[torch.Size] = None


@dataclass(frozen=True)
class MCMCPosteriorParameters:
    """
    Parameters for initializing MCMCPosterior.
    """

    method: Literal[
        "slice_np",
        "slice_np_vectorized",
        "hmc_pyro",
        "nuts_pyro",
        "slice_pymc",
        "hmc_pymc",
        "nuts_pymc",
    ] = "slice_np_vectorized"
    thin: int = -1
    warmup_steps: int = 200
    num_chains: int = 20
    init_strategy: Literal["proposal", "sir", "resample"] = "resample"
    init_strategy_parameters: Optional[Dict[str, Any]] = None
    init_strategy_num_candidates: Optional[int] = None
    num_workers: int = 1
    mp_context: Literal["fork", "spawn"] = "spawn"
    x_shape: Optional[torch.Size] = None


@dataclass(frozen=True)
class RejectionPosteriorParameters:
    """
    Parameters for initializing RejectionPosterior.
    """

    theta_transform: Optional[TorchTransform] = None
    max_sampling_batch_size: int = 10_000
    num_samples_to_find_max: int = 10_000
    num_iter_to_find_max: int = 100
    m: float = 1.2
    x_shape: Optional[torch.Size] = None


@dataclass(frozen=True)
class VectorFieldPosteriorParameters:
    """
    Parameters for initializing VectorFieldPosterior.
    """

    max_sampling_batch_size: int = 10_000
    enable_transform: bool = True
    vector_field_estimator_potential_args: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VIPosteriorParameters:
    """
    Parameters for initializing VIPosterior.
    """

    q: Union[
        Literal["nsf", "scf", "maf", "mcf", "gaussian", "gaussian_diag"],
        PyroTransformedDistribution,
        "VIPosterior",
        Callable,
    ] = "maf"
    vi_method: Literal["rKL", "fKL", "IW", "alpha"] = "rKL"
    x_shape: Optional[torch.Size] = None
    parameters: Optional[Iterable] = None
    modules: Optional[Iterable] = None
