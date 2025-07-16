from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Union

import torch

from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.sbi_types import PyroTransformedDistribution, TorchTransform


@dataclass(frozen=True)
class DirectPosteriorParameters:
    """
    Parameters for initializing DirectPosterior.

    Fields:
        max_sampling_batch_size: Batchsize of samples being drawn from
            the proposal at every iteration.
        x_shape: Deprecated, should not be passed.
        enable_transform: Whether to transform parameters to unconstrained space
            during MAP optimization. When False, an identity transform will be
            returned for `theta_transform`.
    """

    max_sampling_batch_size: int = 10_000
    x_shape: Optional[torch.Size] = None
    enable_transform: bool = True


@dataclass(frozen=True)
class ImportanceSamplingPosteriorParameters:
    """
    Parameters for initializing ImportanceSamplingPosterior.

    Fields:
        theta_transform: Transformation that is applied to parameters. Is not used
            during but only when calling `.map()`.
        method: Either of [`sir`|`importance`]. This sets the behavior of the
            `.sample()` method. With `sir`, approximate posterior samples are
            generated with sampling importance resampling (SIR). With
            `importance`, the `.sample()` method returns a tuple of samples and
            corresponding importance weights.
        oversampling_factor: Number of proposed samples from which only one is
            selected based on its importance weight.
        max_sampling_batch_size: The batch size of samples being drawn from the
            proposal at every iteration.
        x_shape: Deprecated, should not be passed.
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

    Fields:
        method: Method used for MCMC sampling, one of `slice_np`,
            `slice_np_vectorized`, `hmc_pyro`, `nuts_pyro`, `slice_pymc`,
            `hmc_pymc`, `nuts_pymc`. `slice_np` is a custom
            numpy implementation of slice sampling. `slice_np_vectorized` is
            identical to `slice_np`, but if `num_chains>1`, the chains are
            vectorized for `slice_np_vectorized` whereas they are run sequentially
            for `slice_np`. The samplers ending on `_pyro` are using Pyro, and
            likewise the samplers ending on `_pymc` are using PyMC.
        thin: The thinning factor for the chain, default 1 (no thinning).
        warmup_steps: The initial number of samples to discard.
        num_chains: The number of chains. Should generally be at most
            `num_workers - 1`.
        init_strategy: The initialisation strategy for chains; `proposal` will draw
            init locations from `proposal`, whereas `sir` will use Sequential-
            Importance-Resampling (SIR). SIR initially samples
            `init_strategy_num_candidates` from the `proposal`, evaluates all of
            them under the `potential_fn` and `proposal`, and then resamples the
            initial locations with weights proportional to `exp(potential_fn -
            proposal.log_prob`. `resample` is the same as `sir` but
            uses `exp(potential_fn)` as weights.
        init_strategy_parameters: Dictionary of keyword arguments passed to the
            init strategy, e.g., for `init_strategy=sir` this could be
            `num_candidate_samples`, i.e., the number of candidates to find init
            locations (internal default is `1000`), or `device`.
        init_strategy_num_candidates: Number of candidates to find init
             locations in `init_strategy=sir` (deprecated, use
             init_strategy_parameters instead).
        num_workers: number of cpu cores used to parallelize mcmc
        mp_context: Multiprocessing start method, either `"fork"` or `"spawn"`
            (default), used by Pyro and PyMC samplers. `"fork"` can be significantly
            faster than `"spawn"` but is only supported on POSIX-based systems
            (e.g. Linux and macOS, not Windows).
        x_shape: Deprecated, should not be passed.
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

    Fields:
        theta_transform: Transformation that is applied to parameters. Is not used
            during but only when calling `.map()`.
        max_sampling_batch_size: The batchsize of samples being drawn from
            the proposal at every iteration.
        num_samples_to_find_max: The number of samples that are used to find the
            maximum of the `potential_fn / proposal` ratio.
        num_iter_to_find_max: The number of gradient ascent iterations to find the
            maximum of the `potential_fn / proposal` ratio.
        m: Multiplier to the `potential_fn / proposal` ratio.
        x_shape: Deprecated, should not be passed.
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

    Fields:
        max_sampling_batch_size: Batchsize of samples being drawn from
            the proposal at every iteration.
        enable_transform: Whether to transform parameters to unconstrained space
            during MAP optimization. When False, an identity transform will be
            returned for `theta_transform`. True is not supported yet.
        iid_method: Which method to use for computing the score in the iid setting.
            We currently support "fnpe", "gauss", "auto_gauss", "jac_gauss".
        iid_params: Parameters for the iid method, for arguments see
            `IIDScoreFunction`.
        neural_ode_backend: The backend to use for the neural ODE. Currently,
            only "zuko" is supported.
        neural_ode_kwargs: Additional keyword arguments for the neural ODE.
    """

    max_sampling_batch_size: int = 10_000
    enable_transform: bool = True

    # fields passed from VectorfieldPosterior as keyword arguments
    # to VectorFieldBasedPotential __init__ method
    iid_method: Literal["fnpe", "gauss", "auto_gauss", "jac_gauss"] = "auto_gauss"
    iid_params: Optional[Dict[str, Any]] = None
    neural_ode_backend: str = "zuko"
    neural_ode_kwargs: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class VIPosteriorParameters:
    """
    Parameters for initializing VIPosterior.

    Fields:
        q: Variational distribution, either string, `TransformedDistribution`, or a
            `VIPosterior` object. This specifies a parametric class of distribution
            over which the best possible posterior approximation is searched. For
            string input, we currently support [nsf, scf, maf, mcf, gaussian,
            gaussian_diag]. You can also specify your own variational family by
            passing a pyro `TransformedDistribution`.
            Additionally, we allow a `Callable`, which allows you the pass a
            `builder` function, which if called returns a distribution. This may be
            useful for setting the hyperparameters e.g. `num_transfroms` within the
            `get_flow_builder` method specifying the number of transformations
            within a normalizing flow. If q is already a `VIPosterior`, then the
            arguments will be copied from it (relevant for multi-round training).
        vi_method: This specifies the variational methods which are used to fit q to
            the posterior. We currently support [rKL, fKL, IW, alpha]. Note that
            some of the divergences are `mode seeking` i.e. they underestimate
            variance and collapse on multimodal targets (`rKL`, `alpha` for alpha >
            1) and some are `mass covering` i.e. they overestimate variance but
            typically cover all modes (`fKL`, `IW`, `alpha` for alpha < 1).
        x_shape: Deprecated, should not be passed.
        parameters: List of parameters of the variational posterior. This is only
            required for user-defined q i.e. if q does not have a `parameters`
            attribute.
        modules: List of modules of the variational posterior. This is only
            required for user-defined q i.e. if q does not have a `modules`
            attribute.
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
