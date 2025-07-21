# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Optional,
    Union,
    cast,
    get_args,
    get_origin,
)

from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.sbi_types import PyroTransformedDistribution, TorchTransform
from sbi.utils.typechecks import (
    is_nonnegative_int,
    is_positive_float,
    is_positive_int,
)


@dataclass(frozen=True)
class PosteriorParameters(ABC):
    @abstractmethod
    def validate(self):
        """
        Method for subclasses to override and implement
        custom validation logic. Called at the end of __post_init__.
        """
        ...

    def with_param(self, **kwargs):
        """
        Create a new instance of the class with updated field values.

        Only allows updates to fields defined in the dataclass. Raises an error if
        any unknown or invalid field names are passed.

        Args:
            **kwargs: Field-value pairs to override in the new instance.

        Returns:
            A new instance of the same class with updated values.

        Raises:
            ValueError: If any of the provided keys are not valid dataclass fields.
        """

        valid_fields = set(self.__dataclass_fields__)
        for key in kwargs:
            if key not in valid_fields:
                raise ValueError(
                    f"Invalid parameter: '{key}' is not a valid field"
                    f" of {self.__class__.__name__}"
                )
        return replace(self, **kwargs)

    def __post_init__(self) -> None:
        """
        Performs runtime validation and type enforcement after dataclass initialization.

        - Enforces that fields annotated with `Literal[...]` contain valid values.
        - Attempts to cast fields annotated as primitive types (int, float, bool) to
          their expected types if not already correctly typed.
        - Calls the `validate()` method at the end for additional custom checks.

        Raises:
            ValueError: If any field fails its Literal constraint or cannot be cast to
                        the expected primitive type.
        """

        for field in self.__dataclass_fields__.values():
            field_name = field.name
            raw_value = getattr(self, field_name)
            annotation = field.type
            target_type = cast(type, annotation)

            # Check if the value is among the valid choices
            # defined by a Literal annotation
            if get_origin(annotation) is Literal:
                allowed = get_args(annotation)
                if raw_value not in allowed:
                    raise ValueError(
                        f"Field '{field_name}' must be one of {allowed},"
                        f" got {raw_value}"
                    )
            # Attempt to cast primitive type values to ensure type correctness
            elif target_type in (int, float, bool):
                try:
                    value = target_type(raw_value)
                except Exception as e:
                    raise ValueError(
                        f"Could not convert the value of the field {field} to the "
                        f"expected type {target_type}."
                    ) from e

                # Overwrite the original field value with the converted value
                object.__setattr__(self, field_name, value)

        # Run additional validations specified in subclasses
        self.validate()


@dataclass(frozen=True)
class DirectPosteriorParameters(PosteriorParameters):
    """
    Parameters for initializing DirectPosterior.

    Fields:
        max_sampling_batch_size: Batchsize of samples being drawn from
            the proposal at every iteration.
        enable_transform: Whether to transform parameters to unconstrained space
            during MAP optimization. When False, an identity transform will be
            returned for `theta_transform`.
    """

    max_sampling_batch_size: int = 10_000
    enable_transform: bool = True

    def validate(self):
        """Validate DirectPosteriorParameters fields."""

        if not is_positive_int(self.max_sampling_batch_size):
            raise ValueError("max_sampling_batch_size must be greater than 0.")


@dataclass(frozen=True)
class ImportanceSamplingPosteriorParameters(PosteriorParameters):
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
    """

    theta_transform: Optional[TorchTransform] = None
    method: Literal["sir", "importance"] = "sir"
    oversampling_factor: int = 32
    max_sampling_batch_size: int = 10_000

    def validate(self):
        """Validate ImportanceSamplingPosteriorParameters fields."""

        if not (
            self.theta_transform is None
            or isinstance(self.theta_transform, TorchTransform)
        ):
            raise TypeError(
                "theta_transform must be either None or of type TorchTransform"
            )
        if not is_positive_int(self.oversampling_factor):
            raise ValueError("oversampling_factor must be greater than 0.")
        if not is_positive_int(self.max_sampling_batch_size):
            raise ValueError("max_sampling_batch_size must be greater than 0.")


@dataclass(frozen=True)
class MCMCPosteriorParameters(PosteriorParameters):
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
        num_workers: number of cpu cores used to parallelize mcmc
        mp_context: Multiprocessing start method, either `"fork"` or `"spawn"`
            (default), used by Pyro and PyMC samplers. `"fork"` can be significantly
            faster than `"spawn"` but is only supported on POSIX-based systems
            (e.g. Linux and macOS, not Windows).
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
    num_workers: int = 1
    mp_context: Literal["fork", "spawn"] = "spawn"

    def validate(self):
        """Validate MCMCPosteriorParameters fields."""

        if not (
            self.init_strategy_parameters is None
            or isinstance(self.init_strategy_parameters, Dict)
        ):
            raise TypeError(
                "init_strategy_parameters must be either None or of type Dict"
            )
        if self.thin != -1 and not (1 <= self.thin <= 10):
            raise ValueError("thin must be a value between 10 to 1, or -1.")
        if not is_nonnegative_int(self.warmup_steps):
            raise ValueError("warmup_steps must be greater than or equal to 0.")
        if not is_positive_int(self.num_chains):
            raise ValueError("num_chains must be greater than 0.")
        if not is_positive_int(self.num_workers):
            raise ValueError("num_workers must be greater than 0.")


@dataclass(frozen=True)
class RejectionPosteriorParameters(PosteriorParameters):
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
    """

    theta_transform: Optional[TorchTransform] = None
    max_sampling_batch_size: int = 10_000
    num_samples_to_find_max: int = 10_000
    num_iter_to_find_max: int = 100
    m: float = 1.2

    def validate(self):
        """Validate RejectionPosteriorParameters fields."""

        if not (
            self.theta_transform is None
            or isinstance(self.theta_transform, TorchTransform)
        ):
            raise TypeError(
                "theta_transform must be either None or of type TorchTransform"
            )

        if not is_positive_int(self.max_sampling_batch_size):
            raise ValueError("max_sampling_batch_size must be greater than 0.")
        if not is_positive_int(self.num_samples_to_find_max):
            raise ValueError("num_samples_to_find_max must be greater than 0.")
        if not is_nonnegative_int(self.num_iter_to_find_max):
            raise ValueError("num_iter_to_find_max must be greater than or equal to 0.")
        if not is_positive_float(self.m):
            raise ValueError("m must be greater than 0.")


@dataclass(frozen=True)
class VectorFieldPosteriorParameters(PosteriorParameters):
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
    neural_ode_backend: Literal["zuko"] = "zuko"
    neural_ode_kwargs: Optional[Dict[str, Any]] = None

    def validate(self):
        """Validate VectorFieldPosteriorParameters fields."""

        if not (self.iid_params is None or isinstance(self.iid_params, Dict)):
            raise TypeError("iid_params must be either None or of type Dict")
        if not (
            self.neural_ode_kwargs is None or isinstance(self.neural_ode_kwargs, Dict)
        ):
            raise TypeError("neural_ode_kwargs must be either None or of type Dict")
        if not is_positive_int(self.max_sampling_batch_size):
            raise ValueError("max_sampling_batch_size must be greater than 0.")


@dataclass(frozen=True)
class VIPosteriorParameters(PosteriorParameters):
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
    parameters: Optional[Iterable] = None
    modules: Optional[Iterable] = None

    def validate(self):
        """Validate VIPosteriorParameters fields."""

        valid_q = {"nsf", "scf", "maf", "mcf", "gaussian", "gaussian_diag"}

        if isinstance(self.q, str) and self.q not in valid_q:
            raise ValueError(f"If `q` is a string, it must be one of {valid_q}")
        elif not isinstance(
            self.q, (PyroTransformedDistribution, VIPosterior, Callable, str)
        ):
            raise TypeError(
                "q must be either of typr PyroTransformedDistribution,"
                " VIPosterioror or Callable"
            )

        if self.parameters is not None and not isinstance(self.parameters, Iterable):
            raise TypeError("parameters must be iterable or None.")
        if self.modules is not None and not isinstance(self.modules, Iterable):
            raise TypeError("modules must be iterable or None.")
