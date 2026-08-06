# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Typed dataclass configs for density estimator factory functions.

These configs replace the error-prone ``dict(zip(...), **kwargs)`` pattern in
``sbi.neural_nets.factory``.  Use the ``from_kwargs()`` classmethod to construct a
config from user-supplied ``**kwargs``.  Known field names are validated — typos and
unknown parameters trigger a warning — while still being forwarded to the
underlying builder so that library-specific kwargs (e.g. Zuko flow
parameters) pass through.

The ``to_dict()`` method returns only explicitly-set (non-``None``) fields,
preserving the original behaviour where only user-specified values are forwarded
and builder defaults are left intact.

Note: because ``None`` serves as the "unset" sentinel, callers cannot
intentionally forward ``None`` to override a non-None builder default (e.g.
``tails=None`` in ``build_maf_rqs``).  This is an acceptable trade-off for
preserving typed field annotations.
"""

import inspect
from dataclasses import MISSING, dataclass, field, fields
from functools import lru_cache
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Sequence,
    Union,
    get_args,
    get_origin,
)

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from sbi.neural_nets.estimators import ConditionalVectorFieldEstimator
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimator,
    UnconditionalDensityEstimator,
)
from sbi.neural_nets.estimators.mixed_density_estimator import MixedDensityEstimator
from sbi.neural_nets.ratio_estimators import RatioEstimator

_BUILD_KWARG_ALIASES: dict = {
    "z_score_input": "z_score_x",
    "z_score_condition": "z_score_y",
}
"""Maps user-facing field names to the legacy kwarg names expected by the
downstream ``build_*`` functions."""


def _literal_values(tp) -> frozenset:
    """Extract allowed values from a (possibly nested) ``Literal`` type.
    Unwraps ``Optional[Literal[...]]`` and ``Union[Literal[...], ...]``
    recursively.
    """
    if get_origin(tp) is Literal:
        return frozenset(get_args(tp))
    out: frozenset = frozenset()
    for a in get_args(tp):
        out = out | _literal_values(a)
    return out


@dataclass(frozen=True, eq=False, repr=False)
class _EstimatorBuilderBase:
    """Shared base providing ``from_kwargs()``, ``to_dict()``, and the abstract
    ``build()`` contract for all estimator builders."""

    # kw_only so this base field does not become the first positional
    # parameter of every subclass (it would capture, e.g., a model name).
    extra_kwargs: dict = field(default_factory=dict, kw_only=True)

    def __post_init__(self):
        for f in fields(self):
            allowed = _literal_values(f.type)
            val = getattr(self, f.name)
            if allowed and val is not None and val not in allowed:
                raise ValueError(
                    f"Invalid value {val!r} for `{f.name}`. "
                    f"Must be one of {sorted(map(str, allowed))} or None."
                )

    _DISCRIMINATORS: ClassVar[frozenset] = frozenset({
        "model",
        "continuous_model",
        "estimator_type",
    })

    def __repr__(self) -> str:
        cls = type(self)
        parts: list[str] = []
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name == "extra_kwargs":
                if val:
                    parts.append(f"extra_kwargs={val!r}")
                continue
            if f.name in self._DISCRIMINATORS:
                parts.append(f"{f.name}={val!r}")
            elif f.default is not MISSING and val is f.default:
                continue
            elif val is not None:
                parts.append(f"{f.name}={val!r}")
        return f"{cls.__name__}({', '.join(parts)})"

    def _build_kwargs(self) -> dict:
        """Non-None fields as builder kwargs, alias-translated, minus discriminators.

        Field names are translated via ``_BUILD_KWARG_ALIASES`` so that
        user-facing names (e.g. ``z_score_input``) map to the legacy
        kwarg names (``z_score_x``) expected by ``build_*`` functions.
        """
        d = {
            _BUILD_KWARG_ALIASES.get(f.name, f.name): getattr(self, f.name)
            for f in fields(self)
            if f.name not in self._DISCRIMINATORS
            and f.name != "extra_kwargs"
            and getattr(self, f.name) is not None
        }
        d.update(self.extra_kwargs)
        return d

    @staticmethod
    @lru_cache(maxsize=None)
    def _fn_param_names(build_fn) -> frozenset:
        """Return the set of explicit parameter names for *build_fn*,
        excluding ``batch_x``, ``batch_y`` and ``**kwargs``."""
        return frozenset(
            p.name
            for p in inspect.signature(build_fn).parameters.values()
            if p.kind is not inspect.Parameter.VAR_KEYWORD
            and p.name not in ("batch_x", "batch_y")
        )

    def _reject_inapplicable_fields(
        self,
        build_fn,
        *,
        discriminator: str,
        always_ok: frozenset = frozenset(),
    ) -> None:
        """Raise ``ValueError`` if any non-None field would be silently
        ignored by *build_fn*.

        Args:
            build_fn: The downstream builder function whose signature
                defines the set of accepted parameter names.
            discriminator: Name of the field that selects which build
                function is used (e.g. ``"model"``).
            always_ok: Additional field names that are always
                acceptable regardless of *build_fn*'s signature.
        """
        allowed = self._fn_param_names(build_fn) | always_ok
        skip = self._DISCRIMINATORS | {"extra_kwargs"}
        set_fields = {
            f.name
            for f in fields(self)
            if f.name not in skip and getattr(self, f.name) is not None
        }
        bad = {
            n
            for n in set_fields
            if n not in allowed and _BUILD_KWARG_ALIASES.get(n, n) not in allowed
        }
        if bad:
            raise ValueError(
                f"Field(s) {sorted(bad)} are not used by "
                f"{discriminator}={getattr(self, discriminator)!r} and would "
                "be silently ignored. To forward library-specific options, "
                "use `extra_kwargs`."
            )

    @classmethod
    def from_kwargs(cls, **kwargs) -> "_EstimatorBuilderBase":
        """Create a config, forwarding unknown kwargs into ``extra_kwargs``.

        Known fields are set directly on the dataclass; any remaining kwargs
        are stored in ``extra_kwargs`` and merged back by ``to_dict()``.
        A warning is emitted for each unknown kwarg so that typos are still
        surfaced, while legitimate library-specific parameters (e.g. Zuko
        flow kwargs) pass through.
        """
        import warnings

        known_fields = {f.name for f in fields(cls)} - {"extra_kwargs"}
        known = {}
        extra = {}
        for k, v in kwargs.items():
            if k in known_fields:
                known[k] = v
            else:
                extra[k] = v

        if extra:
            warnings.warn(
                f"Unknown kwargs passed to {cls.__name__}: {set(extra)}. "
                f"These will be forwarded to the underlying builder. "
                f"If this is unintentional, check for typos.",
                stacklevel=3,
            )

        return cls(**known, extra_kwargs=extra)

    def build(
        self, batch_input: Tensor, batch_condition: Tensor
    ) -> ConditionalEstimator:
        """Build an estimator from training batches.

        Subclasses must override this method to construct the appropriate
        estimator.  Shape inference and z-scoring are derived from the
        supplied batches by the downstream ``build_*`` functions.

        Args:
            batch_input: Batch of the modeled variable (input to the density estimator)
            batch_condition: Batch of the conditioning variable

        Returns:
            A ``ConditionalEstimator`` subclass instance.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement build().")

    def to_dict(self) -> dict:
        """Return only explicitly-set (non-``None``) fields as a dict.

        Uses shallow field access (not ``dataclasses.asdict``) to avoid
        deep-copying ``nn.Module`` objects stored in fields like
        ``embedding_net``.  Extra (unknown) kwargs are merged in.
        """
        d = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "extra_kwargs" and getattr(self, f.name) is not None
        }
        d.update(self.extra_kwargs)
        return d


@dataclass(frozen=True, eq=False, repr=False)
class ConditionalFlowConfig(_EstimatorBuilderBase):
    """Configuration for conditional normalizing-flow density estimator builders.

    Used by ``posterior_nn`` and ``likelihood_nn``.  Fields cover all parameters
    accepted by any downstream builder (NFlows, Zuko, MDN, MADE, MNLE, MNPE).
    """

    # --- Shared across most builders ---
    z_score_x: Optional[Any] = None
    z_score_y: Optional[Any] = None
    hidden_features: Optional[Any] = None
    num_transforms: Optional[int] = None
    # num_bins: used by NFlows builders directly; Zuko wrappers (build_zuko_nsf,
    # build_zuko_ncsf) translate it to 'bins' before calling build_zuko_flow.
    num_bins: Optional[int] = None
    embedding_net: Optional[Any] = None
    num_components: Optional[int] = None

    # --- NFlows-specific (MAF, NSF, MAF-RQS) ---
    num_blocks: Optional[int] = None
    dropout_probability: Optional[float] = None
    use_batch_norm: Optional[bool] = None
    tail_bound: Optional[float] = None
    hidden_layers_spline_context: Optional[int] = None
    tails: Optional[str] = None
    min_bin_width: Optional[float] = None
    min_bin_height: Optional[float] = None
    min_derivative: Optional[float] = None

    # --- MADE-specific ---
    num_mixture_components: Optional[int] = None

    # --- Zuko shared ---
    x_dist: Optional[Any] = None

    # --- Zuko per-model kwargs (model-specific; ignored by models that don't use them)
    randperm: Optional[bool] = None  # zuko_maf, zuko_naf, zuko_unaf
    randmask: Optional[bool] = None  # zuko_nice
    signal: Optional[int] = None  # zuko_naf, zuko_unaf
    degree: Optional[int] = None  # zuko_sospf, zuko_bpf
    polynomials: Optional[int] = None  # zuko_sospf
    components: Optional[int] = None  # zuko_gf

    # --- Mixed-net specific (MNLE / MNPE) ---
    flow_model: Optional[str] = None
    log_transform_x: Optional[bool] = None
    num_categories_per_variable: Optional[Any] = None
    combined_embedding_net: Optional[Any] = None
    discrete_hidden_features: Optional[int] = None
    discrete_hidden_layers: Optional[int] = None
    continuous_hidden_features: Optional[int] = None

    # --- TabPFN-specific ---
    regressor_init_kwargs: Optional[dict] = None

    # --- Base distribution ---
    dtype: Optional[Any] = None


@dataclass(frozen=True, eq=False, repr=False)
class ClassifierConfig(_EstimatorBuilderBase):
    """Configuration for classifier builders (NRE).

    Covers parameters accepted by ``build_linear_classifier``,
    ``build_mlp_classifier``, and ``build_resnet_classifier``.
    """

    z_score_x: Optional[Any] = None
    z_score_y: Optional[Any] = None
    hidden_features: Optional[int] = None
    embedding_net_x: Optional[Any] = None
    embedding_net_y: Optional[Any] = None

    # --- ResNet-specific ---
    num_blocks: Optional[int] = None
    dropout_probability: Optional[float] = None
    use_batch_norm: Optional[bool] = None


@dataclass(frozen=True, eq=False, repr=False)
class MarginalFlowConfig(_EstimatorBuilderBase):
    """Configuration for marginal density estimator builders.

    Used by ``marginal_nn``.  Covers parameters accepted by
    ``build_zuko_unconditional_flow`` and the underlying Zuko constructors.
    """

    z_score_x: Optional[Any] = None
    hidden_features: Optional[Any] = None
    num_transforms: Optional[int] = None
    num_bins: Optional[int] = None
    num_components: Optional[int] = None

    # --- Zuko per-model kwargs ---
    randperm: Optional[bool] = None  # zuko_maf, zuko_naf, zuko_unaf
    randmask: Optional[bool] = None  # zuko_nice
    signal: Optional[int] = None  # zuko_naf, zuko_unaf
    degree: Optional[int] = None  # zuko_sospf, zuko_bpf
    polynomials: Optional[int] = None  # zuko_sospf
    components: Optional[int] = None  # zuko_gf


def _density_build_fns() -> dict:
    """Build-function mapping for density estimators."""
    from sbi.neural_nets.net_builders.flow import (
        build_made,
        build_maf,
        build_maf_rqs,
        build_nsf,
        build_zuko_bpf,
        build_zuko_gf,
        build_zuko_maf,
        build_zuko_naf,
        build_zuko_ncsf,
        build_zuko_nice,
        build_zuko_nsf,
        build_zuko_sospf,
        build_zuko_unaf,
    )
    from sbi.neural_nets.net_builders.mdn import build_mdn

    return {
        "mdn": build_mdn,
        "made": build_made,
        "maf": build_maf,
        "maf_rqs": build_maf_rqs,
        "nsf": build_nsf,
        "zuko_nice": build_zuko_nice,
        "zuko_maf": build_zuko_maf,
        "zuko_nsf": build_zuko_nsf,
        "zuko_ncsf": build_zuko_ncsf,
        "zuko_sospf": build_zuko_sospf,
        "zuko_naf": build_zuko_naf,
        "zuko_unaf": build_zuko_unaf,
        "zuko_gf": build_zuko_gf,
        "zuko_bpf": build_zuko_bpf,
    }


def _classifier_build_fns() -> dict:
    """Build-function mapping for ratio estimators."""
    from sbi.neural_nets.net_builders.classifier import (
        build_linear_classifier,
        build_mlp_classifier,
        build_resnet_classifier,
    )

    return {
        "linear": build_linear_classifier,
        "mlp": build_mlp_classifier,
        "resnet": build_resnet_classifier,
    }


DENSITY_MODELS = Literal[
    "mdn",
    "made",
    "maf",
    "maf_rqs",
    "nsf",
    "zuko_nice",
    "zuko_maf",
    "zuko_nsf",
    "zuko_ncsf",
    "zuko_sospf",
    "zuko_naf",
    "zuko_unaf",
    "zuko_gf",
    "zuko_bpf",
]

_VALID_DENSITY_MODELS = frozenset(get_args(DENSITY_MODELS))


@dataclass(frozen=True, eq=False, repr=False)
class DensityEstimatorBuilder(_EstimatorBuilderBase):
    """Builder for continuous density estimators (NPE / NLE).

    Covers NFlows (MAF, NSF, MAF-RQS, MADE), all Zuko flow variants, and MDN.
    Mixed density estimators (MNLE / MNPE) are handled by a separate builder.
    Fields mirror the parameters of the underlying ``build_*`` functions;
    see ``ConditionalFlowConfig`` for the full set.
    """

    model: DENSITY_MODELS = "maf"  # type: ignore[valid-type]

    # --- Shared across most builders ---
    z_score_input: Optional[
        Literal["none", "independent", "structured", "transform_to_unconstrained"]
    ] = None
    # The condition side never applies the unconstrained transform (the
    # build functions validate it for the input side only), so it is not
    # offered here.
    z_score_condition: Optional[Literal["none", "independent", "structured"]] = None
    hidden_features: Optional[Union[int, Sequence[int]]] = None
    num_transforms: Optional[int] = None
    num_bins: Optional[int] = None
    embedding_net: Optional[nn.Module] = None
    num_components: Optional[int] = None
    # Distribution over the modeled variable; required by (and only used
    # with) z_score_input="transform_to_unconstrained" on the zuko models
    # to derive the support bounds.
    x_dist: Optional[Distribution] = None

    # --- NFlows-specific (MAF, NSF, MAF-RQS) ---
    num_blocks: Optional[int] = None
    dropout_probability: Optional[float] = None
    use_batch_norm: Optional[bool] = None
    tail_bound: Optional[float] = None
    hidden_layers_spline_context: Optional[int] = None
    tails: Optional[str] = None
    min_bin_width: Optional[float] = None
    min_bin_height: Optional[float] = None
    min_derivative: Optional[float] = None

    # --- MADE-specific ---
    num_mixture_components: Optional[int] = None

    # --- Zuko per-model kwargs (model-specific; ignored by models that don't use them)
    randperm: Optional[bool] = None  # zuko_maf, zuko_naf, zuko_unaf
    randmask: Optional[bool] = None  # zuko_nice
    signal: Optional[int] = None  # zuko_naf, zuko_unaf
    degree: Optional[int] = None  # zuko_sospf, zuko_bpf
    polynomials: Optional[int] = None  # zuko_sospf
    components: Optional[int] = None  # zuko_gf

    def __post_init__(self):
        if self.model not in _VALID_DENSITY_MODELS:
            raise ValueError(
                f"Unknown model {self.model!r}. "
                f"Must be one of {sorted(_VALID_DENSITY_MODELS)}."
            )
        super().__post_init__()
        # x_dist is consumed only on the input side of the flow builders;
        # the condition side never applies the unconstrained transform.
        if self.x_dist is not None and self.z_score_input != (
            "transform_to_unconstrained"
        ):
            raise ValueError(
                "`x_dist` is only used with z_score_input='transform_to_unconstrained'."
            )
        self._reject_inapplicable_fields(
            _density_build_fns()[self.model],
            discriminator="model",
            # x_dist is consumed by the shared zuko flow constructor, which
            # the per-model build functions forward to via **kwargs.
            always_ok=frozenset({"x_dist"}),
        )

    def build(
        self, batch_input: Tensor, batch_condition: Tensor
    ) -> ConditionalDensityEstimator:
        """Build the density estimator by dispatching to the appropriate
        ``build_*`` function.

        Args:
            batch_input: Batch of the modeled variable used for
                shape inference and z-scoring.
            batch_condition: Batch of the conditioning variable
                used for shape inference and z-scoring.

        Returns:
            A ``ConditionalDensityEstimator`` (e.g., ``NFlowsFlow``,
            ``ZukoFlow``, or MDN).
        """
        build_fn = _density_build_fns()[self.model]
        kwargs = self._build_kwargs()
        return build_fn(batch_x=batch_input, batch_y=batch_condition, **kwargs)


_MIXED_ALWAYS_OK: frozenset = frozenset({
    "num_categories_per_variable",
    "embedding_net",
    "combined_embedding_net",
    "log_transform_x",
    "hidden_features",
    "discrete_hidden_features",
    "discrete_hidden_layers",
    "continuous_hidden_features",
    "dropout_probability",
    "z_score_input",
    "z_score_condition",
    "x_dist",
})


@dataclass(frozen=True, eq=False, repr=False)
class MixedDensityEstimatorBuilder(_EstimatorBuilderBase):
    """Builder for mixed (continuous + discrete) density estimators (MNLE / MNPE).

    Sibling of ``DensityEstimatorBuilder`` with a field set tailored to
    mixed-type data.  The continuous-component model is selected via
    ``continuous_model`` (density-estimator model for the continuous
    variables); there is no ``model`` field.
    """

    continuous_model: DENSITY_MODELS = "nsf"  # type: ignore[valid-type]

    # --- Mixed-specific ---
    num_categories_per_variable: Optional[Tensor] = None
    embedding_net: Optional[nn.Module] = None
    combined_embedding_net: Optional[nn.Module] = None
    log_transform_x: bool = False

    # --- Shared sizing ---
    hidden_features: Optional[int] = None
    discrete_hidden_features: Optional[int] = None
    discrete_hidden_layers: Optional[int] = None
    continuous_hidden_features: Optional[int] = None

    # --- Flow-specific (forwarded to the continuous sub-net) ---
    num_transforms: Optional[int] = None
    num_components: Optional[int] = None
    num_bins: Optional[int] = None
    tail_bound: Optional[float] = None
    dropout_probability: Optional[float] = None

    # --- Z-scoring ---
    # "transform_to_unconstrained" is supported on the input side when the
    # continuous model is a zuko flow (non-zuko continuous models raise
    # transitively) and requires `x_dist`. The condition side of the mixed
    # estimator never applies it, so it is not offered there.
    z_score_input: Optional[
        Literal["none", "independent", "structured", "transform_to_unconstrained"]
    ] = None
    z_score_condition: Optional[Literal["none", "independent", "structured"]] = None
    x_dist: Optional[Distribution] = None

    def __post_init__(self):
        if self.continuous_model not in _VALID_DENSITY_MODELS:
            raise ValueError(
                f"Unknown continuous_model {self.continuous_model!r}. "
                f"Must be one of {sorted(_VALID_DENSITY_MODELS)}."
            )
        super().__post_init__()
        # x_dist is consumed only on the input side of the continuous flow;
        # the condition side never applies the unconstrained transform.
        if self.x_dist is not None and self.z_score_input != (
            "transform_to_unconstrained"
        ):
            raise ValueError(
                "`x_dist` is only used with z_score_input='transform_to_unconstrained'."
            )
        from sbi.neural_nets.net_builders.mixed_nets import model_builders

        self._reject_inapplicable_fields(
            model_builders[self.continuous_model],
            discriminator="continuous_model",
            always_ok=_MIXED_ALWAYS_OK,
        )

    def build(
        self, batch_input: Tensor, batch_condition: Tensor
    ) -> MixedDensityEstimator:
        """Build the mixed density estimator by dispatching to
        ``_build_mixed_density_estimator``.

        Args:
            batch_input: Batch of the modeled variable used for
                shape inference and z-scoring.
            batch_condition: Batch of the conditioning variable
                used for shape inference and z-scoring.

        Returns:
            A ``MixedDensityEstimator``.
        """
        from sbi.neural_nets.net_builders.mixed_nets import (
            _build_mixed_density_estimator,
        )

        kwargs = self._build_kwargs()
        return _build_mixed_density_estimator(
            batch_x=batch_input,
            batch_y=batch_condition,
            flow_model=self.continuous_model,
            **kwargs,
        )


CLASSIFIER_MODELS = Literal["linear", "mlp", "resnet"]

_VALID_CLASSIFIER_MODELS = frozenset(get_args(CLASSIFIER_MODELS))


@dataclass(frozen=True, eq=False, repr=False)
class RatioEstimatorBuilder(_EstimatorBuilderBase):
    """Builder for ratio estimators / classifiers (NRE).

    Covers linear, MLP, and ResNet classifiers used by ``NRE_A``, ``NRE_B``,
    ``NRE_C``, and ``BNRE``.  Fields mirror the parameters of the underlying
    ``build_*_classifier`` functions.
    """

    model: CLASSIFIER_MODELS = "resnet"  # type: ignore[valid-type]

    # --- Shared across classifiers ---
    # For NRE, the builder's "input" is theta and the "condition" is x:
    # z_score_input z-scores the parameters, z_score_condition the data.
    z_score_input: Optional[Literal["none", "independent", "structured"]] = None
    z_score_condition: Optional[Literal["none", "independent", "structured"]] = None
    hidden_features: Optional[int] = None
    # User-facing semantics follow `classifier_nn`: embedding_net_theta embeds
    # the parameters, embedding_net_x embeds the data. (The underlying
    # build_*_classifier functions use positional x/y naming where their
    # `embedding_net_x` applies to theta — translated in `_build_kwargs`.)
    embedding_net_theta: Optional[nn.Module] = None
    embedding_net_x: Optional[nn.Module] = None

    # --- ResNet-specific ---
    num_blocks: Optional[int] = None
    dropout_probability: Optional[float] = None
    use_batch_norm: Optional[bool] = None

    # --- MLP-specific ---
    norm_layer: Optional[Callable[[int], nn.Module]] = None

    def __post_init__(self):
        if self.model not in _VALID_CLASSIFIER_MODELS:
            raise ValueError(
                f"Unknown model {self.model!r}. "
                f"Must be one of {sorted(_VALID_CLASSIFIER_MODELS)}."
            )
        super().__post_init__()
        self._reject_inapplicable_fields(
            _classifier_build_fns()[self.model],
            discriminator="model",
            # Translated to the build functions' positional x/y names in
            # `_build_kwargs`; applicable to all classifier models.
            always_ok=frozenset({"embedding_net_theta", "embedding_net_x"}),
        )

    def _build_kwargs(self) -> dict:
        """Translate user-facing embedding names to the build functions'
        positional naming: theta -> embedding_net_x, x -> embedding_net_y."""
        kwargs = super()._build_kwargs()
        if "embedding_net_x" in kwargs:
            kwargs["embedding_net_y"] = kwargs.pop("embedding_net_x")
        if "embedding_net_theta" in kwargs:
            kwargs["embedding_net_x"] = kwargs.pop("embedding_net_theta")
        return kwargs

    def build(self, batch_input: Tensor, batch_condition: Tensor) -> RatioEstimator:
        """Build the classifier by dispatching to the appropriate
        ``build_*_classifier`` function.

        Args:
            batch_input: Batch of the modeled variable used for
                shape inference and z-scoring.
            batch_condition: Batch of the conditioning variable
                used for shape inference and z-scoring.

        Returns:
            A ``RatioEstimator``.
        """
        build_fn = _classifier_build_fns()[self.model]
        kwargs = self._build_kwargs()
        return build_fn(batch_x=batch_input, batch_y=batch_condition, **kwargs)


VF_MODELS = Literal["mlp", "ada_mlp", "transformer", "transformer_cross_attn"]

_VALID_VF_MODELS = frozenset(get_args(VF_MODELS))


def _vector_field_build_fns() -> dict:
    """Build-function mapping for vector field estimators (per architecture)."""
    from sbi.neural_nets.net_builders.vector_field_nets import (
        build_adamlp_network,
        build_standard_mlp_network,
        build_transformer_network,
    )

    return {
        "mlp": build_standard_mlp_network,
        "ada_mlp": build_adamlp_network,
        "transformer": build_transformer_network,
        "transformer_cross_attn": build_transformer_network,
    }


_SCORE_ONLY_FIELDS: frozenset = frozenset({
    "sde_type",
    "train_schedule",
    "solve_schedule",
    "sigma_min",
    "sigma_max",
    "lognormal_mean",
    "lognormal_std",
    "power_law_exponent",
    "beta_min",
    "beta_max",
})

_FLOW_ONLY_FIELDS: frozenset = frozenset({"gaussian_baseline"})


@dataclass(frozen=True, eq=False, repr=False)
class VectorFieldEstimatorBuilder(_EstimatorBuilderBase):
    """Builder for vector-field estimators (FMPE / NPSE).

    Covers MLP, AdaMLP, Transformer, and cross-attention Transformer
    architectures used by ``FMPE`` (flow matching) and ``NPSE`` (score
    matching).  ``build()`` forwards to ``build_vector_field_estimator``,
    which constructs a ``FlowMatchingEstimator`` for ``"flow"`` and a
    ``ConditionalScoreEstimator`` subclass for ``"score"``.
    """

    _DISCRIMINATORS: ClassVar[frozenset] = frozenset({
        "model",
        "estimator_type",
    })

    model: VF_MODELS = "mlp"  # type: ignore[valid-type]
    estimator_type: Optional[Literal["flow", "score"]] = None
    sde_type: Optional[Literal["vp", "ve", "subvp"]] = None

    # --- Shared fields ---
    z_score_input: Optional[Literal["none", "independent", "structured"]] = None
    z_score_condition: Optional[Literal["none", "independent", "structured"]] = None
    hidden_features: Optional[Union[Sequence[int], int]] = None
    num_layers: Optional[int] = None
    time_embedding_dim: Optional[int] = None
    embedding_net: Optional[nn.Module] = None

    # --- Transformer-specific ---
    num_heads: Optional[int] = None
    mlp_ratio: Optional[int] = None

    # --- Flow-matching-specific ---
    gaussian_baseline: Optional[bool] = None

    # --- Shared estimator option ---
    compose_standardization: Optional[bool] = None

    # --- Score-matching-specific (VE schedule) ---
    train_schedule: Optional[Literal["uniform", "lognormal"]] = None
    solve_schedule: Optional[Literal["uniform", "power_law"]] = None
    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None
    lognormal_mean: Optional[float] = None
    lognormal_std: Optional[float] = None
    power_law_exponent: Optional[float] = None

    # --- Score-matching-specific (VP / SubVP) ---
    beta_min: Optional[float] = None
    beta_max: Optional[float] = None

    def __post_init__(self):
        if self.model not in _VALID_VF_MODELS:
            raise ValueError(
                f"Unknown model {self.model!r}. "
                f"Must be one of {sorted(_VALID_VF_MODELS)}."
            )
        super().__post_init__()

        # Reject fields that don't apply to the chosen estimator_type.
        # Skip this guard when estimator_type is None (resolved by the trainer).
        if self.estimator_type is not None:
            blocked = (
                _SCORE_ONLY_FIELDS
                if self.estimator_type == "flow"
                else _FLOW_ONLY_FIELDS
            )
            bad = {
                f.name
                for f in fields(self)
                if f.name in blocked and getattr(self, f.name) is not None
            }
            if bad:
                raise ValueError(
                    f"Field(s) {sorted(bad)} do not apply to "
                    f"estimator_type={self.estimator_type!r}. Remove them or "
                    f"change estimator_type."
                )

        # Reject fields inapplicable to the chosen model architecture.
        always_ok = (
            frozenset({"z_score_input", "z_score_condition", "compose_standardization"})
            | _SCORE_ONLY_FIELDS
            | _FLOW_ONLY_FIELDS
        )
        self._reject_inapplicable_fields(
            _vector_field_build_fns()[self.model],
            discriminator="model",
            always_ok=always_ok,
        )

    def build(
        self, batch_input: Tensor, batch_condition: Tensor
    ) -> ConditionalVectorFieldEstimator:
        """Build the vector-field estimator by dispatching to
        ``build_flow_matching_estimator`` or
        ``build_score_matching_estimator``.

        Args:
            batch_input: Batch of the modeled variable used for
                shape inference and z-scoring.
            batch_condition: Batch of the conditioning variable
                used for shape inference and z-scoring.

        Returns:
            A ``ConditionalVectorFieldEstimator``.
        """
        from sbi.neural_nets.net_builders.vector_field_nets import (
            build_vector_field_estimator,
        )

        if self.estimator_type is None:
            raise ValueError(
                "estimator_type is None. The trainer should resolve this "
                "before calling build(). Use dataclasses.replace() to set "
                "estimator_type='flow' or 'score'."
            )

        kwargs = self._build_kwargs()
        # sde_type is excluded from _build_kwargs() and passed
        # explicitly to build_vector_field_estimator.
        sde_type = self.sde_type or "ve"

        return build_vector_field_estimator(
            batch_x=batch_input,
            batch_y=batch_condition,
            estimator_type=self.estimator_type,
            sde_type=sde_type,
            **kwargs,
        )

    def _build_kwargs(self) -> dict:
        """Non-None fields minus discriminators and sde_type, alias-translated."""
        excluded = self._DISCRIMINATORS | {"sde_type", "extra_kwargs"}
        d = {
            _BUILD_KWARG_ALIASES.get(f.name, f.name): getattr(self, f.name)
            for f in fields(self)
            if f.name not in excluded and getattr(self, f.name) is not None
        }
        # The build function expects `net` for the architecture name.
        d["net"] = self.model
        d.update(self.extra_kwargs)
        return d


@dataclass(frozen=True, eq=False, repr=False)
class MarginalConfigBase:
    """Base configuration for marginal (unconditional) density estimators.

    Marginal estimators model $p(x)$ without a condition, so ``build()`` takes
    only ``batch_x`` and returns an ``UnconditionalDensityEstimator``.  Every
    model is a Zuko flow, selected by the subclass through ``_WHICH_NF``.

    Subclasses add the settings their flow accepts, under Zuko's own parameter
    names.  A setting a flow does not accept is not a field on its config, so
    it raises ``TypeError`` at construction rather than being ignored.  Fields
    carry real defaults, which makes the class the reference for the values a
    build actually uses.

    Args:
        hidden_features: Number of hidden features per transform, or one value
            per transform.
        num_transforms: Number of transforms in the flow.
        z_score_x: Whether to z-score the samples $x$, one of `none`,
            `independent`, or `structured`.  Unconditional flows do not
            implement `transform_to_unconstrained`, so it is not offered.
        extra_kwargs: Additional keyword arguments forwarded to the Zuko flow
            constructor, for settings that have no field of their own.
    """

    hidden_features: Union[int, Sequence[int]] = 50
    num_transforms: int = 5
    z_score_x: Literal["none", "independent", "structured"] = "independent"

    # kw_only so it stays out of the positional argument list, which the
    # per-model settings occupy.
    extra_kwargs: dict = field(default_factory=dict, kw_only=True)

    _WHICH_NF: ClassVar[str]
    """Name of the Zuko flow class this config builds, set by each subclass."""

    def __post_init__(self):
        if type(self) is MarginalConfigBase:
            raise TypeError(
                "MarginalConfigBase only holds the settings shared by all "
                "marginal models. Use a per-model config, e.g. "
                "MarginalNSFConfig()."
            )
        # Python does not check `Literal` values at runtime.
        for f in fields(self):
            allowed = _literal_values(f.type)
            val = getattr(self, f.name)
            if allowed and val not in allowed:
                raise ValueError(
                    f"Invalid value {val!r} for `{f.name}`. "
                    f"Must be one of {sorted(map(str, allowed))}."
                )
        shadowed = set(self.extra_kwargs) & {f.name for f in fields(self)}
        if shadowed:
            raise ValueError(
                f"`extra_kwargs` key(s) {sorted(shadowed)} are fields of "
                f"{type(self).__name__}. Pass them as arguments instead."
            )

    def __repr__(self) -> str:
        parts = []
        for f in fields(self):
            val = getattr(self, f.name)
            default = (
                f.default_factory() if f.default_factory is not MISSING else f.default
            )
            if val == default:
                continue
            parts.append(f"{f.name}={val!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def _build_kwargs(self) -> dict:
        """All fields as builder kwargs, with ``extra_kwargs`` merged in."""
        d = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "extra_kwargs"
        }
        d.update(self.extra_kwargs)
        return d

    def build(self, batch_x: Tensor) -> UnconditionalDensityEstimator:
        """Build the marginal density estimator.

        Args:
            batch_x: Batch of samples $x$, used to infer dimensionality and
                (optional) z-scoring.

        Returns:
            A ``ZukoUnconditionalFlow`` over $x$.
        """
        from sbi.neural_nets.net_builders.flow import build_zuko_unconditional_flow

        return build_zuko_unconditional_flow(
            which_nf=self._WHICH_NF, batch_x=batch_x, **self._build_kwargs()
        )


# Field blocks shared by several Zuko flows. A block exists only where the
# meaning and the default are the same for every flow that uses it; `degree`,
# for instance, is not shared because BPF defaults to 16 and SOSPF to 4.


@dataclass(frozen=True, eq=False, repr=False)
class _ZukoSplineFields:
    """Spline settings shared by the NSF and NCSF flows."""

    bins: int = 10


@dataclass(frozen=True, eq=False, repr=False)
class _ZukoPermutationField:
    """Permutation setting shared by the autoregressive flows."""

    randperm: bool = False


@dataclass(frozen=True, eq=False, repr=False)
class _ZukoMonotonicFields(_ZukoPermutationField):
    """Monotonic-network settings shared by the NAF and UNAF flows."""

    signal: int = 16


@dataclass(frozen=True, eq=False, repr=False)
class MarginalBPFConfig(MarginalConfigBase):
    """Marginal Bernstein polynomial flow.

    Args:
        degree: Degree of the Bernstein polynomial.
    """

    degree: int = 16

    _WHICH_NF: ClassVar[str] = "BPF"


@dataclass(frozen=True, eq=False, repr=False)
class MarginalGFConfig(MarginalConfigBase):
    """Marginal Gaussianization flow.

    Args:
        components: Number of mixture components per Gaussianization step.
    """

    components: int = 8

    _WHICH_NF: ClassVar[str] = "GF"


@dataclass(frozen=True, eq=False, repr=False)
class MarginalMAFConfig(_ZukoPermutationField, MarginalConfigBase):
    """Marginal masked autoregressive flow.

    Args:
        randperm: Whether features are randomly permuted between transforms.
    """

    _WHICH_NF: ClassVar[str] = "MAF"


@dataclass(frozen=True, eq=False, repr=False)
class MarginalNAFConfig(_ZukoMonotonicFields, MarginalConfigBase):
    """Marginal neural autoregressive flow.

    Args:
        randperm: Whether features are randomly permuted between transforms.
        signal: Number of signal features of the monotonic network.
    """

    _WHICH_NF: ClassVar[str] = "NAF"


@dataclass(frozen=True, eq=False, repr=False)
class MarginalNCSFConfig(_ZukoSplineFields, MarginalConfigBase):
    """Marginal neural circular spline flow.

    Args:
        bins: Number of bins of the spline transforms.
    """

    _WHICH_NF: ClassVar[str] = "NCSF"


@dataclass(frozen=True, eq=False, repr=False)
class MarginalNICEConfig(MarginalConfigBase):
    """Marginal non-linear independent components estimation flow.

    Args:
        randmask: Whether the coupling masks are randomly drawn.
    """

    randmask: bool = False

    _WHICH_NF: ClassVar[str] = "NICE"


@dataclass(frozen=True, eq=False, repr=False)
class MarginalNSFConfig(_ZukoSplineFields, MarginalConfigBase):
    """Marginal neural spline flow.

    Args:
        bins: Number of bins of the spline transforms.
    """

    _WHICH_NF: ClassVar[str] = "NSF"


@dataclass(frozen=True, eq=False, repr=False)
class MarginalSOSPFConfig(MarginalConfigBase):
    """Marginal sum-of-squares polynomial flow.

    Args:
        degree: Degree of the polynomials.
        polynomials: Number of polynomials.
    """

    degree: int = 4
    polynomials: int = 3

    _WHICH_NF: ClassVar[str] = "SOSPF"


@dataclass(frozen=True, eq=False, repr=False)
class MarginalUNAFConfig(_ZukoMonotonicFields, MarginalConfigBase):
    """Marginal unconstrained neural autoregressive flow.

    Args:
        randperm: Whether features are randomly permuted between transforms.
        signal: Number of signal features of the monotonic network.
    """

    _WHICH_NF: ClassVar[str] = "UNAF"


MARGINAL_MODELS = Literal[
    "bpf",
    "gf",
    "maf",
    "naf",
    "ncsf",
    "nice",
    "nsf",
    "sospf",
    "unaf",
]

# Kept internal: the per-model configs are the public way to pick a model. This
# dict only backs the deprecated string path and the trainer default.
_MARGINAL_CONFIGS: dict = {
    "bpf": MarginalBPFConfig,
    "gf": MarginalGFConfig,
    "maf": MarginalMAFConfig,
    "naf": MarginalNAFConfig,
    "ncsf": MarginalNCSFConfig,
    "nice": MarginalNICEConfig,
    "nsf": MarginalNSFConfig,
    "sospf": MarginalSOSPFConfig,
    "unaf": MarginalUNAFConfig,
}


def _marginal_config_from_model(model: str) -> MarginalConfigBase:
    """Return the default config of a marginal model given its name.

    Args:
        model: Name of the model, case-insensitive.

    Returns:
        A default-constructed config for that model.
    """
    try:
        config_cls = _MARGINAL_CONFIGS[model.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown marginal model {model!r}. "
            f"Must be one of {sorted(_MARGINAL_CONFIGS)}."
        ) from None
    return config_cls()
