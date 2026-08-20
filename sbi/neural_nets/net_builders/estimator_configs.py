# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Typed dataclass configs for the estimator families.

The per-model configs are how a model is picked and configured: one small class
per model, holding only the fields that model accepts, with real defaults
rather than ``None`` sentinels.  A setting a model does not use is not a field
on its class, so it raises ``TypeError`` at construction instead of being
dropped on the way to the network.  ``_PerModelConfigBase`` holds what every
role needs; the role bases (``DensityConfigBase``, ``ClassifierConfigBase``,
``MarginalConfigBase``, and ``MixedConfig``) add their fields and ``build()``.

Each per-model class adds the settings its own model accepts and points at the
``build_*`` function that consumes them.  A setting a model does not accept is
therefore not a field on its config, and raises ``TypeError`` at construction
rather than being ignored.  Fields carry real defaults, which makes the class
the reference for the values a build actually uses.

The legacy ``*Config`` classes above them are the factory-side validators from
before that change.  They keep one flat field set per family with ``None`` as
the "unset" sentinel, and ``from_kwargs()`` warns on unknown names while still
forwarding them.  Only the paths without a per-model config still use them.

``VectorFieldEstimatorBuilder`` is likewise still a flat builder, with the
signature-derived ``_reject_inapplicable_fields`` guard that the per-model
classes make unnecessary.  Its conversion needs its own design round, because
its build function selects along more than one axis at a time.
"""

import inspect
import warnings
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

import torch
import torch.nn as nn
from nflows.transforms.splines import (
    rational_quadratic,  # pyright: ignore[reportAttributeAccessIssue]
)
from torch import Tensor
from torch.distributions import Distribution

from sbi.neural_nets.estimators import ConditionalVectorFieldEstimator
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimator,
)
from sbi.neural_nets.estimators.mixed_density_estimator import MixedDensityEstimator
from sbi.neural_nets.estimators.zuko_flow import ZukoUnconditionalFlow
from sbi.neural_nets.net_builders.classifier import (
    build_linear_classifier,
    build_mlp_classifier,
    build_resnet_classifier,
)
from sbi.neural_nets.net_builders.flow import (
    build_made,
    build_maf,
    build_maf_rqs,
    build_nsf,
    build_tabpfn_flow,
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


def _check_literal_values(config, allow_none: bool) -> None:
    """Raise if a ``Literal`` field holds a value outside its annotation.

    Python does not check ``Literal`` values at runtime, so a typo like
    ``z_score_input="indepedent"`` would otherwise construct and then be read
    as "do not z-score" downstream.

    Args:
        config: The config instance to check.
        allow_none: Whether ``None`` is a valid value for every field, as it is
            on the configs that use it as the "not set" sentinel.
    """
    for f in fields(config):
        allowed = _literal_values(f.type)
        val = getattr(config, f.name)
        if not allowed or (allow_none and val is None):
            continue
        if val not in allowed:
            raise ValueError(
                f"Invalid value {val!r} for `{f.name}`. "
                f"Must be one of {sorted(map(str, allowed))}"
                f"{' or None' if allow_none else ''}."
            )


@dataclass(frozen=True, eq=False, repr=False)
class _EstimatorBuilderBase:
    """Shared base providing ``from_kwargs()``, ``to_dict()``, and the abstract
    ``build()`` contract for all estimator builders."""

    # kw_only so this base field does not become the first positional
    # parameter of every subclass (it would capture, e.g., a model name).
    extra_kwargs: dict = field(default_factory=dict, kw_only=True)

    def __post_init__(self):
        _check_literal_values(self, allow_none=True)

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
class _PerModelConfigBase:
    """Shared behaviour of the per-model configs.

    Holds what every role needs and nothing a role has to choose: the runtime
    check on ``Literal`` values, the ``extra_kwargs`` escape valve, a ``repr``
    that shows only what differs from the class defaults, and the kwargs
    assembly.  The role bases add their fields and their ``build()``.
    """

    # kw_only so it stays out of the positional argument list, which the
    # per-model settings occupy.
    extra_kwargs: dict = field(default_factory=dict, kw_only=True)

    _SHADOWED_EXTRA_KWARGS: ClassVar[frozenset[str]] = frozenset()

    def _reject_if_abstract(self, base: type, example: str) -> None:
        """Raise if *self* is *base* itself, which selects no model."""
        if type(self) is base:
            raise TypeError(
                f"{base.__name__} only holds the settings shared by a family "
                f"of models. Use a per-model config, e.g. {example}."
            )

    def __post_init__(self):
        _check_literal_values(self, allow_none=False)
        field_names = {f.name for f in fields(self)}
        build_names = {_BUILD_KWARG_ALIASES.get(name, name) for name in field_names}
        shadowed = set(self.extra_kwargs) & (
            field_names | build_names | self._SHADOWED_EXTRA_KWARGS
        )
        if shadowed:
            raise ValueError(
                f"`extra_kwargs` key(s) {sorted(shadowed)} are fields of "
                f"{type(self).__name__} or their downstream aliases. Pass the "
                "user-facing fields instead."
            )

    def __repr__(self) -> str:
        parts = []
        for f in fields(self):
            val = getattr(self, f.name)
            default = (
                f.default_factory() if f.default_factory is not MISSING else f.default
            )
            # Modules do not compare by value, so an untouched `nn.Identity()`
            # would otherwise always show up as a set field.
            if isinstance(default, nn.Module):
                if type(val) is type(default):
                    continue
            elif isinstance(default, _PerModelConfigBase):
                if repr(val) == repr(default):
                    continue
            elif val == default:
                continue
            parts.append(f"{f.name}={val!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def _build_kwargs(self) -> dict:
        """All fields as builder kwargs, with ``extra_kwargs`` merged in.

        Field names are translated via ``_BUILD_KWARG_ALIASES``, so that the
        user-facing ``z_score_input`` reaches the build function as its
        ``z_score_x``.
        """
        d = {
            _BUILD_KWARG_ALIASES.get(f.name, f.name): getattr(self, f.name)
            for f in fields(self)
            if f.name != "extra_kwargs"
        }
        d.update(self.extra_kwargs)
        return d

    def _warn_unknown_extra_kwargs(self, build_fn: Callable) -> None:
        """Warn when an escape-valve key is not explicit in the build signature."""
        if not self.extra_kwargs:
            return
        accepted = {
            name
            for name, param in inspect.signature(build_fn).parameters.items()
            if param.kind is not inspect.Parameter.VAR_KEYWORD
        }
        unknown = set(self.extra_kwargs) - accepted
        if unknown:
            warnings.warn(
                f"Unknown `extra_kwargs` for {type(self).__name__}: "
                f"{sorted(unknown)}. They will be forwarded to the underlying "
                "builder; check for typos if this is unintentional.",
                stacklevel=3,
            )


_ESTIMATOR_CONFIG_BASES = (_EstimatorBuilderBase, _PerModelConfigBase)


def _reject_filtered_extra_kwargs(config) -> None:
    """Raise for ``extra_kwargs`` names the Zuko build functions drop.

    They are not fields, so the shadowing check lets them through, and they
    never reach Zuko, so nothing downstream raises either.

    Args:
        config: The config instance to check.
    """
    from sbi.neural_nets.net_builders.flow import nflow_specific_kwargs

    dropped = set(config.extra_kwargs) & set(nflow_specific_kwargs)
    if dropped:
        raise ValueError(
            f"`extra_kwargs` key(s) {sorted(dropped)} never reach the flow: "
            "they are filtered out before it is built. Use the field of the "
            "same name if the model has one."
        )


@dataclass(frozen=True, eq=False, repr=False)
class MarginalConfigBase(_PerModelConfigBase):
    """Base configuration for marginal (unconditional) density estimators.

    Marginal estimators model $p(x)$ without a condition, so ``build()`` takes
    only ``batch_x`` and returns a ``ZukoUnconditionalFlow``.  Every
    model is a Zuko flow, selected by the subclass through ``_WHICH_NF``, and
    its settings carry Zuko's own parameter names.

    Args:
        hidden_features: Number of hidden features per transform, or one value
            per transform.
        num_transforms: Number of transforms in the flow.
        z_score_input: Whether to z-score the samples $x$, one of `none`,
            `independent`, or `structured`.  Unconditional flows do not
            implement `transform_to_unconstrained`, so it is not offered.
        extra_kwargs: Additional keyword arguments forwarded to the Zuko flow
            constructor, for settings that have no field of their own.
    """

    hidden_features: Union[int, Sequence[int]] = 50
    num_transforms: int = 5
    z_score_input: Literal["none", "independent", "structured"] = "independent"

    _WHICH_NF: ClassVar[str]
    """Name of the Zuko flow class this config builds, set by each subclass."""

    def __post_init__(self):
        self._reject_if_abstract(MarginalConfigBase, "MarginalNSFConfig()")
        super().__post_init__()
        _reject_filtered_extra_kwargs(self)

    def build(self, batch_x: Tensor) -> ZukoUnconditionalFlow:
        """Build the marginal density estimator.

        Args:
            batch_x: Batch of samples $x$, used to infer dimensionality and
                (optional) z-scoring.

        Returns:
            A ``ZukoUnconditionalFlow`` over $x$.
        """
        from sbi.neural_nets.net_builders.flow import build_zuko_unconditional_flow

        self._warn_unknown_extra_kwargs(build_zuko_unconditional_flow)
        return build_zuko_unconditional_flow(
            which_nf=self._WHICH_NF, batch_x=batch_x, **self._build_kwargs()
        )


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

    Zuko forwards the settings it does not name to an element-wise transform,
    which builds no network when there is no condition.  A marginal flow never
    has one, so `hidden_features` and `extra_kwargs` would be dropped on the
    way there and are rejected instead.  Size the flow with `num_transforms`
    and `components`.

    Args:
        components: Number of mixture components per Gaussianization step.
    """

    components: int = 8

    _WHICH_NF: ClassVar[str] = "GF"

    def __post_init__(self):
        super().__post_init__()
        default_width = MarginalConfigBase.hidden_features
        ignored = []
        if self.hidden_features != default_width:
            ignored.append("hidden_features")
        if self.extra_kwargs:
            ignored.append("extra_kwargs")
        if ignored:
            raise ValueError(
                f"GF does not use {ignored}: Zuko passes them to an "
                "element-wise transform, which takes no network without a "
                "condition. Use `num_transforms` or `components` instead."
            )


@dataclass(frozen=True, eq=False, repr=False)
class MarginalMAFConfig(MarginalConfigBase):
    """Marginal masked autoregressive flow.

    Args:
        randperm: Whether features are randomly permuted between transforms.
    """

    randperm: bool = False

    _WHICH_NF: ClassVar[str] = "MAF"


@dataclass(frozen=True, eq=False, repr=False)
class MarginalNAFConfig(MarginalConfigBase):
    """Marginal neural autoregressive flow.

    Args:
        randperm: Whether features are randomly permuted between transforms.
        signal: Number of signal features of the monotonic network.
    """

    randperm: bool = False
    signal: int = 16

    _WHICH_NF: ClassVar[str] = "NAF"


@dataclass(frozen=True, eq=False, repr=False)
class MarginalNCSFConfig(MarginalConfigBase):
    """Marginal neural circular spline flow.

    Args:
        bins: Number of bins of the spline transforms.
    """

    bins: int = 10

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
class MarginalNSFConfig(MarginalConfigBase):
    """Marginal neural spline flow.

    Args:
        bins: Number of bins of the spline transforms.
    """

    bins: int = 10

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
class MarginalUNAFConfig(MarginalConfigBase):
    """Marginal unconstrained neural autoregressive flow.

    Args:
        randperm: Whether features are randomly permuted between transforms.
        signal: Number of signal features of the monotonic network.
    """

    randperm: bool = False
    signal: int = 16

    _WHICH_NF: ClassVar[str] = "UNAF"


MARGINAL_MODELS = Literal[
    "bpf", "gf", "maf", "naf", "ncsf", "nice", "nsf", "sospf", "unaf"
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


@dataclass(frozen=True, eq=False, repr=False)
class _ConditionalDensityConfigBase(_PerModelConfigBase):
    """Shared fields and ``build()`` of the conditional density configs.

    Args:
        z_score_input: Whether to z-score the modeled variable, one of `none`,
            `independent`, or `structured`.  Models that also support
            `transform_to_unconstrained` widen this on their own class.
        z_score_condition: Whether to z-score the conditioning variable, same
            options as `z_score_input`.
        embedding_net: Embedding network for the conditioning variable.
        extra_kwargs: Additional keyword arguments forwarded to the build
            function, for settings that have no field of their own.
    """

    z_score_input: Literal["none", "independent", "structured"] = "independent"
    z_score_condition: Literal["none", "independent", "structured"] = "independent"
    embedding_net: nn.Module = field(default_factory=nn.Identity)

    _BUILD_FN: ClassVar[Callable]
    """Build function this config feeds, set by each subclass."""

    def build(
        self, batch_input: Tensor, batch_condition: Tensor
    ) -> ConditionalDensityEstimator:
        """Build the density estimator.

        Args:
            batch_input: Batch of the modeled variable, used for shape
                inference and z-scoring.
            batch_condition: Batch of the conditioning variable, used for shape
                inference and z-scoring.

        Returns:
            A ``ConditionalDensityEstimator``.
        """
        self._warn_unknown_extra_kwargs(self._BUILD_FN)
        return self._BUILD_FN(
            batch_x=batch_input, batch_y=batch_condition, **self._build_kwargs()
        )


@dataclass(frozen=True, eq=False, repr=False)
class DensityConfigBase(_ConditionalDensityConfigBase):
    """Base configuration for trainable conditional density estimators.

    Used by ``NPE`` and ``NLE``.
    """

    def __post_init__(self):
        self._reject_if_abstract(DensityConfigBase, "NSFConfig()")
        super().__post_init__()


@dataclass(frozen=True, eq=False, repr=False)
class PretrainedConfigBase(_ConditionalDensityConfigBase):
    """Base configuration for pretrained density estimators.

    A pretrained model comes with its weights and has no training loss, so the
    trainers that fit a network reject it and it needs its own trainer.
    """

    _REJECTION_HINT: ClassVar[str] = ""
    """Hint appended when a trainer that fits a network rejects this config."""

    def __post_init__(self):
        self._reject_if_abstract(PretrainedConfigBase, "TabPFNConfig()")
        super().__post_init__()


@dataclass(frozen=True, eq=False, repr=False)
class _UnconstrainedCapableConfigBase(DensityConfigBase):
    """Base for the models that implement `transform_to_unconstrained`.

    Only these derive the input-side reparametrization from a distribution's
    support rather than from batch statistics, so only they offer the extra
    z-scoring mode and the ``x_dist`` that goes with it.

    Args:
        x_dist: Distribution over the modeled variable, used to derive the
            support bounds.  Only used with
            `z_score_input="transform_to_unconstrained"`.
    """

    z_score_input: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent"
    x_dist: Optional[Distribution] = None

    def __post_init__(self):
        super().__post_init__()
        if self.x_dist is not None and self.z_score_input != (
            "transform_to_unconstrained"
        ):
            raise ValueError(
                "`x_dist` is only used with z_score_input='transform_to_unconstrained'."
            )


@dataclass(frozen=True, eq=False, repr=False)
class _ZukoDensityConfigBase(_UnconstrainedCapableConfigBase):
    """Base for the conditional Zuko flows.

    Args:
        hidden_features: Number of hidden features per transform, or one value
            per transform.
        num_transforms: Number of transforms in the flow.
    """

    hidden_features: Union[int, Sequence[int]] = 50
    num_transforms: int = 5

    def __post_init__(self):
        super().__post_init__()
        _reject_filtered_extra_kwargs(self)


@dataclass(frozen=True, eq=False, repr=False)
class ZukoBPFConfig(_ZukoDensityConfigBase):
    """Conditional Bernstein polynomial flow.

    Args:
        degree: Degree of the Bernstein polynomial.
    """

    degree: int = 16

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_zuko_bpf)


@dataclass(frozen=True, eq=False, repr=False)
class ZukoGFConfig(_ZukoDensityConfigBase):
    """Conditional Gaussianization flow.

    Args:
        components: Number of mixture components per Gaussianization step.
    """

    components: int = 8

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_zuko_gf)


@dataclass(frozen=True, eq=False, repr=False)
class ZukoMAFConfig(_ZukoDensityConfigBase):
    """Conditional masked autoregressive flow.

    Args:
        randperm: Whether features are randomly permuted between transforms.
    """

    randperm: bool = False

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_zuko_maf)


@dataclass(frozen=True, eq=False, repr=False)
class ZukoNAFConfig(_ZukoDensityConfigBase):
    """Conditional neural autoregressive flow.

    Args:
        randperm: Whether features are randomly permuted between transforms.
        signal: Number of signal features of the monotonic network.
    """

    randperm: bool = False
    signal: int = 16

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_zuko_naf)


@dataclass(frozen=True, eq=False, repr=False)
class ZukoNCSFConfig(_ZukoDensityConfigBase):
    """Conditional neural circular spline flow.

    Args:
        num_bins: Number of bins of the spline transforms.
    """

    num_bins: int = 10

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_zuko_ncsf)
    _SHADOWED_EXTRA_KWARGS: ClassVar[frozenset[str]] = frozenset({"bins"})


@dataclass(frozen=True, eq=False, repr=False)
class ZukoNICEConfig(_ZukoDensityConfigBase):
    """Conditional non-linear independent components estimation flow.

    Args:
        randmask: Whether the coupling masks are randomly drawn.
    """

    randmask: bool = False

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_zuko_nice)


@dataclass(frozen=True, eq=False, repr=False)
class ZukoNSFConfig(_ZukoDensityConfigBase):
    """Conditional neural spline flow.

    Args:
        num_bins: Number of bins of the spline transforms.
    """

    num_bins: int = 10

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_zuko_nsf)
    _SHADOWED_EXTRA_KWARGS: ClassVar[frozenset[str]] = frozenset({"bins"})


@dataclass(frozen=True, eq=False, repr=False)
class ZukoSOSPFConfig(_ZukoDensityConfigBase):
    """Conditional sum-of-squares polynomial flow.

    Args:
        degree: Degree of the polynomials.
        polynomials: Number of polynomials.
    """

    degree: int = 4
    polynomials: int = 3

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_zuko_sospf)


@dataclass(frozen=True, eq=False, repr=False)
class ZukoUNAFConfig(_ZukoDensityConfigBase):
    """Conditional unconstrained neural autoregressive flow.

    Args:
        randperm: Whether features are randomly permuted between transforms.
        signal: Number of signal features of the monotonic network.
    """

    randperm: bool = False
    signal: int = 16

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_zuko_unaf)


@dataclass(frozen=True, eq=False, repr=False)
class _NFlowsFlowConfigBase(DensityConfigBase):
    """Base for the nflows-based autoregressive flows.

    Args:
        hidden_features: Number of hidden features per transform.
        num_transforms: Number of transforms in the flow.
        num_blocks: Number of residual blocks in each transform's net.
        dropout_probability: Dropout probability in each transform's net.
        use_batch_norm: Whether to use batch normalization.
        dtype: Floating-point dtype of the base distribution.
    """

    hidden_features: int = 50
    num_transforms: int = 5
    num_blocks: int = 2
    dropout_probability: float = 0.0
    use_batch_norm: bool = False
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True, eq=False, repr=False)
class MAFConfig(_NFlowsFlowConfigBase):
    """Masked autoregressive flow with affine transforms."""

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_maf)


@dataclass(frozen=True, eq=False, repr=False)
class MAFRQSConfig(_NFlowsFlowConfigBase):
    """Masked autoregressive flow with rational-quadratic spline transforms.

    Args:
        num_bins: Number of bins of the spline transforms.
        tails: How to handle values outside `tail_bound`.
        tail_bound: Bound beyond which the transform is the identity.
        min_bin_width: Lower bound on the width of a spline bin.
        min_bin_height: Lower bound on the height of a spline bin.
        min_derivative: Lower bound on the derivative at a spline knot.
    """

    num_bins: int = 10
    tails: Optional[str] = "linear"
    tail_bound: float = 3.0
    min_bin_width: float = rational_quadratic.DEFAULT_MIN_BIN_WIDTH
    min_bin_height: float = rational_quadratic.DEFAULT_MIN_BIN_HEIGHT
    min_derivative: float = rational_quadratic.DEFAULT_MIN_DERIVATIVE

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_maf_rqs)


@dataclass(frozen=True, eq=False, repr=False)
class NSFConfig(_NFlowsFlowConfigBase):
    """Neural spline flow.

    Args:
        num_bins: Number of bins of the spline transforms.
        tail_bound: Bound beyond which the transform is the identity.
        hidden_layers_spline_context: Number of hidden layers of the net that
            maps the context to the spline parameters.
    """

    num_bins: int = 10
    tail_bound: float = 3.0
    hidden_layers_spline_context: int = 1

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_nsf)


@dataclass(frozen=True, eq=False, repr=False)
class MADEConfig(DensityConfigBase):
    """Masked autoencoder for distribution estimation, with a mixture output.

    Args:
        hidden_features: Number of hidden features.
        num_mixture_components: Number of components of the output mixture.
    """

    hidden_features: int = 50
    num_mixture_components: int = 10

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_made)


@dataclass(frozen=True, eq=False, repr=False)
class MDNConfig(_UnconstrainedCapableConfigBase):
    """Mixture density network.

    Args:
        hidden_features: Number of hidden features.
        num_components: Number of components of the Gaussian mixture.
    """

    hidden_features: int = 50
    num_components: int = 10

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_mdn)


@dataclass(frozen=True, eq=False, repr=False)
class TabPFNConfig(PretrainedConfigBase):
    """TabPFN-based density estimator.

    TabPFN preprocesses both variables itself, so neither is z-scored and
    `z_score_input` offers `none` only.

    Args:
        regressor_init_kwargs: Keyword arguments passed to `TabPFNRegressor`.
        max_context_size: Maximum number of context samples to store.
    """

    z_score_input: Literal["none"] = "none"
    z_score_condition: Literal["none", "independent", "structured"] = "none"
    regressor_init_kwargs: Optional[dict] = None
    max_context_size: int = 10_000

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_tabpfn_flow)
    _REJECTION_HINT: ClassVar[str] = (
        "TabPFNFlow has no training loss. Use NPE_PFN instead."
    )


# Kept internal: the per-model configs are the public way to pick a model. This
# dict only backs the deprecated string paths, the factories, and the defaults.
_DENSITY_CONFIGS: dict = {
    "mdn": MDNConfig,
    "made": MADEConfig,
    "maf": MAFConfig,
    "maf_rqs": MAFRQSConfig,
    "nsf": NSFConfig,
    "zuko_nice": ZukoNICEConfig,
    "zuko_maf": ZukoMAFConfig,
    "zuko_nsf": ZukoNSFConfig,
    "zuko_ncsf": ZukoNCSFConfig,
    "zuko_sospf": ZukoSOSPFConfig,
    "zuko_naf": ZukoNAFConfig,
    "zuko_unaf": ZukoUNAFConfig,
    "zuko_gf": ZukoGFConfig,
    "zuko_bpf": ZukoBPFConfig,
    "tabpfn": TabPFNConfig,
}


@dataclass(frozen=True, eq=False, repr=False)
class ClassifierConfigBase(_PerModelConfigBase):
    r"""Base configuration for ratio estimators / classifiers (NRE).

    Args:
        z_score_input: Whether to z-score the parameters $\theta$, one of
            `none`, `independent`, or `structured`.
        z_score_condition: Whether to z-score the data $x$, same options as
            `z_score_input`.
        embedding_net_theta: Embedding network for the parameters $\theta$.
        embedding_net_x: Embedding network for the data $x$.
        extra_kwargs: Additional keyword arguments forwarded to the build
            function, for settings that have no field of their own.
    """

    z_score_input: Literal["none", "independent", "structured"] = "independent"
    z_score_condition: Literal["none", "independent", "structured"] = "independent"
    # User-facing semantics follow `classifier_nn`: embedding_net_theta embeds
    # the parameters, embedding_net_x the data. The build functions use
    # positional x/y naming instead, translated in `_build_kwargs`.
    embedding_net_theta: nn.Module = field(default_factory=nn.Identity)
    embedding_net_x: nn.Module = field(default_factory=nn.Identity)

    _BUILD_FN: ClassVar[Callable]
    """Build function this config feeds, set by each subclass."""

    def __post_init__(self):
        self._reject_if_abstract(ClassifierConfigBase, "ResNetClassifierConfig()")
        super().__post_init__()
        if "embedding_net_y" in self.extra_kwargs:
            raise ValueError(
                "`extra_kwargs` key 'embedding_net_y' duplicates "
                "`embedding_net_x`. Pass the user-facing field instead."
            )

    def _build_kwargs(self) -> dict:
        """Translate the embedding names to the build functions' x/y naming."""
        kwargs = super()._build_kwargs()
        kwargs["embedding_net_y"] = kwargs.pop("embedding_net_x")
        kwargs["embedding_net_x"] = kwargs.pop("embedding_net_theta")
        return kwargs

    def build(self, batch_input: Tensor, batch_condition: Tensor) -> RatioEstimator:
        r"""Build the classifier.

        Args:
            batch_input: Batch of the parameters $\theta$, used for shape
                inference and z-scoring.
            batch_condition: Batch of the data $x$, used for shape inference
                and z-scoring.

        Returns:
            A ``RatioEstimator``.
        """
        self._warn_unknown_extra_kwargs(self._BUILD_FN)
        return self._BUILD_FN(
            batch_x=batch_input, batch_y=batch_condition, **self._build_kwargs()
        )


@dataclass(frozen=True, eq=False, repr=False)
class LinearClassifierConfig(ClassifierConfigBase):
    """Linear classifier."""

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_linear_classifier)


@dataclass(frozen=True, eq=False, repr=False)
class MLPClassifierConfig(ClassifierConfigBase):
    """Multi-layer perceptron classifier.

    Args:
        hidden_features: Number of hidden features.
        norm_layer: Normalization layer, constructed from a feature count.
    """

    hidden_features: int = 50
    norm_layer: Callable[[int], nn.Module] = nn.LayerNorm

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_mlp_classifier)


@dataclass(frozen=True, eq=False, repr=False)
class ResNetClassifierConfig(ClassifierConfigBase):
    """Residual-network classifier.

    Args:
        hidden_features: Number of hidden features.
        num_blocks: Number of residual blocks.
        dropout_probability: Dropout probability in each block.
        use_batch_norm: Whether to use batch normalization.
    """

    hidden_features: int = 50
    num_blocks: int = 2
    dropout_probability: float = 0.0
    use_batch_norm: bool = False

    _BUILD_FN: ClassVar[Callable] = staticmethod(build_resnet_classifier)


_CLASSIFIER_CONFIGS: dict = {
    "linear": LinearClassifierConfig,
    "mlp": MLPClassifierConfig,
    "resnet": ResNetClassifierConfig,
}

# TabPFN is the one density model the mixed build function cannot use.
_MIXED_CONTINUOUS_CONFIGS: frozenset = frozenset(
    cls for cls in _DENSITY_CONFIGS.values() if issubclass(cls, DensityConfigBase)
)


def _default_mixed_continuous() -> DensityConfigBase:
    """Return the continuous config MNLE and MNPE have always used.

    The mixed build function overrode the spline tail bound with 10.0, so the
    default carries that rather than `NSFConfig`'s own 3.0.
    """
    return NSFConfig(tail_bound=10.0)


@dataclass(frozen=True, eq=False, repr=False)
class MixedConfig(_PerModelConfigBase):
    """Configuration for mixed density estimators (MNLE / MNPE).

    The continuous component is configured by nesting the config of the model
    that estimates it, so its settings are validated by that model's own class.
    The z-scoring of the modeled variable therefore lives on the nested config;
    `z_score_condition` here applies to the conditioning variable.

    Args:
        continuous: Config of the model for the continuous variables.
        z_score_condition: Whether to z-score the conditioning variable, one of
            `none`, `independent`, or `structured`.
        num_categories_per_variable: Number of categories of each discrete
            variable.  Inferred from the data when None.
        embedding_net: Embedding network for the conditioning variable.
        combined_embedding_net: Network that combines the embedded condition
            with the discrete variables.  Built automatically when None.
        log_transform_x: Whether to log-transform the continuous variables.
        discrete_hidden_features: Number of hidden features of the categorical
            net.  Falls back to the continuous config's when None, so that a
            wider continuous net does not leave the discrete one behind.
        discrete_hidden_layers: Number of hidden layers of the categorical net.
        combined_embedding_features: Number of hidden features of the combined
            embedding net built when `combined_embedding_net` is None.  Falls
            back to the continuous config's when None.
        dropout_probability: Dropout probability of the categorical net.
        extra_kwargs: Unsupported for mixed estimators because there is no
            unambiguous downstream target. Put continuous-model options in
            ``continuous.extra_kwargs`` instead.
    """

    continuous: DensityConfigBase = field(default_factory=_default_mixed_continuous)
    z_score_condition: Literal["none", "independent", "structured"] = "independent"
    num_categories_per_variable: Optional[Tensor] = None
    embedding_net: nn.Module = field(default_factory=nn.Identity)
    combined_embedding_net: Optional[nn.Module] = None
    log_transform_x: bool = False
    discrete_hidden_features: Optional[int] = None
    discrete_hidden_layers: int = 2
    combined_embedding_features: Optional[int] = None
    dropout_probability: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if type(self.continuous) not in _MIXED_CONTINUOUS_CONFIGS:
            allowed = sorted(c.__name__ for c in _MIXED_CONTINUOUS_CONFIGS)
            raise TypeError(
                f"{type(self.continuous).__name__} cannot estimate the "
                f"continuous component of a mixed estimator. Use one of "
                f"{allowed}."
            )
        # Both nets fall back to the continuous net's width, which can only be
        # read from a single number.
        falls_back = self.discrete_hidden_features is None or (
            self.combined_embedding_net is None
            and self.combined_embedding_features is None
        )
        if falls_back and not isinstance(self.continuous.hidden_features, int):  # type: ignore[attr-defined]
            raise ValueError(
                "The continuous config of a mixed estimator needs a single "
                "`hidden_features` value, because `discrete_hidden_features` "
                "and `combined_embedding_features` fall back to it. Set them "
                "explicitly to use a per-transform width."
            )
        if (
            self.continuous.z_score_condition != "independent"
            or type(self.continuous.embedding_net) is not nn.Identity
        ):
            raise ValueError(
                "The continuous config's `z_score_condition` and `embedding_net` "
                "are replaced when its mixed condition is built. Configure them "
                "with `MixedConfig.z_score_condition`, `embedding_net`, and "
                "`combined_embedding_net` instead."
            )
        if self.extra_kwargs:
            raise ValueError(
                "MixedConfig has no downstream pass-through for `extra_kwargs`. "
                "Put continuous-model options in `continuous.extra_kwargs`."
            )

    def build(
        self, batch_input: Tensor, batch_condition: Tensor
    ) -> MixedDensityEstimator:
        """Build the mixed density estimator.

        Args:
            batch_input: Batch of the modeled variable, used for shape
                inference and z-scoring.
            batch_condition: Batch of the conditioning variable, used for shape
                inference and z-scoring.

        Returns:
            A ``MixedDensityEstimator``.
        """
        from sbi.neural_nets.net_builders.mixed_nets import (
            _build_mixed_density_estimator,
        )

        return _build_mixed_density_estimator(
            batch_x=batch_input,
            batch_y=batch_condition,
            config=self,
        )


def _factory_defaults(factory: Callable, field_to_param: dict) -> dict:
    """Return each config field's default, read from the factory's signature.

    Args:
        factory: The factory function the arguments came from.
        field_to_param: Maps a config field name to the factory's name for it.

    Returns:
        The factory's default for each field.
    """
    params = inspect.signature(factory).parameters
    return {field: params[param].default for field, param in field_to_param.items()}


def _config_from_factory_kwargs(
    model: str,
    configs: dict,
    role: str,
    family_args: dict,
    factory_defaults: dict,
    extra: dict,
) -> _PerModelConfigBase:
    """Build a per-model config from a factory's arguments.

    A factory's arguments span the whole family, so a model that does not use
    one of them gets it dropped while it still holds the factory's own default,
    and rejected once the caller has set it: that value would otherwise be
    silently ignored.  An argument still at the factory's default is not
    forwarded either, so that a model narrowing a field keeps its own default.
    Names no model knows keep the factories' warn-and-forward behaviour through
    ``extra_kwargs``.

    Args:
        model: Name of the model to configure.
        configs: Registry to look the name up in.
        role: Name of the estimator family, for the error messages.
        family_args: The factory's family-wide arguments, under the config's
            field names.
        factory_defaults: Default of each family-wide argument.
        extra: The factory's ``**kwargs``.

    Returns:
        The config for that model.
    """
    try:
        config_cls = configs[model]
    except KeyError:
        raise ValueError(
            f"Unknown {role} model {model!r}. Must be one of {sorted(configs)}."
        ) from None

    known = {f.name for f in fields(config_cls)} - {"extra_kwargs"}
    family_known = {f.name for cls in configs.values() for f in fields(cls)} - {
        "extra_kwargs"
    }
    accepted, ignored = {}, []
    for name, value in family_args.items():
        if value == factory_defaults[name]:
            # Untouched by the caller, so the config's own default stands.
            continue
        if name in known:
            accepted[name] = value
        else:
            ignored.append(name)
    unknown = {}
    for name, value in extra.items():
        if name in known:
            accepted[name] = value
        elif name in family_known:
            ignored.append(name)
        else:
            unknown[name] = value
    if ignored:
        raise ValueError(
            f"Argument(s) {sorted(ignored)} are not used by model={model!r} "
            f"and would be silently ignored. Configure the model directly with "
            f"{config_cls.__name__}, or use `extra_kwargs` to forward "
            f"library-specific options."
        )
    if unknown:
        warnings.warn(
            f"Unknown kwargs passed to {config_cls.__name__}: "
            f"{sorted(unknown)}. "
            f"These will be forwarded to the underlying builder. "
            f"If this is unintentional, check for typos.",
            stacklevel=3,
        )

    return config_cls(**accepted, extra_kwargs=unknown)


def _mixed_config_from_factory_kwargs(
    family_args: dict,
    factory_defaults: dict,
    extra: dict,
) -> MixedConfig:
    """Build a mixed config from the deprecated factories' flat arguments."""
    extra = dict(extra)
    flow_model = extra.pop("flow_model", None)
    if flow_model is None:
        flow_model = "nsf"
    config_cls = _DENSITY_CONFIGS.get(flow_model)

    mixed_fields = (
        "num_categories_per_variable",
        "combined_embedding_net",
        "log_transform_x",
        "discrete_hidden_features",
        "discrete_hidden_layers",
        "combined_embedding_features",
        "dropout_probability",
    )
    # The flat path dropped a recognised None before building, so it still
    # means unset. A name no model knows keeps its None, and with it the
    # warning that catches a typo.
    recognised = {"continuous_hidden_features", *mixed_fields}
    if config_cls is not None:
        recognised |= {f.name for f in fields(config_cls)}
    extra = {
        name: value
        for name, value in extra.items()
        if value is not None or name not in recognised
    }

    mixed_kwargs = {
        "z_score_condition": family_args["z_score_condition"],
        "embedding_net": family_args["embedding_net"],
    }
    for name in mixed_fields:
        if name in extra:
            mixed_kwargs[name] = extra.pop(name)

    continuous_args = {
        name: family_args[name]
        for name in (
            "z_score_input",
            "hidden_features",
            "num_transforms",
            "num_bins",
            "num_components",
        )
    }
    continuous_defaults = {name: factory_defaults[name] for name in continuous_args}
    continuous_args = {
        name: continuous_defaults[name] if value is None else value
        for name, value in continuous_args.items()
    }

    if "continuous_hidden_features" in extra:
        # The flat API let this override only the continuous width while the
        # categorical width kept falling back to `hidden_features`.
        mixed_kwargs.setdefault(
            "discrete_hidden_features", continuous_args["hidden_features"]
        )
        continuous_args["hidden_features"] = extra.pop("continuous_hidden_features")

    dropout_probability = mixed_kwargs.get("dropout_probability")
    if (
        dropout_probability is not None
        and config_cls is not None
        and "dropout_probability" in {f.name for f in fields(config_cls)}
    ):
        extra["dropout_probability"] = dropout_probability

    # The flat mixed API passed `tail_bound` to every builder, so the models
    # that read it saw 10.0 rather than their own narrower default.
    if config_cls is not None and "tail_bound" in {f.name for f in fields(config_cls)}:
        extra.setdefault("tail_bound", 10.0)

    continuous = _config_from_factory_kwargs(
        flow_model,
        _DENSITY_CONFIGS,
        "mixed continuous density",
        family_args=continuous_args,
        factory_defaults=continuous_defaults,
        extra=extra,
    )
    return MixedConfig(continuous=continuous, **mixed_kwargs)
