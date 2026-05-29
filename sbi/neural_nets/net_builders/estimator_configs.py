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

from dataclasses import dataclass, fields
from typing import Any, Optional

from sbi.neural_nets.build_context import BuildContext


@dataclass
class _EstimatorBuilderBase:
    """Shared base providing ``from_kwargs()``, ``to_dict()``, and the abstract
    ``build()`` contract for all estimator builders."""

    extra_kwargs: dict = None  # type: ignore

    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}

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

    def build(self, context: BuildContext) -> Any:
        """Build an estimator from a ``BuildContext``.

        Subclasses must override this method to construct the appropriate
        estimator using the shapes, device, and z-score stats in *context*.

        Args:
            context: Data-derived state (shapes, device, z-score stats).

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


@dataclass
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


@dataclass
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


@dataclass
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
