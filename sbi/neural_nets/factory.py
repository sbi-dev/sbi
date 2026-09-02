# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from dataclasses import replace
from enum import Enum
from typing import Any, Callable, Literal, Optional, Union

from torch import Tensor, nn

from sbi.neural_nets.net_builders.estimator_configs import (
    VF_MODELS,
    _BUILD_KWARG_ALIASES,
    _CLASSIFIER_CONFIGS,
    _DENSITY_CONFIGS,
    ConditionalFlowConfig,
    MarginalFlowConfig,
    _config_from_factory_kwargs,
    _factory_defaults,
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
    build_zuko_unconditional_flow,
)
from sbi.neural_nets.net_builders.mdn import build_mdn
from sbi.neural_nets.net_builders.mixed_nets import build_mnle, build_mnpe
from sbi.neural_nets.net_builders.vector_field_nets import (
    FlowEstimatorConfig,
    ScoreEstimatorConfig,
    build_vector_field_estimator,
)
from sbi.utils.nn_utils import check_net_device
from sbi.utils.vector_field_utils import VectorFieldNet

model_builders = {
    "mdn": build_mdn,
    "made": build_made,
    "maf": build_maf,
    "maf_rqs": build_maf_rqs,
    "nsf": build_nsf,
    "mnle": build_mnle,
    "mnpe": build_mnpe,
    "zuko_nice": build_zuko_nice,
    "zuko_maf": build_zuko_maf,
    "zuko_nsf": build_zuko_nsf,
    "zuko_ncsf": build_zuko_ncsf,
    "zuko_sospf": build_zuko_sospf,
    "zuko_naf": build_zuko_naf,
    "zuko_unaf": build_zuko_unaf,
    "zuko_gf": build_zuko_gf,
    "zuko_bpf": build_zuko_bpf,
    "tabpfn": build_tabpfn_flow,
}


# TODO: currently only used for marginal_nn, adapt to use for all
class ZukoFlowType(Enum):
    """Enumeration of Zuko flow types."""

    BPF = "bpf"
    GF = "gf"
    MAF = "maf"
    NAF = "naf"
    NCSF = "ncsf"
    NICE = "nice"
    NSF = "nsf"
    SOSPF = "sospf"
    UNAF = "unaf"


embedding_net_warn_msg = """The passed embedding net will be moved to cpu for
                        constructing the net building function."""

# Maps a config field to the name the factory exposes it under, so that the
# factory's own defaults can be read back from its signature.
_CLASSIFIER_FACTORY_FIELDS: dict = {
    "z_score_input": "z_score_theta",
    "z_score_condition": "z_score_x",
    "hidden_features": "hidden_features",
    "embedding_net_theta": "embedding_net_theta",
    "embedding_net_x": "embedding_net_x",
}

_POSTERIOR_FACTORY_FIELDS: dict = {
    "z_score_input": "z_score_theta",
    "z_score_condition": "z_score_x",
    "hidden_features": "hidden_features",
    "num_transforms": "num_transforms",
    "num_bins": "num_bins",
    "embedding_net": "embedding_net",
    "num_components": "num_components",
}

# NLE models p(x|theta), so the two z-scoring arguments swap roles.
_LIKELIHOOD_FACTORY_FIELDS: dict = {
    **_POSTERIOR_FACTORY_FIELDS,
    "z_score_input": "z_score_x",
    "z_score_condition": "z_score_theta",
}


_Z_SCORE_FIELDS: frozenset = frozenset({"z_score_input", "z_score_condition"})


def _normalize_z_scoring(family_args: dict) -> dict:
    """Map the z-scoring arguments left at ``None`` to ``"none"``.

    The factories document ``None`` as "do not z-score", and forwarded it to
    ``z_score_parser``, which reads it that way. The configs do not accept it,
    so it is translated here rather than reaching them.
    """
    return {
        k: ("none" if v is None and k in _Z_SCORE_FIELDS else v)
        for k, v in family_args.items()
    }


def _density_family_args(embedding_net: nn.Module, **family_args: Any) -> dict:
    """Return the density factories' family-wide arguments as config fields."""
    return _normalize_z_scoring(
        dict(
            family_args,
            embedding_net=check_net_device(
                embedding_net, "cpu", embedding_net_warn_msg
            ),
        )
    )


def _legacy_density_build_fn(
    model: str,
    family_args: dict,
    extra: dict,
    input_is_theta: bool,
) -> Callable:
    """Return a build function for the models that have no config yet.

    ``mnle`` and ``mnpe`` are reached through the density factories by the
    deprecated string path of MNLE and MNPE, and are built from flat kwargs by
    ``build_mnle`` / ``build_mnpe`` rather than from a config.
    """
    config = ConditionalFlowConfig.from_kwargs(
        **{_BUILD_KWARG_ALIASES.get(k, k): v for k, v in family_args.items()},
        **extra,
    )
    builder_kwargs = config.to_dict()

    def build_fn(batch_theta, batch_x):
        if model not in model_builders:
            raise NotImplementedError(f"Model {model} is not implemented")

        modeled, condition = (
            (batch_theta, batch_x) if input_is_theta else (batch_x, batch_theta)
        )
        return model_builders[model](
            batch_x=modeled, batch_y=condition, **builder_kwargs
        )

    return build_fn


def classifier_nn(
    model: str,
    z_score_theta: Optional[
        Literal["independent", "structured", "none"]
    ] = "independent",
    z_score_x: Optional[Literal["independent", "structured", "none"]] = "independent",
    hidden_features: int = 50,
    embedding_net_theta: nn.Module = nn.Identity(),
    embedding_net_x: nn.Module = nn.Identity(),
    **kwargs: Any,
) -> Callable:
    r"""
    Returns a function that builds a classifier for learning density ratios.

    This function will usually be used for SNRE. The returned function is to be passed
    to the inference class when using the flexible interface.

    Note that in the view of the SNRE classifier we build below, x=theta and y=x.

    Args:
        model: The type of classifier that will be created. One of [`linear`, `mlp`,
            `resnet`].
        z_score_theta: Whether to z-score parameters $\theta$ before passing them into
            the network, can take one of the following:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network, with the same options as `z_score_theta`.
        hidden_features: Number of hidden features.
        embedding_net_theta:  Optional embedding network for parameters $\theta$.
        embedding_net_x:  Optional embedding network for simulation outputs $x$. This
            embedding net allows to learn features from potentially high-dimensional
            simulation outputs.
        **kwargs: Additional classifier arguments.  Valid keys are the fields of
            the chosen model's config; a key the model does not use raises, and
            an unknown key triggers a warning and is forwarded to the builder.
    """

    # Map user-facing parameter names to the config's field names.
    family_args = _normalize_z_scoring(
        dict(
            z_score_input=z_score_theta,
            z_score_condition=z_score_x,
            hidden_features=hidden_features,
            embedding_net_theta=check_net_device(
                embedding_net_theta, "cpu", embedding_net_warn_msg
            ),
            embedding_net_x=check_net_device(
                embedding_net_x, "cpu", embedding_net_warn_msg
            ),
        )
    )
    config = _config_from_factory_kwargs(
        model,
        _CLASSIFIER_CONFIGS,
        "classifier",
        family_args=family_args,
        factory_defaults=_factory_defaults(classifier_nn, _CLASSIFIER_FACTORY_FIELDS),
        extra=kwargs,
    )

    def build_fn(batch_theta, batch_x):
        return config.build(batch_input=batch_theta, batch_condition=batch_x)

    return build_fn


def likelihood_nn(
    model: str,
    z_score_theta: Optional[
        Literal["independent", "structured", "none"]
    ] = "independent",
    z_score_x: Optional[
        Literal["independent", "structured", "transform_to_unconstrained", "none"]
    ] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    num_components: int = 10,
    **kwargs: Any,
) -> Callable:
    r"""
    Returns a function that builds a density estimator for learning the likelihood.

    This function will usually be used for SNLE. The returned function is to be passed
    to the inference class when using the flexible interface.

    Args:
        model: The type of density estimator that will be created. One of [`mdn`,
            `made`, `maf`, `maf_rqs`, `nsf`].
        z_score_theta: Whether to z-score parameters $\theta$ before passing them into
            the network, can take one of the following:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network, with the same options as `z_score_theta`. Supported flow
            configs additionally accept `transform_to_unconstrained` for this modeled
            variable.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms when a flow is used. Only relevant if
            density estimator is a normalizing flow (i.e. currently either a `maf` or a
            `nsf`). A non-default value raises if the chosen model does not use it.
        num_bins: Number of bins used for spline models. A non-default value raises
            if the chosen model does not use it.
        embedding_net: Optional embedding network for parameters $\theta$.
        num_components: Number of mixture components for a mixture of Gaussians.
            A non-default value raises if the chosen model is not an MDN.
        **kwargs: Additional estimator arguments.  Valid keys are the fields of
            the chosen model's config; a key the model does not use raises, and
            an unknown key triggers a warning and is forwarded to the builder.
    """

    family_args = _density_family_args(
        z_score_input=z_score_x,
        z_score_condition=z_score_theta,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_bins=num_bins,
        embedding_net=embedding_net,
        num_components=num_components,
    )
    if model not in _DENSITY_CONFIGS:
        return _legacy_density_build_fn(
            model, family_args, kwargs, input_is_theta=False
        )

    config = _config_from_factory_kwargs(
        model,
        _DENSITY_CONFIGS,
        "density",
        family_args=family_args,
        factory_defaults=_factory_defaults(likelihood_nn, _LIKELIHOOD_FACTORY_FIELDS),
        extra=kwargs,
    )

    def build_fn(batch_theta, batch_x):
        # NLE models p(x|theta), so the modeled variable is x.
        return config.build(batch_input=batch_x, batch_condition=batch_theta)

    return build_fn


def posterior_nn(
    model: str,
    z_score_theta: Optional[
        Literal["independent", "structured", "transform_to_unconstrained", "none"]
    ] = "independent",
    z_score_x: Optional[Literal["independent", "structured", "none"]] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    num_components: int = 10,
    **kwargs: Any,
) -> Callable:
    r"""
    Returns a function that builds a density estimator for learning the posterior.

    This function will usually be used for SNPE. The returned function is to be passed
    to the inference class when using the flexible interface.

    Args:
        model: The type of density estimator that will be created. One of [`mdn`,
            `made`, `maf`, `maf_rqs`, `nsf`].
        z_score_theta: Whether to z-score parameters $\theta$ before passing them into
            the network, can take one of the following:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
            Supported flow configs additionally accept `transform_to_unconstrained`
            for this modeled variable.
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network: `none`, `independent`, or `structured`.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms when a flow is used. Only relevant if
            density estimator is a normalizing flow (i.e. currently either a `maf` or a
            `nsf`). A non-default value raises if the chosen model does not use it.
        num_bins: Number of bins used for spline models. A non-default value raises
            if the chosen model does not use it.
        embedding_net: Optional embedding network for simulation outputs $x$. This
            embedding net allows to learn features from potentially high-dimensional
            simulation outputs.
        num_components: Number of mixture components for a mixture of Gaussians.
            A non-default value raises if the chosen model is not an MDN.
        **kwargs: Additional estimator arguments.  Valid keys are the fields of
            the chosen model's config; a key the model does not use raises, and
            an unknown key triggers a warning and is forwarded to the builder.
    """

    family_args = _density_family_args(
        z_score_input=z_score_theta,
        z_score_condition=z_score_x,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_bins=num_bins,
        embedding_net=embedding_net,
        num_components=num_components,
    )

    if model == "mdn_snpe_a":
        if num_components != 10:
            raise ValueError(
                "You set `num_components`. For NPE-A, this has to be done at "
                "instantiation of the inference object, i.e. "
                "`inference = NPE_A(..., num_components=20)`"
            )
        # NPE-A overrides the number of components per round, so it stays a
        # call-time argument rather than a field of the config.
        base = _config_from_factory_kwargs(
            "mdn",
            _DENSITY_CONFIGS,
            "density",
            family_args={k: v for k, v in family_args.items() if k != "num_components"},
            factory_defaults=_factory_defaults(posterior_nn, _POSTERIOR_FACTORY_FIELDS),
            extra=kwargs,
        )

        def build_fn_snpe_a(batch_theta, batch_x, num_components):
            """Build function for SNPE-A.

            ``num_components`` stays a call-time argument so that it can later be
            overridden with `functools.partial`. This is necessary in order to
            make sure that the MDN in SNPE-A only has one component when running
            the Algorithm 1 part.
            """
            return replace(base, num_components=num_components).build(
                batch_input=batch_theta, batch_condition=batch_x
            )

        return build_fn_snpe_a

    if model not in _DENSITY_CONFIGS:
        return _legacy_density_build_fn(model, family_args, kwargs, input_is_theta=True)

    config = _config_from_factory_kwargs(
        model,
        _DENSITY_CONFIGS,
        "density",
        family_args=family_args,
        factory_defaults=_factory_defaults(posterior_nn, _POSTERIOR_FACTORY_FIELDS),
        extra=kwargs,
    )

    def build_fn(batch_theta, batch_x):
        # NPE models p(theta|x), so the modeled variable is theta.
        return config.build(batch_input=batch_theta, batch_condition=batch_x)

    return build_fn


def posterior_score_nn(
    model: Union[
        VF_MODELS,
        VectorFieldNet,
    ] = "mlp",
    sde_type: str = "ve",
    z_score_theta: Optional[
        Literal["independent", "structured", "transform_to_unconstrained", "none"]
    ] = "independent",
    z_score_x: Optional[
        Literal["independent", "structured", "transform_to_unconstrained", "none"]
    ] = "independent",
    hidden_features: int = 100,
    num_layers: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    time_emb_type: Literal["sinusoidal", "random_fourier"] = "sinusoidal",
    t_embedding_dim: int = 32,
    compose_standardization: bool = False,
    **kwargs: Any,
) -> Callable:
    """Build util function that builds a ScoreEstimator object for score-based
    posteriors.

    Args:
        sde_type: SDE type used, which defines the mean and std functions. One of:
            - 'vp': Variance preserving.
            - 'subvp': Sub-variance preserving.
            - 've': Variance exploding.
            Defaults to 've'.
        model: Type of regression network. One of:
            - 'mlp': Fully connected feed-forward network.
            - 'ada_mlp': Fully connected feed-forward with adaptive
               layer normalization for conditioning.
            - 'transformer': Transformer network.
            - 'transformer_cross_attn': Transformer with cross-attention.
                Requires sequence-shaped conditioning (3-D ``batch_y`` or an
                ``embedding_net`` returning ``(batch, seq_len, emb_dim)``).
            -  nn.Module: Custom network
            Defaults to 'mlp'.
        z_score_theta: Whether to z-score thetas passing into the network, can be one
            of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_x: Whether to z-score xs passing into the network, same options as
            z_score_theta.
        hidden_features: Number of hidden units per layer. Defaults to 100.
        embedding_net: Embedding network for x (conditioning variable). Defaults to
            nn.Identity().
        time_emb_type: Type of time embedding. Defaults to 'sinusoidal'.
        t_embedding_dim: Embedding dimension of diffusion time. Defaults to 32.
        compose_standardization: Opt-in per-dim affine standardization theta<->z
            for scale-equivariant calibration. Defaults to False.
        **kwargs: Additional estimator / network arguments.  Valid keys are
            defined by ``ScoreEstimatorConfig``; unknown keys raise
            ``TypeError``.

    Returns:
        Constructor function for NPSE.
    """
    if compose_standardization and z_score_theta != "independent":
        raise ValueError(
            "compose_standardization=True requires z_score_theta='independent'."
        )

    # Map user-facing parameter names to internal names.
    # Builder takes batch_x=batch_theta, so its z_score_x is the theta setting.
    mapped = dict(
        z_score_x=z_score_theta,
        z_score_y=z_score_x,
        hidden_features=hidden_features,
        num_layers=num_layers,
        embedding_net=check_net_device(embedding_net, "cpu", embedding_net_warn_msg),
        time_embedding_dim=t_embedding_dim,
        time_emb_type=time_emb_type,
        net=model,
        compose_standardization=compose_standardization,
    )

    # Validate against known fields — warns on unknown kwargs (typos)
    # while still forwarding them to the underlying builder.
    config = ScoreEstimatorConfig.from_kwargs(**mapped, **kwargs)
    builder_kwargs = config.to_dict()

    def build_fn(batch_theta, batch_x):
        return build_vector_field_estimator(
            batch_x=batch_theta,
            batch_y=batch_x,
            estimator_type="score",
            sde_type=sde_type,
            **builder_kwargs,
        )

    return build_fn


def posterior_flow_nn(
    model: Union[
        VF_MODELS,
        VectorFieldNet,
    ] = "mlp",
    z_score_theta: Optional[
        Literal["independent", "structured", "transform_to_unconstrained", "none"]
    ] = "independent",
    z_score_x: Optional[
        Literal["independent", "structured", "transform_to_unconstrained", "none"]
    ] = "independent",
    hidden_features: int = 100,
    num_layers: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    time_emb_type: Literal["sinusoidal", "random_fourier"] = "sinusoidal",
    t_embedding_dim: int = 32,
    gaussian_baseline: bool = False,
    compose_standardization: bool = False,
    **kwargs: Any,
) -> Callable:
    """Build util function that builds a FlowMatchingEstimator object for flow-based
    posteriors.

    Args:
        model: Type of regression network. One of:
            - 'mlp': Fully connected feed-forward network.
            - 'ada_mlp': Fully connected feed-forward with adaptive
                layer normalization for conditioning.
            - 'transformer': Transformer network.
            - 'transformer_cross_attn': Transformer with cross-attention.
                Requires sequence-shaped conditioning (3-D ``batch_y`` or an
                ``embedding_net`` returning ``(batch, seq_len, emb_dim)``).
            -  nn.Module: Custom network
            Defaults to 'mlp'.
        z_score_theta: Whether to z-score theta for time-dependent normalization.
            This enables time-dependent z-scoring which helps FMPE learn when
            theta is far from N(0,1). Defaults to 'independent'.
        z_score_x: Whether to z-score observations (x) before passing to the
            embedding network. Defaults to 'independent'.
        hidden_features: Number of hidden units per layer. Defaults to 100.
        num_layers: Number of hidden layers. Defaults to 5.
        embedding_net: Embedding network for x (conditioning variable). Defaults to
            nn.Identity().
        time_emb_type: Type of time embedding. Defaults to 'sinusoidal'.
        t_embedding_dim: Embedding dimension of diffusion time. Defaults to 32.
        gaussian_baseline: If True, use analytical Gaussian baseline velocity
            derived from Bayes' rule. The network then only learns the residual.
            Defaults to False.
        compose_standardization: Opt-in per-dim affine standardization theta<->z
            for scale-equivariant calibration. Defaults to False.
        **kwargs: Additional estimator / network arguments.  Valid keys are
            defined by ``FlowEstimatorConfig``; unknown keys raise
            ``TypeError``.

    Returns:
        Constructor function for FMPE.
    """
    if compose_standardization and z_score_theta != "independent":
        raise ValueError(
            "compose_standardization=True requires z_score_theta='independent'."
        )

    # Map user-facing parameter names to internal names.
    mapped = dict(
        z_score_x=z_score_theta,
        z_score_y=z_score_x,
        hidden_features=hidden_features,
        num_layers=num_layers,
        embedding_net=check_net_device(embedding_net, "cpu", embedding_net_warn_msg),
        time_embedding_dim=t_embedding_dim,
        time_emb_type=time_emb_type,
        net=model,
        gaussian_baseline=gaussian_baseline,
        compose_standardization=compose_standardization,
    )

    # Validate against known fields — warns on unknown kwargs (typos)
    # while still forwarding them to the underlying builder.
    config = FlowEstimatorConfig.from_kwargs(**mapped, **kwargs)
    builder_kwargs = config.to_dict()

    def build_fn(batch_theta, batch_x):
        return build_vector_field_estimator(
            batch_x=batch_theta,
            batch_y=batch_x,
            estimator_type="flow",
            **builder_kwargs,
        )

    return build_fn


def marginal_nn(
    model: ZukoFlowType,
    z_score_x: Optional[
        Literal["independent", "structured", "transform_to_unconstrained", "none"]
    ] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    num_components: int = 10,
    **kwargs: Any,
) -> Callable:
    r"""
    Returns a function that builds a density estimator for learning the marginal.

    Args:
        model: The type of density estimator that will be created.
        z_score_x: Whether to z-score samples $x$ before passing them into
            the network.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms when a flow is used.
        num_bins: Number of bins used for the splines in `nsf`.
        num_components: Number of mixture components for a mixture of Gaussians.
        **kwargs: Additional estimator arguments.  Valid keys are defined by
            ``MarginalFlowConfig``; unknown keys trigger a warning and are forwarded to
            the builder.
    """

    # Map user-facing parameter names to internal names (no renaming needed here).
    mapped = dict(
        z_score_x=z_score_x,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_bins=num_bins,
        num_components=num_components,
    )

    # Validate against known fields — warns on unknown kwargs (typos)
    # while still forwarding them to the underlying builder.
    config = MarginalFlowConfig.from_kwargs(**mapped, **kwargs)
    builder_kwargs = config.to_dict()

    def build_fn(batch_x: Tensor) -> Any:
        return build_zuko_unconditional_flow(
            which_nf=model.value.upper(), batch_x=batch_x, **builder_kwargs
        )

    return build_fn
