# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from enum import Enum
from typing import Any, Callable, Literal, Optional, Union

from torch import Tensor, nn

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
    build_flow_matching_estimator,
    build_score_matching_estimator,
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
}


# TODO: currently only used for marginal_nn, adapt to use for all
class ZukoFlowType(Enum):
    """Enumeration of Zuko flow types."""

    BPF = "bpf"
    MAF = "maf"
    NAF = "naf"
    NCSF = "ncsf"
    NSF = "nsf"
    SOSPF = "sospf"
    UNAF = "unaf"


embedding_net_warn_msg = """The passed embedding net will be moved to cpu for
                        constructing the net building function."""


def classifier_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
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
            the network, same options as z_score_theta.
        hidden_features: Number of hidden features.
        embedding_net_theta:  Optional embedding network for parameters $\theta$.
        embedding_net_x:  Optional embedding network for simulation outputs $x$. This
            embedding net allows to learn features from potentially high-dimensional
            simulation outputs.
        kwargs: additional custom arguments passed to downstream build functions.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "hidden_features",
                "embedding_net_x",
                "embedding_net_y",
            ),
            (
                z_score_theta,
                z_score_x,
                hidden_features,
                check_net_device(embedding_net_theta, "cpu", embedding_net_warn_msg),
                check_net_device(embedding_net_x, "cpu", embedding_net_warn_msg),
            ),
            strict=False,
        ),
        **kwargs,
    )

    def build_fn(batch_theta, batch_x):
        if model == "linear":
            return build_linear_classifier(
                batch_x=batch_theta, batch_y=batch_x, **kwargs
            )
        if model == "mlp":
            return build_mlp_classifier(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        if model == "resnet":
            return build_resnet_classifier(
                batch_x=batch_theta, batch_y=batch_x, **kwargs
            )
        else:
            raise NotImplementedError

    return build_fn


def likelihood_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
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
            the network, same options as z_score_theta.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms when a flow is used. Only relevant if
            density estimator is a normalizing flow (i.e. currently either a `maf` or a
            `nsf`). Ignored if density estimator is a `mdn` or `made`.
        num_bins: Number of bins used for the splines in `nsf`. Ignored if density
            estimator not `nsf`.
        embedding_net: Optional embedding network for parameters $\theta$.
        num_components: Number of mixture components for a mixture of Gaussians.
            Ignored if density estimator is not an mdn.
        kwargs: additional custom arguments passed to downstream build functions.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "hidden_features",
                "num_transforms",
                "num_bins",
                "embedding_net",
                "num_components",
            ),
            (
                z_score_x,
                z_score_theta,
                hidden_features,
                num_transforms,
                num_bins,
                check_net_device(embedding_net, "cpu", embedding_net_warn_msg),
                num_components,
            ),
            strict=False,
        ),
        **kwargs,
    )

    def build_fn(batch_theta, batch_x):
        if model not in model_builders:
            raise NotImplementedError(f"Model {model} in not implemented")

        return model_builders[model](batch_x=batch_x, batch_y=batch_theta, **kwargs)

    return build_fn


def posterior_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
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
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network, same options as z_score_theta.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms when a flow is used. Only relevant if
            density estimator is a normalizing flow (i.e. currently either a `maf` or a
            `nsf`). Ignored if density estimator is a `mdn` or `made`.
        num_bins: Number of bins used for the splines in `nsf`. Ignored if density
            estimator not `nsf`.
        embedding_net: Optional embedding network for simulation outputs $x$. This
            embedding net allows to learn features from potentially high-dimensional
            simulation outputs.
        num_components: Number of mixture components for a mixture of Gaussians.
            Ignored if density estimator is not an mdn.
        kwargs: additional custom arguments passed to downstream build functions.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "hidden_features",
                "num_transforms",
                "num_bins",
                "embedding_net",
                "num_components",
            ),
            (
                z_score_theta,
                z_score_x,
                hidden_features,
                num_transforms,
                num_bins,
                check_net_device(embedding_net, "cpu", embedding_net_warn_msg),
                num_components,
            ),
            strict=False,
        ),
        **kwargs,
    )

    def build_fn_snpe_a(batch_theta, batch_x, num_components):
        """Build function for SNPE-A

        Extract the number of components from the kwargs, such that they are exposed as
        a kwargs, offering the possibility to later override this kwarg with
        `functools.partial`. This is necessary in order to make sure that the MDN in
        SNPE-A only has one component when running the Algorithm 1 part.
        """
        return build_mdn(
            batch_x=batch_theta,
            batch_y=batch_x,
            num_components=num_components,
            **kwargs,
        )

    def build_fn(batch_theta, batch_x):
        if model not in model_builders:
            raise NotImplementedError(f"Model {model} in not implemented")

        # The naming might be a bit confusing.
        # batch_x are the latent variables, batch_y the conditioned variables.
        # batch_theta are the parameters and batch_x the observable variables.
        return model_builders[model](batch_x=batch_theta, batch_y=batch_x, **kwargs)

    if model == "mdn_snpe_a":
        if num_components != 10:
            raise ValueError(
                "You set `num_components`. For SNPE-A, this has to be done at "
                "instantiation of the inference object, i.e. "
                "`inference = SNPE_A(..., num_components=20)`"
            )
        kwargs.pop("num_components")

    return build_fn_snpe_a if model == "mdn_snpe_a" else build_fn


def posterior_score_nn(
    model: Union[
        Literal["mlp", "ada_mlp", "transformer", "transformer_cross_attn"],
        VectorFieldNet,
    ] = "mlp",
    sde_type: str = "ve",
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 100,
    num_layers: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    time_emb_type: Literal["sinusoidal", "fourier"] = "sinusoidal",
    t_embedding_dim: int = 32,
    score_net_type: Optional[
        Union[
            Literal["mlp", "ada_mlp", "transformer", "transformer_cross_attn"],
            VectorFieldNet,
        ]
    ] = None,
    **kwargs: Any,
) -> Callable:
    """Build util function that builds a ScoreEstimator object for score-based
    posteriors.

    Args:
        sde_type: SDE type used, which defines the mean and std functions. One of:
            - 'vp': Variance preserving.
            - 'subvp': Sub-variance preserving.
            - 've': Variance exploding.
            Defaults to 'vp'.
        model: Type of regression network. One of:
            - 'mlp': Fully connected feed-forward network.
            - 'ada_mlp': Fully connected feed-forward with adaptive
               layer normalization for conditioning.
            - 'transformer': Transformer network.
            - 'transformer_cross_attention': Transformer with cross-attention.
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
        hidden_features: Number of hidden units per layer. Defaults to 50.
        embedding_net: Embedding network for x (conditioning variable). Defaults to
            nn.Identity().
        time_emb_type: Type of time embedding. Defaults to 'sinusoidal'.
        t_embedding_dim: Embedding dimension of diffusion time. Defaults to 32.

    Returns:
        Constructor function for NPSE.
    """

    if score_net_type is not None:
        model = score_net_type
        warnings.warn(
            "score_net_type is deprecated and will be removed in a future release. "
            "Please use model instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "hidden_features",
                "num_layers",
                "embedding_net",
                "time_embedding_dim",
                "time_emb_type",
                "net",
            ),
            (
                z_score_x,
                z_score_theta,
                hidden_features,
                num_layers,
                check_net_device(embedding_net, "cpu", embedding_net_warn_msg),
                t_embedding_dim,
                time_emb_type,
                model,
            ),
            strict=False,
        ),
        **kwargs,
    )

    def build_fn(batch_theta, batch_x):
        # Build the score matching estimator
        return build_score_matching_estimator(
            batch_x=batch_theta,
            batch_y=batch_x,
            sde_type=sde_type,
            **kwargs,
        )

    return build_fn


# TODO: remove this function on next release
def flowmatching_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 64,
    num_layers: int = 5,
    num_blocks: int = 5,
    num_frequencies: int = 3,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs: Any,
) -> Callable:
    r"""Returns a function that builds a neural net that can act as
    a vector field estimator for Flow Matching. This function will usually
    be used for Flow Matching. The returned function is to be passed to the

    Args:
        model: the type of regression network to learn the vector field. One of ['mlp',
            'resnet'].
        z_score_theta: Whether to z-score parameters $\theta$ before passing them into
            the network, can take one of the following:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network, same options as z_score_theta.
        hidden_features: Number of hidden features.
        num_layers: Number of transforms when a flow is used. Only relevant if
            density estimator is a normalizing flow (i.e. currently either a `maf` or a
            `nsf`). Ignored if density estimator is a `mdn` or `made`.
        num_blocks: Number of blocks if a ResNet is used.
        num_frequencies: Number of frequencies for the time embedding.
        embedding_net: Optional embedding network for the condition.
        kwargs: additional custom arguments passed to downstream build functions.
    """
    # NOTE: I keep this function because it was used in the documentation notebook
    # examples.
    warnings.warn(
        "flowmatching_nn is deprecated and will be removed in a future release. "
        "Please use posterior_flow_nn or the new vector field estimator builders "
        "instead.",
        FutureWarning,
        stacklevel=2,
    )
    implemented_models = ["mlp", "resnet"]

    if model not in implemented_models:
        raise NotImplementedError(f"Model {model} in not implemented for FMPE")

    model_str = model + "_flowmatcher"

    def build_fn(batch_theta, batch_x):
        return model_builders[model_str](
            batch_x=batch_theta,
            batch_y=batch_x,
            z_score_x=z_score_theta,
            z_score_y=z_score_x,
            hidden_features=hidden_features,
            num_layers=num_layers,
            num_blocks=num_blocks,
            num_freqs=num_frequencies,
            embedding_net=check_net_device(
                embedding_net, "cpu", embedding_net_warn_msg
            ),
            **kwargs,
        )

    return build_fn


def posterior_flow_nn(
    model: Union[
        Literal["mlp", "ada_mlp", "transformer", "transformer_cross_attn"],
        VectorFieldNet,
    ] = "mlp",
    z_score_theta: Optional[str] = None,
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 100,
    num_layers: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    time_emb_type: Literal["sinusoidal", "fourier"] = "sinusoidal",
    t_embedding_dim: int = 32,
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
            - 'transformer_cross_attention': Transformer with cross-attention.
            -  nn.Module: Custom network
            Defaults to 'mlp'.
        z_score_theta: This is not supported for FMPE and will raise an error.
        z_score_x: Whether to z-score xs passing into the network, same options as
            z_score_theta.
        hidden_features: Number of hidden units per layer. Defaults to 50.
        num_layers: Number of hidden layers. Defaults to 5.
        embedding_net: Embedding network for x (conditioning variable). Defaults to
            nn.Identity().
        time_emb_type: Type of time embedding. Defaults to 'sinusoidal'.
        t_embedding_dim: Embedding dimension of diffusion time. Defaults to 32.

    Returns:
        Constructor function for FMPE.
    """

    if z_score_theta is not None:
        raise ValueError(
            "z_score_theta is not supported for FMPE. For simulator "
            "parameters with highly different scales, we recommend "
            "z-scoring the inputs manually beforehand."
        )

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "embedding_net",
                "hidden_features",
                "time_embedding_dim",
                "time_emb_type",
                "net",
                "num_layers",
            ),
            (
                z_score_x,
                z_score_theta,
                embedding_net,
                hidden_features,
                t_embedding_dim,
                time_emb_type,
                model,
                num_layers,
            ),
            strict=False,
        ),
        **kwargs,
    )

    def build_fn(batch_theta, batch_x):
        # Build the flow matching estimator
        return build_flow_matching_estimator(
            batch_x=batch_theta,
            batch_y=batch_x,
            **kwargs,
        )

    return build_fn


def marginal_nn(
    model: ZukoFlowType,
    z_score_x: Optional[str] = "independent",
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
        kwargs: additional custom arguments passed to downstream build functions.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "hidden_features",
                "num_transforms",
                "num_bins",
                "num_components",
            ),
            (
                z_score_x,
                hidden_features,
                num_transforms,
                num_bins,
                num_components,
            ),
            strict=False,
        ),
        **kwargs,
    )

    def build_fn(batch_x: Tensor) -> Any:
        return build_zuko_unconditional_flow(
            which_nf=model.value.upper(), batch_x=batch_x, **kwargs
        )

    return build_fn
