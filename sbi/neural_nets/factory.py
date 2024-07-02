# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Any, Callable, Optional

from torch import nn

from sbi.neural_nets.classifier import (
    build_linear_classifier,
    build_mlp_classifier,
    build_resnet_classifier,
)
from sbi.neural_nets.flow import (
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
from sbi.neural_nets.mdn import build_mdn
from sbi.neural_nets.mnle import build_mnle
from sbi.utils.nn_utils import check_net_device

model_builders = {
    "mdn": build_mdn,
    "made": build_made,
    "maf": build_maf,
    "maf_rqs": build_maf_rqs,
    "nsf": build_nsf,
    "mnle": build_mnle,
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
