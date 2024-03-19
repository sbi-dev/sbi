# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Callable, Optional
from warnings import warn

from torch import nn

from sbi.neural_nets import classifier_nn as classifier_nn_new
from sbi.neural_nets import likelihood_nn as likelihood_nn_new
from sbi.neural_nets import posterior_nn as posterior_nn_new


def classifier_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 50,
    embedding_net_theta: nn.Module = nn.Identity(),
    embedding_net_x: nn.Module = nn.Identity(),
    **kwargs,
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

    warn(
        "This method is deprecated and will be removed in the next release."
        "Please use classifier_nn from sbi.neural_nets in the future.",
        DeprecationWarning,
        stacklevel=2,
    )

    return classifier_nn_new(
        model,
        z_score_theta,
        z_score_x,
        hidden_features,
        embedding_net_theta,
        embedding_net_x,
        **kwargs,
    )


def likelihood_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    num_components: int = 10,
    **kwargs,
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

    warn(
        "This method is deprecated and will be removed in the next release. "
        "Please use likelihood_nn from sbi.neural_nets in the future.",
        DeprecationWarning,
        stacklevel=2,
    )

    return likelihood_nn_new(
        model,
        z_score_theta,
        z_score_x,
        hidden_features,
        num_transforms,
        num_bins,
        embedding_net,
        num_components,
        **kwargs,
    )


def posterior_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    num_components: int = 10,
    **kwargs,
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

    warn(
        "This method is deprecated and will be removed in the next release."
        "Please use posterior_nn from sbi.neural_nets in the future.",
        DeprecationWarning,
        stacklevel=2,
    )

    return posterior_nn_new(
        model,
        z_score_theta,
        z_score_x,
        hidden_features,
        num_transforms,
        num_bins,
        embedding_net,
        num_components,
        **kwargs,
    )
