# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Callable, Optional
from warnings import warn

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
    build_zuko_maf,
    build_zuko_flow_matching
)
from sbi.neural_nets.mdn import build_mdn
from sbi.neural_nets.mnle import build_mnle


def classifier_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 50,
    embedding_net_theta: nn.Module = nn.Identity(),
    embedding_net_x: nn.Module = nn.Identity(),
    **kwargs: Any,
) -> Callable:
    r"""This method is deprecated and will be removed in a future release.
    Please use `from sbi.neural_nets import classifier_nn` in the future.
    """

    warn(
        "This method is deprecated and will be removed in a future release."
        "Please use `from sbi.neural_nets import classifier_nn` in the future.",
        DeprecationWarning,
        stacklevel=2,
    )

    return classifier_nn_moved_to_neural_nets(
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
    **kwargs: Any,
) -> Callable:
    r"""This method is deprecated and will be removed in a future release.
    Please use `from sbi.neural_nets import likelihood_nn` in the future.
    """

    warn(
        "This method is deprecated and will be removed in a future release. "
        "Please use `from sbi.neural_nets import likelihood_nn` in the future.",
        DeprecationWarning,
        stacklevel=2,
    )

    return likelihood_nn_moved_to_neural_nets(
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
    **kwargs: Any,
) -> Callable:
    r"""This method is deprecated and will be removed in a future release.
    Please use `from sbi.neural_nets import posterior_nn` in the future.
    """

    warn(
        "This method is deprecated and will be removed in a future release."
        "Please use `from sbi.neural_nets import posterior_nn` in the future.",
        DeprecationWarning,
        stacklevel=2,
    )

    return posterior_nn_moved_to_neural_nets(
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
        if model == "mdn":
            return build_mdn(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "made":
            return build_made(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "maf":
            return build_maf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "maf_rqs":
            return build_maf_rqs(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "nsf":
            return build_nsf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "zuko_maf":
            return build_zuko_maf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "zuko_fm":
            return build_zuko_flow_matching(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        else:
            raise NotImplementedError

    if model == "mdn_snpe_a":
        if num_components != 10:
            raise ValueError(
                "You set `num_components`. For SNPE-A, this has to be done at "
                "instantiation of the inference object, i.e. "
                "`inference = SNPE_A(..., num_components=20)`"
            )
        kwargs.pop("num_components")

    return build_fn_snpe_a if model == "mdn_snpe_a" else build_fn
