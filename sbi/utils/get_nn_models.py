# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Optional
from warnings import warn

from torch import nn

from sbi.neural_nets import classifier_nn as classifier_nn_moved_to_neural_nets
from sbi.neural_nets import likelihood_nn as likelihood_nn_moved_to_neural_nets
from sbi.neural_nets import posterior_nn as posterior_nn_moved_to_neural_nets


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
