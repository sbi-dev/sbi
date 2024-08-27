# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

from pyknos.mdn.mdn import MultivariateGaussianMDN
from pyknos.nflows import flows, transforms
from torch import Tensor, nn

from sbi.neural_nets.estimators import NFlowsFlow
from sbi.utils.nn_utils import get_numel
from sbi.utils.sbiutils import (
    standardizing_net,
    standardizing_transform,
    z_score_parser,
)
from sbi.utils.user_input_checks import check_data_device


def build_mdn(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    num_components: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> NFlowsFlow:
    """Builds MDN p(x|y).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Number of hidden features.
        num_components: Number of components.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for MDNs and are therefore ignored.

    Returns:
        Neural network.
    """
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x, embedding_net=None)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    transform = transforms.IdentityTransform()

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        transform_zx = standardizing_transform(batch_x, structured_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    distribution = MultivariateGaussianMDN(
        features=x_numel,
        context_features=y_numel,
        hidden_features=hidden_features,
        hidden_net=nn.Sequential(
            nn.Linear(y_numel, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ),
        num_components=num_components,
        custom_initialization=True,
    )

    neural_net = flows.Flow(transform, distribution, embedding_net)
    flow = NFlowsFlow(
        neural_net, input_shape=batch_x[0].shape, condition_shape=batch_y[0].shape
    )

    return flow
