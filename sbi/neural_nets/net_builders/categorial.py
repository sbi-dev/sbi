# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

from torch import Tensor, nn, unique

from sbi.neural_nets.estimators import CategoricalMassEstimator, CategoricalNet
from sbi.utils.nn_utils import get_numel
from sbi.utils.sbiutils import (
    standardizing_net,
    z_score_parser,
)
from sbi.utils.user_input_checks import check_data_device


def build_categoricalmassestimator(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "none",
    z_score_y: Optional[str] = "independent",
    num_hidden: int = 20,
    num_layers: int = 2,
    embedding_net: nn.Module = nn.Identity(),
):
    """Returns a density estimator for a categorical random variable.

    Args:
        batch_x: A batch of input data.
        batch_y: A batch of condition data.
        z_score_x: Whether to z-score the input data.
        z_score_y: Whether to z-score the condition data.
        num_hidden: Number of hidden units per layer.
        num_layers: Number of hidden layers.
        embedding_net: Embedding net for y.
    """

    if z_score_x != "none":
        raise ValueError("Categorical input should not be z-scored.")

    check_data_device(batch_x, batch_y)
    if batch_x.shape[1] > 1:
        raise NotImplementedError("CategoricalMassEstimator only supports 1D input.")
    num_categories = unique(batch_x).numel()
    dim_condition = get_numel(batch_y, embedding_net=embedding_net)

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    categorical_net = CategoricalNet(
        num_input=dim_condition,
        num_categories=num_categories,
        num_hidden=num_hidden,
        num_layers=num_layers,
        embedding_net=embedding_net,
    )

    return CategoricalMassEstimator(
        categorical_net, input_shape=batch_x[0].shape, condition_shape=batch_y[0].shape
    )
