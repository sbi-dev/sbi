# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# build function for flow matching neural networks
# like in classifier.py, we need to build a network that can z-score the inputs

from typing import Optional, Sequence, Union

import torch
import zuko
from torch import Tensor, nn
from zuko.nn import MLP

from sbi.neural_nets.density_estimators.zuko_flow import FlowMatchingEstimator
from sbi.utils.nn_utils import get_numel
from sbi.utils.sbiutils import (
    standardizing_net,
    z_score_parser,
    z_standardization,
)
from sbi.utils.user_input_checks import check_data_device


def build_mlp_flow_matcher(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: Union[Sequence[int], int] = 64,
    num_transforms: int = 5,
    num_freqs: int = 3,
    embedding_net: nn.Module = nn.Identity(),
    embedding_net_x: nn.Module = nn.Identity(),
    **kwargs,
) -> FlowMatchingEstimator:
    """Builds a flow matching neural network.

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
        num_transforms: Number of transforms in the vector field regressor.
        num_freqs: Number of frequencies in the time embeddings.
        embedding_net_x: Embedding network for the input.
        embedding_net_y: Embedding network for the condition.
        kwargs: Additional keyword arguments passed to the FlowMatchingEstimator.
    """
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    # create a list of layers for the regression network; the vector field
    # regressor is a MLP consisting of num_transforms of layers with
    # hidden_features neurons each
    if isinstance(hidden_features, int):
        hidden_features = [hidden_features] * num_transforms

    vector_field_regression_net = MLP(
        in_features=x_numel + y_numel + 2 * num_freqs,
        out_features=x_numel,
        hidden_features=hidden_features,
        activation=nn.ELU,
    )

    # input data is only z-scored, not embedded.
    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        t_mean, t_std = z_standardization(batch_x, structured_x)
        z_score_transform = torch.distributions.AffineTransform(
            -t_mean / t_std, 1 / t_std
        )
    else:
        z_score_transform = zuko.transforms.IdentityTransform()

    # pre-pend the z-scoring layer to the embedding net.
    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    # create the flow matching estimator, will take care of time embeddings.
    flow_matching_estimator = FlowMatchingEstimator(
        net=vector_field_regression_net,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
        zscore_transform_input=z_score_transform,
        embedding_net_input=embedding_net,
        embedding_net_condition=embedding_net_x,
        num_freqs=num_freqs,
        **kwargs,  # e.g., noise_scale
    )

    return flow_matching_estimator
