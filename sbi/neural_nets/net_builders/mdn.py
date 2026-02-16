# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

import torch
from torch import Tensor, nn

from sbi.neural_nets.estimators.mixture_density_estimator import (
    MixtureDensityEstimator,
    MultivariateGaussianMDN,
)
from sbi.utils.nn_utils import get_numel
from sbi.utils.sbiutils import (
    standardizing_net,
    z_score_parser,
    z_standardization,
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
) -> MixtureDensityEstimator:
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
        MixtureDensityEstimator for conditional density estimation.
    """
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x, embedding_net=None)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    # Handle z-scoring for x (input)
    transform_input = None
    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        x_mean, x_std = z_standardization(batch_x, structured_x)
        # Store as [shift, scale] tensor for the estimator
        transform_input = torch.stack([x_mean, x_std], dim=0)

    # Handle z-scoring for y (condition) via embedding net
    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    # Create the MDN network
    mdn_net = MultivariateGaussianMDN(
        features=x_numel,
        context_features=y_numel,
        hidden_features=hidden_features,
        num_components=num_components,
        custom_initialization=True,
    )

    # Wrap in MixtureDensityEstimator
    estimator = MixtureDensityEstimator(
        net=mdn_net,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
        embedding_net=embedding_net,
        transform_input=transform_input,
    )

    return estimator
