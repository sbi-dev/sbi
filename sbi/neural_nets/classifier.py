# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Optional

from pyknos.nflows.nn import nets
from torch import Tensor, nn, relu

from sbi.neural_nets.ratio_estimators import RatioEstimator
from sbi.utils.nn_utils import get_numel
from sbi.utils.sbiutils import standardizing_net, z_score_parser
from sbi.utils.user_input_checks import check_data_device


def build_z_scored_embedding_net(
    batch: Tensor,
    z_score: Optional[str] = "independent",
    embedding_net: nn.Module = nn.Identity(),
) -> nn.Module:
    """Builds input layer for classifiers that optionally z-scores.

    Args:
        batch: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        z_score: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        embedding_net: Optional embedding network for x.

    Returns:
        Input layer that optionally z-scores.
    """
    z_score_bool, structured = z_score_parser(z_score)
    if z_score_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch, structured), embedding_net
        )
    return embedding_net


def build_linear_classifier(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    embedding_net_x: nn.Module = nn.Identity(),
    embedding_net_y: nn.Module = nn.Identity(),
    **kwargs,
) -> RatioEstimator:
    """Builds linear classifier.

    In SNRE, the classifier will receive batches of thetas and xs.

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
        embedding_net_x: Optional embedding network for x.
        embedding_net_y: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for linear classifiers and are therefore ignored.

    Returns:
        Neural network.
    """
    # Infer the output dimensionalities of the embedding_net by making a forward
    # pass.
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x, embedding_net=embedding_net_x)
    y_numel = get_numel(batch_y, embedding_net=embedding_net_y)

    neural_net = nn.Linear(x_numel + y_numel, 1)

    embedding_net_x = build_z_scored_embedding_net(batch_x, z_score_x, embedding_net_x)
    embedding_net_y = build_z_scored_embedding_net(batch_y, z_score_y, embedding_net_y)

    return RatioEstimator(
        net=neural_net,
        theta_shape=batch_x[0].shape,
        x_shape=batch_y[0].shape,
        embedding_net_theta=embedding_net_x,
        embedding_net_x=embedding_net_y,
    )


def build_mlp_classifier(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    embedding_net_x: nn.Module = nn.Identity(),
    embedding_net_y: nn.Module = nn.Identity(),
) -> RatioEstimator:
    """Builds MLP classifier.

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
        embedding_net_x: Optional embedding network for x.
        embedding_net_y: Optional embedding network for y.

    Returns:
        Neural network.
    """
    # Infer the output dimensionalities of the embedding_net by making a forward pass.
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x, embedding_net=embedding_net_x)
    y_numel = get_numel(batch_y, embedding_net=embedding_net_y)

    neural_net = nn.Sequential(
        nn.Linear(x_numel + y_numel, hidden_features),
        nn.BatchNorm1d(hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, hidden_features),
        nn.BatchNorm1d(hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, 1),
    )

    embedding_net_x = build_z_scored_embedding_net(batch_x, z_score_x, embedding_net_x)
    embedding_net_y = build_z_scored_embedding_net(batch_y, z_score_y, embedding_net_y)

    return RatioEstimator(
        net=neural_net,
        theta_shape=batch_x[0].shape,
        x_shape=batch_y[0].shape,
        embedding_net_theta=embedding_net_x,
        embedding_net_x=embedding_net_y,
    )


def build_resnet_classifier(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    embedding_net_x: nn.Module = nn.Identity(),
    embedding_net_y: nn.Module = nn.Identity(),
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
) -> RatioEstimator:
    """Builds ResNet classifier.

    In SNRE, the classifier will receive batches of thetas and xs.

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
        embedding_net_x: Optional embedding network for x.
        embedding_net_y: Optional embedding network for y.

    Returns:
        Neural network.
    """
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x, embedding_net=embedding_net_x)
    y_numel = get_numel(batch_y, embedding_net=embedding_net_y)

    neural_net = nets.ResidualNet(
        in_features=x_numel + y_numel,
        out_features=1,
        hidden_features=hidden_features,
        context_features=None,
        num_blocks=num_blocks,
        activation=relu,
        dropout_probability=dropout_probability,
        use_batch_norm=use_batch_norm,
    )

    embedding_net_x = build_z_scored_embedding_net(batch_x, z_score_x, embedding_net_x)
    embedding_net_y = build_z_scored_embedding_net(batch_y, z_score_y, embedding_net_y)

    return RatioEstimator(
        net=neural_net,
        theta_shape=batch_x[0].shape,
        x_shape=batch_y[0].shape,
        embedding_net_theta=embedding_net_x,
        embedding_net_x=embedding_net_y,
    )
