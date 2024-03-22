# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Optional

from pyknos.nflows.nn import nets
from torch import Tensor, nn, relu

from sbi.neural_nets.ratio_estimators import TensorRatioEstimator
from sbi.utils.sbiutils import standardizing_net, z_score_parser
from sbi.utils.user_input_checks import check_data_device, check_embedding_net_device


def build_z_scored_embedding_nets(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    embedding_net_x: nn.Module = nn.Identity(),
    embedding_net_y: nn.Module = nn.Identity(),
) -> tuple[nn.Module, nn.Module]:
    """Builds input layer for critics that optionally z-score.

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
        Input layer that optionally z-scores.
    """
    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        embedding_net_x = nn.Sequential(
            standardizing_net(batch_x, structured_x), embedding_net_x
        )

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net_y = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net_y
        )

    return embedding_net_x, embedding_net_y


def build_linear_critic(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    embedding_net_x: nn.Module = nn.Identity(),
    embedding_net_y: nn.Module = nn.Identity(),
    **kwargs,
) -> TensorRatioEstimator:
    """Builds linear critic.

    In SNRE, the critic will receive batches of thetas and xs.

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
            relevant for linear critics and are therefore ignored.

    Returns:
        Neural network.
    """
    check_data_device(batch_x, batch_y)
    check_embedding_net_device(embedding_net=embedding_net_x, datum=batch_y)
    check_embedding_net_device(embedding_net=embedding_net_y, datum=batch_y)

    # Infer the output dimensionalities of the embedding_net by making a forward pass.
    x_numel = embedding_net_x(batch_x[:1]).size(-1)
    y_numel = embedding_net_y(batch_y[:1]).size(-1)

    neural_net = nn.Linear(x_numel + y_numel, 1)

    embedding_net_x, embedding_net_y = build_z_scored_embedding_nets(
        batch_x, batch_y, z_score_x, z_score_y, embedding_net_x, embedding_net_y
    )

    return TensorRatioEstimator(
        net=neural_net,
        embedding_net_theta=embedding_net_x,
        embedding_net_x=embedding_net_y,
    )


def build_mlp_critic(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    embedding_net_x: nn.Module = nn.Identity(),
    embedding_net_y: nn.Module = nn.Identity(),
) -> TensorRatioEstimator:
    """Builds MLP critic.

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
    check_embedding_net_device(embedding_net=embedding_net_x, datum=batch_y)
    check_embedding_net_device(embedding_net=embedding_net_y, datum=batch_y)

    # Infer the output dimensionalities of the embedding_net by making a forward pass.
    x_numel = embedding_net_x(batch_x[:1]).size(-1)
    y_numel = embedding_net_y(batch_y[:1]).size(-1)

    neural_net = nn.Sequential(
        nn.Linear(x_numel + y_numel, hidden_features),
        nn.BatchNorm1d(hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, hidden_features),
        nn.BatchNorm1d(hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, 1),
    )

    embedding_net_x, embedding_net_y = build_z_scored_embedding_nets(
        batch_x, batch_y, z_score_x, z_score_y, embedding_net_x, embedding_net_y
    )

    return TensorRatioEstimator(
        net=neural_net,
        embedding_net_theta=embedding_net_x,
        embedding_net_x=embedding_net_y,
    )


def build_resnet_critic(
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
) -> TensorRatioEstimator:
    """Builds ResNet critic.

    In SNRE, the critic will receive batches of thetas and xs.

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
    check_embedding_net_device(embedding_net=embedding_net_x, datum=batch_y)
    check_embedding_net_device(embedding_net=embedding_net_y, datum=batch_y)

    # Infer the output dimensionalities of the embedding_net by making a forward pass.
    x_numel = embedding_net_x(batch_x[:1]).size(-1)
    y_numel = embedding_net_y(batch_y[:1]).size(-1)

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

    embedding_net_x, embedding_net_y = build_z_scored_embedding_nets(
        batch_x, batch_y, z_score_x, z_score_y, embedding_net_x, embedding_net_y
    )

    return TensorRatioEstimator(
        net=neural_net,
        embedding_net_theta=embedding_net_x,
        embedding_net_x=embedding_net_y,
    )
