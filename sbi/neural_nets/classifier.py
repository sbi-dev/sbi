# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import torch
from pyknos.nflows.nn import nets
from torch import Tensor, nn, relu

from sbi.utils.sbiutils import standardizing_net


def build_input_layer(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
) -> nn.Module:
    """Builds input layer for classifiers that optionally z-scores

    In SNRE, the classifier will receive batches of thetas and xs.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring
        z_score_x: Whether to z-score xs passing into the network
        z_score_y: Whether to z-score ys passing into the network
        hidden_features: Number of hidden features

    Returns:
        Input layer that optionally z-scores
    """
    if z_score_x:
        embedding_net_x = standardizing_net(batch_x)
    else:
        embedding_net_x = nn.Identity()

    if z_score_y:
        embedding_net_y = standardizing_net(batch_y)
    else:
        embedding_net_y = nn.Identity()

    class StandardizeInputs(nn.Module):
        def __init__(self, embedding_net_x, embedding_net_y, dim_x, dim_y):
            super().__init__()
            self.embedding_net_x = embedding_net_x
            self.embedding_net_y = embedding_net_y
            self.dim_x = dim_x
            self.dim_y = dim_y

        def forward(self, t):
            out = torch.cat(
                [
                    self.embedding_net_x(t[:, : self.dim_x]),
                    self.embedding_net_y(t[:, self.dim_x : self.dim_x + self.dim_y]),
                ],
                dim=1,
            )
            return out

    input_layer = StandardizeInputs(
        embedding_net_x, embedding_net_y, dim_x=batch_x.shape[1], dim_y=batch_y.shape[1]
    )

    return input_layer


def build_linear_classifier(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
) -> nn.Module:
    """Builds linear classifier

    In SNRE, the classifier will receive batches of thetas and xs.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring
        z_score_x: Whether to z-score xs passing into the network
        z_score_y: Whether to z-score ys passing into the network

    Returns:
        Neural network
    """
    x_numel = batch_x[0].numel()
    y_numel = batch_y[0].numel()

    neural_net = nn.Linear(x_numel + y_numel, 1)

    input_layer = build_input_layer(batch_x, batch_y, z_score_x, z_score_y)

    neural_net = nn.Sequential(input_layer, neural_net)

    return neural_net


def build_mlp_classifier(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
    hidden_features: int = 50,
) -> nn.Module:
    """Builds MLP classifier

    In SNRE, the classifier will receive batches of thetas and xs.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring
        z_score_x: Whether to z-score xs passing into the network
        z_score_y: Whether to z-score ys passing into the network
        hidden_features: Number of hidden features

    Returns:
        Neural network
    """
    x_numel = batch_x[0].numel()
    y_numel = batch_y[0].numel()

    neural_net = nn.Sequential(
        nn.Linear(x_numel + y_numel, hidden_features),
        nn.BatchNorm1d(hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, hidden_features),
        nn.BatchNorm1d(hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, 1),
    )

    input_layer = build_input_layer(batch_x, batch_y, z_score_x, z_score_y)

    neural_net = nn.Sequential(input_layer, neural_net)

    return neural_net


def build_resnet_classifier(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
    hidden_features: int = 50,
) -> nn.Module:
    """Builds ResNet classifier

    In SNRE, the classifier will receive batches of thetas and xs.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring
        z_score_x: Whether to z-score xs passing into the network
        z_score_y: Whether to z-score ys passing into the network
        hidden_features: Number of hidden features

    Returns:
        Neural network
    """
    x_numel = batch_x[0].numel()
    y_numel = batch_y[0].numel()

    neural_net = nets.ResidualNet(
        in_features=x_numel + y_numel,
        out_features=1,
        hidden_features=hidden_features,
        context_features=None,
        num_blocks=2,
        activation=relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    )

    input_layer = build_input_layer(batch_x, batch_y, z_score_x, z_score_y)

    neural_net = nn.Sequential(input_layer, neural_net)

    return neural_net
