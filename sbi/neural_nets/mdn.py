# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from pyknos.mdn.mdn import MultivariateGaussianMDN
from torch import Tensor, nn

from sbi.utils.sbiutils import standardizing_net


def build_mdn(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
    hidden_features: int = 50,
    num_components: int = 10,
) -> nn.Module:
    """Builds MDN p(x|y)

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring
        z_score_x: Whether to z-score xs passing into the network
        z_score_y: Whether to z-score ys passing into the network
        hidden_features: Number of hidden features
        num_components: Number of components

    Returns:
        Neural network
    """
    x_numel = batch_x[0].numel()
    y_numel = batch_y[0].numel()

    neural_net = MultivariateGaussianMDN(
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

    if z_score_x:
        embedding_net_x = standardizing_net(batch_x)
    else:
        embedding_net_x = nn.Identity()

    if z_score_y:
        embedding_net_y = standardizing_net(batch_y)
    else:
        embedding_net_y = nn.Identity()

    neural_net = nn.Sequential(embedding_net_y, neural_net, embedding_net_x)

    return neural_net
