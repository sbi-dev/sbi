# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Optional

from torch import Tensor, nn, unique

from sbi.neural_nets.density_estimators import CategoricalMassEstimator, CategoricalNet


def build_categoricalmassestimator(
    input: Tensor,
    condition: Tensor,
    num_hidden: int = 20,
    num_layers: int = 2,
    embedding_net: Optional[nn.Module] = None,
):
    """Returns a density estimator for a categorical random variable."""
    # Infer input and output dims.
    if embedding_net is None:
        dim_input = condition[0].numel()
    else:
        dim_input = embedding_net(condition[:1]).numel()
    num_categories = unique(input).numel()

    categorical_net = CategoricalNet(
        num_input=dim_input,
        num_categories=num_categories,
        num_hidden=num_hidden,
        num_layers=num_layers,
        embedding_net=embedding_net,
    )

    return CategoricalMassEstimator(categorical_net)
