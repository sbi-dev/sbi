# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Optional

from torch import nn, Tensor, unique

from sbi.neural_nets.density_estimators import CategoricalMassEstimator, CategoricalNet


def build_categoricalmassestimator(
    input: Tensor,
    condition: Tensor,
    num_hidden: int = 20,
    num_layers: int = 2,
    embedding: Optional[nn.Module] = None,
):
    """Returns a density estimator for a categorical random variable."""
    # Infer input and output dims.
    dim_parameters = condition[0].numel()
    num_categories = unique(input).numel()

    categorical_net = CategoricalNet(
        num_input=dim_parameters,
        num_categories=num_categories,
        num_hidden=num_hidden,
        num_layers=num_layers,
        embedding=embedding,
    )

    categorical_mass_estimator = CategoricalMassEstimator(categorical_net)

    return categorical_mass_estimator
