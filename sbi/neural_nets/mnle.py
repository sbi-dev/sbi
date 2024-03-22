# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor, nn, unique

from sbi.neural_nets.density_estimators import (
    CategoricalMassEstimator,
    CategoricalNet,
    MixedDensityEstimator,
)
from sbi.neural_nets.flow import build_nsf
from sbi.utils.sbiutils import standardizing_net
from sbi.utils.user_input_checks import check_data_device


def build_categoricalmassestimator(
    num_input: int = 4,
    num_categories: int = 2,
    num_hidden: int = 20,
    num_layers: int = 2,
    embedding: Optional[nn.Module] = None,
):
    """Returns a density estimator for a categorical random variable."""

    categorical_net = CategoricalNet(
        num_input=num_input,
        num_categories=num_categories,
        num_hidden=num_hidden,
        num_layers=num_layers,
        embedding=embedding,
    )

    categorical_mass_estimator = CategoricalMassEstimator(categorical_net)

    return categorical_mass_estimator


def build_mnle(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    num_transforms: int = 2,
    num_bins: int = 5,
    hidden_features: int = 50,
    hidden_layers: int = 2,
    tail_bound: float = 10.0,
    log_transform_x: bool = True,
    **kwargs,
):
    """Returns a density estimator for mixed data types.

    Uses a categorical net to model the discrete part and a neural spline flow (NSF) to
    model the continuous part of the data.

    Args:
        batch_x: batch of data
        batch_y: batch of parameters
        z_score_x: whether to z-score x.
        z_score_y: whether to z-score y.
        num_transforms: number of transforms in the NSF
        num_bins: bins per spline for NSF.
        hidden_features: number of hidden features used in both nets.
        hidden_layers: number of hidden layers in the categorical net.
        tail_bound: spline tail bound for NSF.
        log_transform_x: whether to apply a log-transform to x to move it to unbounded
            space, e.g., in case x consists of reaction time data (bounded by zero).

    Returns:
        MixedDensityEstimator: nn.Module for performing MNLE.
    """

    check_data_device(batch_x, batch_y)
    embedding = standardizing_net(batch_y) if z_score_y == "independent" else None

    warnings.warn(
        """The mixed neural likelihood estimator assumes that x contains
        continuous data in the first n-1 columns (e.g., reaction times) and
        categorical data in the last column (e.g., corresponding choices). If
        this is not the case for the passed `x` do not use this function.""",
        stacklevel=2,
    )
    # Separate continuous and discrete data.
    cont_x, disc_x = _separate_x(batch_x)

    # Infer input and output dims.
    dim_parameters = batch_y[0].numel()
    num_categories = unique(disc_x).numel()

    # Set up a categorical RV neural net for modelling the discrete data.
    disc_nle = build_categoricalmassestimator(
        num_input=dim_parameters,
        num_categories=num_categories,
        num_hidden=hidden_features,
        num_layers=hidden_layers,
        embedding=embedding,
    )

    # Set up a NSF for modelling the continuous data, conditioned on the discrete data.
    cont_nle = build_nsf(
        batch_x=(
            torch.log(cont_x) if log_transform_x else cont_x
        ),  # log transform manually.
        batch_y=torch.cat((batch_y, disc_x), dim=1),  # condition on discrete data too.
        z_score_y=z_score_y,
        z_score_x=z_score_x,
        num_bins=num_bins,
        num_transforms=num_transforms,
        tail_bound=tail_bound,
        hidden_features=hidden_features,
    )

    return MixedDensityEstimator(
        discrete_net=disc_nle,
        continuous_net=cont_nle,
        log_transform_x=log_transform_x,
        condition_shape=torch.Size([]),
    )


def _separate_x(x: Tensor, num_discrete_columns: int = 1) -> Tuple[Tensor, Tensor]:
    """Returns the continuous and discrete part of the given x.

    Assumes the discrete data to live in the last columns of x.
    """

    assert x.ndim == 2, f"x must have two dimensions but has {x.ndim}."

    return x[:, :-num_discrete_columns], x[:, -num_discrete_columns:]
