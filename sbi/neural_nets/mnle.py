# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Optional

import torch
from torch import Tensor

from sbi.neural_nets.categorial import build_categoricalmassestimator
from sbi.neural_nets.density_estimators import MixedDensityEstimator
from sbi.neural_nets.density_estimators.mixed_density_estimator import _separate_input
from sbi.neural_nets.flow import build_nsf
from sbi.utils.sbiutils import standardizing_net
from sbi.utils.user_input_checks import check_data_device


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
    embedding_net = standardizing_net(batch_y) if z_score_y == "independent" else None

    warnings.warn(
        """The mixed neural likelihood estimator assumes that x contains
        continuous data in the first n-1 columns (e.g., reaction times) and
        categorical data in the last column (e.g., corresponding choices). If
        this is not the case for the passed `x` do not use this function.""",
        stacklevel=2,
    )
    # Separate continuous and discrete data.
    cont_x, disc_x = _separate_input(batch_x)

    # Set up a categorical RV neural net for modelling the discrete data.
    disc_nle = build_categoricalmassestimator(
        disc_x,
        batch_y,
        num_hidden=hidden_features,
        num_layers=hidden_layers,
        embedding_net=embedding_net,
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
        log_transform_input=log_transform_x,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
    )
