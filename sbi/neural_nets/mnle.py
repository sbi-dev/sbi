# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Optional

import torch
from torch import Tensor, nn

from sbi.neural_nets.categorial import build_categoricalmassestimator
from sbi.neural_nets.density_estimators import MixedDensityEstimator
from sbi.neural_nets.density_estimators.mixed_density_estimator import _separate_input
from sbi.neural_nets.embedding_nets import PartialEmbedding
from sbi.neural_nets.flow import (
    build_made,
    build_maf,
    build_maf_rqs,
    build_nsf,
    build_zuko_bpf,
    build_zuko_gf,
    build_zuko_maf,
    build_zuko_naf,
    build_zuko_ncsf,
    build_zuko_nice,
    build_zuko_nsf,
    build_zuko_sospf,
    build_zuko_unaf,
)
from sbi.neural_nets.mdn import build_mdn
from sbi.utils.user_input_checks import check_data_device

model_builders = {
    "mdn": build_mdn,
    "made": build_made,
    "maf": build_maf,
    "maf_rqs": build_maf_rqs,
    "nsf": build_nsf,
    "zuko_nice": build_zuko_nice,
    "zuko_maf": build_zuko_maf,
    "zuko_nsf": build_zuko_nsf,
    "zuko_ncsf": build_zuko_ncsf,
    "zuko_sospf": build_zuko_sospf,
    "zuko_naf": build_zuko_naf,
    "zuko_unaf": build_zuko_unaf,
    "zuko_gf": build_zuko_gf,
    "zuko_bpf": build_zuko_bpf,
}


def build_mnle(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    flow_model: str = "nsf",
    embedding_net: nn.Module = nn.Identity(),
    num_transforms: int = 2,
    num_components: int = 5,
    num_bins: int = 5,
    hidden_features: int = 50,
    hidden_layers: int = 2,
    tail_bound: float = 10.0,
    log_transform_x: bool = True,
    **kwargs,
):
    """Returns a density estimator for mixed data types.

    Uses a categorical net to model the discrete part and a conditional density
    estimator to model the continuous part of the data.

    Args:
        batch_x: Batch of xs, used to infer dimensionality. batch_y: Batch of
        ys, used to infer dimensionality and (optional) z-scoring. z_score_x:
        Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean
              and std
            over the entire batch, instead of per-dimension. Should be used when
            each sample is, for example, a time series or an image. For MNLE,
            this applies to the continuous part of the data only.
        z_score_y: Whether to z-score ys passing into the network, same options
            as z_score_x.
        flow_model: type of flow model to use for the continuous part of the
            data.
        embedding_net: Optional embedding network for y.
        num_transforms: number of transforms in the flow model.
        num_components: number of components in the mixture model.
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
    discrete_net = build_categoricalmassestimator(
        disc_x,
        batch_y,
        z_score_x="none",  # discrete data should not be z-scored.
        z_score_y=z_score_y,
        num_hidden=hidden_features,
        num_layers=hidden_layers,
        embedding_net=embedding_net,
    )

    # for the continuous part, we need to construct a new embedding net for the
    # condition which takes both the discrete part disc_x and the parameters in
    # batch_y, but embeds only the parameters in batch_y.
    mixed_embedding = PartialEmbedding(embedding_net, num_dims_skipped=disc_x.shape[1])

    # Set up a flow for modelling the continuous data, conditioned on the discrete data.
    continuous_net = model_builders[flow_model](
        batch_x=(
            torch.log(cont_x) if log_transform_x else cont_x
        ),  # log transform manually.
        batch_y=torch.cat((batch_y, disc_x), dim=1),  # condition on discrete data too.
        z_score_y=z_score_y,
        z_score_x=z_score_x,
        embedding_net=mixed_embedding,
        num_bins=num_bins,
        num_transforms=num_transforms,
        num_components=num_components,
        tail_bound=tail_bound,
        hidden_features=hidden_features,
    )

    return MixedDensityEstimator(
        discrete_net=discrete_net,
        continuous_net=continuous_net,
        log_transform_input=log_transform_x,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
    )
