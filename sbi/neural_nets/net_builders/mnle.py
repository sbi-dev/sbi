# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Optional

import torch
from torch import Tensor, nn

from sbi.neural_nets.estimators import MixedDensityEstimator
from sbi.neural_nets.estimators.mixed_density_estimator import _separate_input
from sbi.neural_nets.net_builders.categorial import build_categoricalmassestimator
from sbi.neural_nets.net_builders.flow import (
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
from sbi.neural_nets.net_builders.mdn import build_mdn
from sbi.utils.sbiutils import (
    standardizing_net,
    z_score_parser,
)
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
    combined_embedding_net: Optional[nn.Module] = None,
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

    Note: If the condition y is > 1D, an embedding net must be provided. Then,
    during inference, we need to combine the embedded condition with the
    discrete part of the input data. To do this, we use a combined embedding net
    that takes the discrete part of the input and the embedded condition as
    input and passes it on to the flow model.
    To this end, we build the z-scoring and the embedding net in this function
    here, i.e., outside of the flow model, and then pass z_score_x="none" to the
    flow builder.
    The y-embedding is passed to MixedDensityEstimator, which then uses it to
    embed the continuous part and concatenate it with the discrete part during
    log_prob evaluation and sampling.
    The combined embedding net is passed to the flow model, which then uses it
    to combine the already embedded and combined condition.

    Args:
        batch_x: Batch of xs, used to infer dimensionality.
        batch_y: Batch of ys, used to infer dimensionality and (optional)
            z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
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
        embedding_net: Optional embedding network for y, required if y is > 1D.
        combined_embedding_net: Optional embedding for combining the discrete
            part of the input and the embedded condition into a joined
            condition. If None, a shallow MLP is used.
        num_transforms: number of transforms in the flow model.
        num_components: number of components in the mixture model.
        num_bins: bins per spline for NSF.
        hidden_features: number of hidden features used in both nets.
        hidden_layers: number of hidden layers in the categorical net.
        tail_bound: spline tail bound for NSF.
        log_transform_x: whether to apply a log-transform to x to move it to unbounded
            space, e.g., in case x consists of reaction time data (bounded by
            zero).
        kwargs: additional keyword arguments passed to the flow model.

    Returns:
        MixedDensityEstimator: nn.Module for performing MNLE.
    """

    check_data_device(batch_x, batch_y)

    warnings.warn(
        "The mixed neural likelihood estimator assumes that x contains "
        "continuous data in the first n-1 columns (e.g., reaction times) and "
        "categorical data in the last column (e.g., corresponding choices). If "
        "this is not the case for the passed `x` do not use this function.",
        stacklevel=2,
    )
    # Separate continuous and discrete data.
    cont_x, disc_x = _separate_input(batch_x)

    # Set up y-embedding net with z-scoring.
    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )
    # Get size of the embedded condition data.
    embedded_batch_y = embedding_net(batch_y)
    combined_condition = torch.cat([disc_x, embedded_batch_y], dim=-1)

    # Set up a categorical RV neural net for modelling the discrete data.
    discrete_net = build_categoricalmassestimator(
        disc_x,
        batch_y,
        z_score_x="none",  # discrete data should not be z-scored.
        z_score_y="none",  # y-embedding net already z-scores.
        num_hidden=hidden_features,
        num_layers=hidden_layers,
        embedding_net=embedding_net,
    )

    if combined_embedding_net is None:
        # set up linear embedding net for combining discrete and continuous
        # data.
        combined_embedding_net = nn.Sequential(
            nn.Linear(combined_condition.shape[-1], hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        )

    # Set up a flow for modelling the continuous data, conditioned on the discrete data.
    continuous_net = model_builders[flow_model](
        # TODO: add support for optional log-transform in flow builders.
        batch_x=(
            torch.log(cont_x) if log_transform_x else cont_x
        ),  # log transform manually.
        batch_y=combined_condition,
        z_score_x=z_score_x,
        z_score_y="none",  # combined condition is already z-scored.
        # combined embedding net for discrete and continuous data.
        embedding_net=combined_embedding_net,
        num_bins=num_bins,
        num_transforms=num_transforms,
        num_components=num_components,
        tail_bound=tail_bound,
        hidden_features=hidden_features,
        **kwargs,
    )

    return MixedDensityEstimator(
        discrete_net=discrete_net,
        continuous_net=continuous_net,
        embedding_net=embedding_net,  # pass embedding for continuous condition part.
        log_transform_input=log_transform_x,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
    )
