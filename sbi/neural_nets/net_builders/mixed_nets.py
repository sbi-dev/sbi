# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from dataclasses import replace

import torch
from torch import Tensor, nn

from sbi.neural_nets.estimators import MixedDensityEstimator
from sbi.neural_nets.estimators.mixed_density_estimator import (
    _is_discrete,
    _separate_input,
)
from sbi.neural_nets.net_builders.categorial import (
    build_categoricalmassestimator,
)
from sbi.neural_nets.net_builders.estimator_configs import (
    MixedConfig,
    _factory_defaults,
    _mixed_config_from_factory_kwargs,
)
from sbi.utils.sbiutils import standardizing_net, z_score_parser
from sbi.utils.user_input_checks import check_data_device


def _build_mixed_density_estimator(
    batch_x: Tensor,
    batch_y: Tensor,
    config: MixedConfig,
) -> MixedDensityEstimator:
    """Build a mixed density estimator from its typed configuration.

    This function contains the shared logic between MNLE and MNPE.

    Args:
        batch_x: Batch of xs, used to infer dimensionality.
        batch_y: Batch of ys, used to infer dimensionality and (optional)
            z-scoring.
        config: Mixed estimator configuration.

    Returns:
        MixedDensityEstimator: nn.Module for performing MNLE or MNPE.
    """
    check_data_device(batch_x, batch_y)

    continuous_hidden_features = config.continuous.hidden_features  # type: ignore[attr-defined]
    _discrete_hf = (
        config.discrete_hidden_features
        if config.discrete_hidden_features is not None
        else continuous_hidden_features
    )

    warnings.warn(
        "The mixed neural density estimator assumes that inferred variable contains "
        "continuous data in the first n-k columns and "
        "categorical data in the last k columns.",
        stacklevel=2,
    )

    # Separate continuous and discrete data.
    if config.num_categories_per_variable is None:
        num_disc = int(torch.sum(_is_discrete(batch_x)))
    else:
        num_disc = len(config.num_categories_per_variable)
    cont_x, disc_x = _separate_input(batch_x, num_discrete_columns=num_disc)

    embedding_net = config.embedding_net
    z_score_y_bool, structured_y = z_score_parser(config.z_score_condition)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    embedded_batch_y = embedding_net(batch_y)
    combined_condition = torch.cat([disc_x, embedded_batch_y], dim=-1)

    # Set up a categorical RV neural net for modelling the discrete data.
    discrete_net = build_categoricalmassestimator(
        disc_x,
        batch_y,
        z_score_x="none",  # discrete data should not be z-scored
        z_score_y="none",  # y-embedding net already z-scores
        num_hidden=_discrete_hf,
        num_layers=config.discrete_hidden_layers,
        embedding_net=embedding_net,
        num_categories_per_variable=config.num_categories_per_variable,
        dropout_probability=config.dropout_probability,
    )

    combined_embedding_net = config.combined_embedding_net
    if combined_embedding_net is None:
        _combined_hf = (
            config.combined_embedding_features
            if config.combined_embedding_features is not None
            else continuous_hidden_features
        )
        combined_embedding_net = nn.Sequential(
            nn.Linear(combined_condition.shape[-1], _combined_hf),
            nn.ReLU(),
            nn.Linear(_combined_hf, _combined_hf),
            nn.ReLU(),
        )

    # TODO: add support for optional log-transform in flow builders.
    continuous_x = torch.log(cont_x + 1e-10) if config.log_transform_x else cont_x

    # The combined condition is already z-scored and embedded here.
    continuous_net = replace(
        config.continuous,
        z_score_condition="none",
        embedding_net=combined_embedding_net,
    ).build(batch_input=continuous_x, batch_condition=combined_condition)

    return MixedDensityEstimator(
        discrete_net=discrete_net,
        continuous_net=continuous_net,
        embedding_net=embedding_net,  # pass embedding for continuous condition part.
        log_transform_input=config.log_transform_x,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
    )


def _config_from_flat_kwargs(log_transform_x: bool, kwargs: dict) -> MixedConfig:
    """Translate the exported mixed builders' legacy flat arguments."""
    from sbi.neural_nets.factory import _LIKELIHOOD_FACTORY_FIELDS, likelihood_nn

    extra = dict(kwargs)
    family_defaults = _factory_defaults(likelihood_nn, _LIKELIHOOD_FACTORY_FIELDS)
    flat_names = {"z_score_input": "z_score_x", "z_score_condition": "z_score_y"}
    family_args = {
        name: extra.pop(flat_names.get(name, name), default)
        for name, default in family_defaults.items()
    }
    for name in ("z_score_input", "z_score_condition"):
        if family_args[name] is None:
            family_args[name] = "none"
    extra["log_transform_x"] = log_transform_x
    return _mixed_config_from_factory_kwargs(
        family_args=family_args,
        factory_defaults=family_defaults,
        extra=extra,
    )


def build_mnle(
    batch_x: Tensor,
    batch_y: Tensor,
    log_transform_x: bool = False,
    **kwargs,
) -> MixedDensityEstimator:
    """Returns a mixed neural likelihood estimator.

    This estimator models p(x|theta) where x contains both continuous and discrete data.

    Args:
        batch_x: Batch of xs (data), used to infer dimensionality.
        batch_y: Batch of ys (parameters), used to infer dimensionality.
        log_transform_x: whether to apply a log-transform to x. This is by default false
            because x has to be strictly positive to apply log-transform.
        **kwargs: Legacy flat mixed-estimator arguments.

    Returns:
        MixedDensityEstimator for MNLE.
    """
    return _config_from_flat_kwargs(log_transform_x, kwargs).build(batch_x, batch_y)


def build_mnpe(
    batch_x: Tensor,
    batch_y: Tensor,
    log_transform_x: bool = False,
    **kwargs,
) -> MixedDensityEstimator:
    """Returns a mixed neural posterior estimator.

    This estimator models p(theta|x) where x contains both continuous and discrete data.

    Args:
        batch_x: Batch of xs (parameters), used to infer dimensionality.
        batch_y: Batch of ys (data), used to infer dimensionality.
        log_transform_x: whether to apply a log-transform to x. This is by default false
            because x has to be strictly positive to apply log-transform.
        **kwargs: Legacy flat mixed-estimator arguments.

    Returns:
        MixedDensityEstimator for MNPE.
    """
    return _config_from_flat_kwargs(log_transform_x, kwargs).build(batch_x, batch_y)
