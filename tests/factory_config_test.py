# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Tests for config-based kwarg validation in factory functions.

The primary contract: unknown / misspelled kwargs emit a warning (so typos are
surfaced) but are still forwarded to the downstream builder, allowing
library-specific parameters (e.g. Zuko flow kwargs) to pass through.
"""

import pytest
import torch

from sbi.neural_nets.factory import (
    ZukoFlowType,
    classifier_nn,
    likelihood_nn,
    marginal_nn,
    posterior_flow_nn,
    posterior_nn,
    posterior_score_nn,
)
from sbi.neural_nets.net_builders.estimator_configs import ConditionalFlowConfig


def test_config_to_dict_filters_none():
    """to_dict() returns only explicitly-set fields."""
    cfg = ConditionalFlowConfig(hidden_features=64)
    d = cfg.to_dict()
    assert d == {"hidden_features": 64}
    assert "z_score_x" not in d


def test_config_from_kwargs_extra():
    """from_kwargs() stores unknown keys in extra_kwargs and merges in to_dict()."""
    with pytest.warns(UserWarning, match="Unknown kwargs"):
        cfg = ConditionalFlowConfig.from_kwargs(
            hidden_features=64, some_zuko_param=True
        )
    d = cfg.to_dict()
    assert d == {"hidden_features": 64, "some_zuko_param": True}


@pytest.mark.parametrize(
    "factory_fn, factory_args, bad_kwarg",
    [
        (posterior_nn, ("maf",), {"hiden_features": 64}),
        (likelihood_nn, ("maf",), {"num_tranforms": 3}),
        (classifier_nn, ("resnet",), {"drpout_probability": 0.5}),
        (marginal_nn, (ZukoFlowType.NSF,), {"num_tranforms": 3}),
        (posterior_score_nn, (), {"sigmaMin": 0.01}),
        (posterior_flow_nn, (), {"hiden_features": 64}),
        (posterior_flow_nn, (), {"sigma_min": 0.01}),  # score-only param
    ],
)
def test_factory_warns_on_unknown_kwargs(factory_fn, factory_args, bad_kwarg):
    with pytest.warns(UserWarning, match="Unknown kwargs"):
        factory_fn(*factory_args, **bad_kwarg)


def test_posterior_nn_accepts_valid_extra_kwargs():
    build_fn = posterior_nn("maf", dtype=torch.float64)
    assert callable(build_fn)
