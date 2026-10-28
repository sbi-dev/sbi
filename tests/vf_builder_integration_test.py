# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Tests for per-model vector-field configs."""

import inspect
from dataclasses import fields as dc_fields
from typing import get_args

import pytest
import torch
from torch import nn, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import FMPE, NPSE
from sbi.neural_nets.estimators.flowmatching_estimator import FlowMatchingEstimator
from sbi.neural_nets.estimators.score_estimator import (
    SubVPScoreEstimator,
    VEScoreEstimator,
    VPScoreEstimator,
)
from sbi.neural_nets.factory import posterior_flow_nn, posterior_score_nn
from sbi.neural_nets.net_builders.estimator_configs import (
    MAFConfig,
)
from sbi.neural_nets.net_builders.vector_field_nets import (
    AdaMLPConfig,
    FlowMatchingConfig,
    MLPConfig,
    SubVPScoreConfig,
    TransformerConfig,
    VEScoreConfig,
    VPScoreConfig,
    build_standard_mlp_network,
)
from sbi.utils.vector_field_utils import VectorFieldNet

NET_CONFIGS = [MLPConfig, AdaMLPConfig, TransformerConfig]
SCORE_CONFIGS = [VEScoreConfig, VPScoreConfig, SubVPScoreConfig]
ALL_CONFIGS = [FlowMatchingConfig, *SCORE_CONFIGS]
NET_CLASS_NAMES = {
    MLPConfig: "VectorFieldMLP",
    AdaMLPConfig: "VectorFieldAdaMLP",
    TransformerConfig: "VectorFieldTransformer",
}


@pytest.fixture
def gaussian_sims():
    prior = MultivariateNormal(zeros(2), torch.eye(2))
    theta = prior.sample((200,))
    x = theta + 0.1 * torch.randn_like(theta)
    return prior, theta, x


@pytest.fixture
def batches():
    return torch.randn(32, 2), torch.randn(32, 3)


def _assert_same_state(actual, expected):
    assert type(actual) is type(expected)
    assert type(actual.net) is type(expected.net)
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name])


def test_net_config_rejects_a_list_of_hidden_features():
    with pytest.raises(TypeError, match="hidden_features"):
        MLPConfig(hidden_features=[16, 32])


@pytest.mark.parametrize("config_cls", ALL_CONFIGS + NET_CONFIGS)
def test_invalid_literal_value_raises(config_cls):
    field = "z_score_input" if config_cls in ALL_CONFIGS else "time_emb_type"
    with pytest.raises(ValueError, match=field):
        config_cls(**{field: "not_a_value"})


@pytest.mark.parametrize("net", ["mlp", nn.Linear(3, 3)])
def test_estimator_config_rejects_an_invalid_network(net):
    with pytest.raises(TypeError, match="VectorFieldNet"):
        FlowMatchingConfig(net=net)


@pytest.mark.parametrize("condition_dim, valid", [(7, True), (3, False)])
def test_custom_network_is_checked_against_the_embedded_condition(
    condition_dim, valid, batches
):
    class CustomNet(VectorFieldNet):
        def __init__(self):
            super().__init__()
            self.norm = nn.BatchNorm1d(condition_dim)
            self.linear = nn.Linear(2 + condition_dim, 2)

        def forward(self, input, condition, time):
            return self.linear(torch.cat([input, self.norm(condition)], dim=-1))

    net = CustomNet()
    config = FlowMatchingConfig(net=net, embedding_net=nn.Linear(3, 7))
    if not valid:
        with pytest.raises(ValueError, match=r"embedded condition shape \(7,\)"):
            config.build(*batches)
        return

    running_mean = net.norm.running_mean.clone()
    estimator = config.build(*batches)

    assert estimator.net is net
    assert net.training
    assert torch.equal(net.norm.running_mean, running_mean)
    assert torch.isfinite(estimator.loss(*batches)).all()


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_embedding_net_is_wired_once(config_cls, batches):
    theta, x = batches
    embedding_net = nn.Linear(3, 7)
    estimator = config_cls(embedding_net=embedding_net).build(theta, x)

    assert not any(m is embedding_net for m in estimator.net.modules())
    assert any(m is embedding_net for m in estimator._embedding_net.modules())


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_compose_standardization_is_set_by_the_constructor(config_cls, batches):
    estimator = config_cls(compose_standardization=True).build(*batches)

    assert estimator.compose_enabled
    assert (estimator.mean_0 == 0).all() and (estimator.std_0 == 1).all()


@pytest.mark.parametrize("z_score_input", [None, "none", "structured"])
def test_compose_standardization_requires_independent_z_scoring(z_score_input):
    with pytest.raises(ValueError, match="z_score_input='independent'"):
        VEScoreConfig(compose_standardization=True, z_score_input=z_score_input)


def test_compose_standardization_rejects_the_gaussian_baseline():
    with pytest.raises(ValueError, match="gaussian_baseline"):
        FlowMatchingConfig(compose_standardization=True, gaussian_baseline=True)


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
def test_z_scoring_of_the_condition_wraps_the_embedding(config_cls, batches):
    theta, x = batches
    without = config_cls(z_score_condition="none").build(theta, x)
    with_zscore = config_cls(z_score_condition="independent").build(theta, x)

    assert isinstance(without._embedding_net, nn.Identity)
    assert isinstance(with_zscore._embedding_net, nn.Sequential)


@pytest.mark.parametrize(
    "trainer_cls, config_cls, estimator_cls",
    [
        (FMPE, FlowMatchingConfig, FlowMatchingEstimator),
        (NPSE, VPScoreConfig, VPScoreEstimator),
    ],
)
def test_trainer_trains_and_samples_with_a_config(
    trainer_cls, config_cls, estimator_cls, gaussian_sims
):
    prior, theta, x = gaussian_sims
    config = config_cls(net=MLPConfig(hidden_features=16, num_layers=2))
    trainer = trainer_cls(prior, config, show_progress_bars=False)
    estimator = trainer.append_simulations(theta, x).train(
        max_num_epochs=1, training_batch_size=100
    )
    assert isinstance(estimator, estimator_cls)
    posterior = trainer.build_posterior(estimator)
    samples = posterior.sample(
        (5,),
        x=x[:1],
        steps=3,
        reject_outside_prior=False,
        show_progress_bars=False,
    )
    assert samples.shape == (5, theta.shape[-1])
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize(
    "trainer_cls, wrong_config",
    [
        (FMPE, VEScoreConfig()),
        (FMPE, VPScoreConfig()),
        (NPSE, FlowMatchingConfig()),
        (FMPE, MAFConfig()),
        (NPSE, MAFConfig()),
    ],
)
def test_trainer_rejects_the_wrong_family(trainer_cls, wrong_config, gaussian_sims):
    prior, _, _ = gaussian_sims
    with pytest.raises(TypeError, match="requires a"):
        trainer_cls(prior, wrong_config, show_progress_bars=False)


def test_npse_rejects_sde_type_together_with_a_config(gaussian_sims):
    prior, _, _ = gaussian_sims
    with pytest.raises(ValueError, match="already selects the SDE"):
        NPSE(prior, VEScoreConfig(), sde_type="vp", show_progress_bars=False)


@pytest.mark.parametrize("trainer_cls", [FMPE, NPSE])
def test_string_path_warns_and_names_the_import(trainer_cls, gaussian_sims):
    prior, _, _ = gaussian_sims
    with pytest.warns(FutureWarning, match="from sbi.neural_nets import"):
        trainer_cls(prior, "mlp", show_progress_bars=False)


@pytest.mark.parametrize(
    "trainer_cls, kwarg",
    [
        (FMPE, "density_estimator"),
        (NPSE, "score_estimator"),
        (NPSE, "density_estimator"),
    ],
)
def test_legacy_kwarg_warns(trainer_cls, kwarg, gaussian_sims):
    prior, _, _ = gaussian_sims
    with pytest.warns(FutureWarning, match="deprecated"):
        trainer_cls(prior, **{kwarg: "mlp"}, show_progress_bars=False)


@pytest.mark.parametrize(
    "trainer_cls, kwarg",
    [
        (FMPE, "density_estimator"),
        (NPSE, "score_estimator"),
        (NPSE, "density_estimator"),
    ],
)
def test_legacy_and_vf_estimator_conflict(trainer_cls, kwarg, gaussian_sims):
    prior, _, _ = gaussian_sims
    with pytest.raises(ValueError, match="Cannot pass both"):
        trainer_cls(
            prior, vf_estimator="mlp", **{kwarg: "mlp"}, show_progress_bars=False
        )


@pytest.mark.parametrize(
    "factory_fn", [posterior_flow_nn, posterior_score_nn], ids=["flow", "score"]
)
def test_advertised_time_emb_types_all_build(factory_fn):
    annotation = inspect.signature(factory_fn).parameters["time_emb_type"].annotation
    values = get_args(annotation)
    assert values, "time_emb_type lost its Literal annotation"
    for value in values:
        builder = factory_fn(model="mlp", time_emb_type=value)
        builder(torch.randn(10, 2), torch.randn(10, 3))


def test_factory_routes_estimator_and_network_settings(batches):
    estimator = posterior_score_nn(
        model="transformer", hidden_features=64, num_heads=2, sigma_max=20.0
    )(*batches)

    assert estimator.sigma_max == 20.0
    assert {
        m.num_heads for m in estimator.net.modules() if hasattr(m, "num_heads")
    } == {2}


def test_factory_rejects_network_settings_for_a_custom_network(batches):
    theta, x = batches
    custom = build_standard_mlp_network(theta, x)
    with pytest.raises(ValueError, match="silently ignored"):
        posterior_flow_nn(model=custom, hidden_features=64)


@pytest.mark.parametrize(
    "factory_fn, factory_kwargs, config_cls, estimator_cls",
    [
        (posterior_flow_nn, {}, FlowMatchingConfig, FlowMatchingEstimator),
        (posterior_score_nn, {"sde_type": "ve"}, VEScoreConfig, VEScoreEstimator),
        (posterior_score_nn, {"sde_type": "vp"}, VPScoreConfig, VPScoreEstimator),
        (
            posterior_score_nn,
            {"sde_type": "subvp"},
            SubVPScoreConfig,
            SubVPScoreEstimator,
        ),
    ],
    ids=["flow", "ve", "vp", "subvp"],
)
@pytest.mark.parametrize(
    "model, net_config",
    [
        (None, MLPConfig()),
        ("mlp", MLPConfig()),
        ("ada_mlp", AdaMLPConfig()),
        ("transformer", TransformerConfig()),
        ("transformer_cross_attn", TransformerConfig(is_x_emb_seq=True)),
    ],
)
def test_config_defaults_match_the_factory(
    factory_fn, factory_kwargs, config_cls, estimator_cls, model, net_config, batches
):
    theta, condition = batches
    if model == "transformer_cross_attn":
        condition = torch.randn(32, 5, 3)
    if model is not None:
        factory_kwargs = {**factory_kwargs, "model": model}
    config = config_cls() if model is None else config_cls(net=net_config)

    torch.manual_seed(0)
    from_factory = factory_fn(**factory_kwargs)(theta, condition)
    torch.manual_seed(0)
    from_config = config.build(theta, condition)

    _assert_same_state(from_factory, from_config)
    assert isinstance(from_config, estimator_cls)
    assert type(from_config.net).__name__ == NET_CLASS_NAMES[type(net_config)]


@pytest.mark.parametrize("factory_fn", [posterior_flow_nn, posterior_score_nn])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_features": None},
        {"num_layers": None},
        {"t_embedding_dim": None},
        {"time_emb_type": None},
        {"layer_norm": None, "skip_connections": None},
        {"num_heads": None},
        {"sigma_min": None, "train_schedule": None},
        {"beta_min": None},
    ],
)
def test_factory_none_keeps_known_field_defaults(factory_fn, kwargs, batches):
    torch.manual_seed(0)
    expected = factory_fn()(*batches)
    torch.manual_seed(0)
    actual = factory_fn(**kwargs)(*batches)

    _assert_same_state(actual, expected)


@pytest.mark.parametrize("factory_fn", [posterior_flow_nn, posterior_score_nn])
def test_factory_none_means_no_z_scoring(factory_fn):
    theta = torch.randn(32, 2) + 5.0
    x = torch.randn(32, 3) + 7.0

    from_none = factory_fn(z_score_theta=None, z_score_x=None)(theta, x)
    explicit = factory_fn(z_score_theta="none", z_score_x="none")(theta, x)
    default = factory_fn()(theta, x)

    assert torch.equal(from_none.mean_0, explicit.mean_0)
    assert torch.equal(from_none.std_0, explicit.std_0)
    assert isinstance(from_none._embedding_net, nn.Identity)
    assert isinstance(explicit._embedding_net, nn.Identity)
    assert not torch.equal(default.mean_0, from_none.mean_0)
    assert isinstance(default._embedding_net, nn.Sequential)


@pytest.mark.parametrize(
    "trainer_cls, trainer_kwargs, config",
    [
        (FMPE, {}, FlowMatchingConfig()),
        (NPSE, {}, VEScoreConfig()),
        (NPSE, {"sde_type": "vp"}, VPScoreConfig()),
        (NPSE, {"sde_type": "subvp"}, SubVPScoreConfig()),
    ],
)
def test_trainer_default_matches_the_config(
    trainer_cls, trainer_kwargs, config, batches
):
    # theta and x differ in size, so swapped roles change the weight shapes.
    theta, x = batches
    prior = MultivariateNormal(zeros(2), torch.eye(2))
    trainer = trainer_cls(prior, **trainer_kwargs, show_progress_bars=False)

    torch.manual_seed(0)
    from_trainer = trainer._build_neural_net(theta, x)
    torch.manual_seed(0)
    from_config = config.build(theta, x)
    _assert_same_state(from_trainer, from_config)


def test_estimator_extra_kwargs_are_forwarded(batches):
    estimator = VEScoreConfig(extra_kwargs={"t_max": 0.9}).build(*batches)
    assert estimator.t_max == 0.9


@pytest.mark.parametrize("config_cls", ALL_CONFIGS + NET_CONFIGS)
def test_extra_kwargs_rejects_a_name_that_is_a_field(config_cls):
    name = dc_fields(config_cls)[0].name
    with pytest.raises(ValueError, match="Pass the"):
        config_cls(extra_kwargs={name: None})
