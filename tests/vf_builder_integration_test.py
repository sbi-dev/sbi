# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Tests for per-model vector-field configs."""

import inspect
import warnings
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
    _VALID_VF_MODELS,
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
    VectorFieldConfigBase,
    VectorFieldMLP,
    _VectorFieldNetConfigBase,
    _vf_net_config_from_model,
    build_standard_mlp_network,
)

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


@pytest.mark.parametrize(
    "config_cls, estimator_cls",
    [
        (FlowMatchingConfig, FlowMatchingEstimator),
        (VEScoreConfig, VEScoreEstimator),
        (VPScoreConfig, VPScoreEstimator),
        (SubVPScoreConfig, SubVPScoreEstimator),
    ],
)
def test_config_class_selects_the_estimator(config_cls, estimator_cls, batches):
    assert isinstance(config_cls().build(*batches), estimator_cls)


@pytest.mark.parametrize("config_cls", ALL_CONFIGS)
@pytest.mark.parametrize("net_cls", NET_CONFIGS)
def test_net_choice_is_independent_of_the_estimator(config_cls, net_cls, batches):
    estimator = config_cls(net=net_cls()).build(*batches)
    assert type(estimator.net).__name__ == NET_CLASS_NAMES[net_cls]


@pytest.mark.parametrize(
    "config_cls, bad_kwarg",
    [
        (FlowMatchingConfig, {"sigma_min": 0.1}),
        (FlowMatchingConfig, {"beta_min": 0.1}),
        (FlowMatchingConfig, {"sde_type": "vp"}),
        (VEScoreConfig, {"gaussian_baseline": True}),
        (VEScoreConfig, {"beta_min": 0.1}),
        (VPScoreConfig, {"sigma_max": 5.0}),
        (SubVPScoreConfig, {"train_schedule": "lognormal"}),
    ],
)
def test_estimator_config_rejects_a_setting_it_does_not_have(config_cls, bad_kwarg):
    with pytest.raises(TypeError):
        config_cls(**bad_kwarg)


@pytest.mark.parametrize(
    "net_cls, bad_kwarg",
    [
        (MLPConfig, {"num_heads": 4}),
        (MLPConfig, {"mlp_ratio": 2}),
        (MLPConfig, {"adamlp_ratio": 2}),
        (MLPConfig, {"hidden_features": [16, 32]}),
        (AdaMLPConfig, {"layer_norm": False}),
        (AdaMLPConfig, {"num_heads": 4}),
        (TransformerConfig, {"layer_norm": False}),
        (TransformerConfig, {"adamlp_ratio": 2}),
    ],
)
def test_net_config_rejects_a_setting_it_does_not_have(net_cls, bad_kwarg):
    with pytest.raises(TypeError):
        net_cls(**bad_kwarg)


@pytest.mark.parametrize("config_cls", ALL_CONFIGS + NET_CONFIGS)
def test_invalid_literal_value_raises(config_cls):
    field = "z_score_input" if config_cls in ALL_CONFIGS else "time_emb_type"
    with pytest.raises(ValueError, match=field):
        config_cls(**{field: "not_a_value"})


@pytest.mark.parametrize("base_cls", [VectorFieldConfigBase, _VectorFieldNetConfigBase])
def test_role_base_cannot_be_instantiated(base_cls):
    with pytest.raises(TypeError, match="per-model config"):
        base_cls()


@pytest.mark.parametrize("model", sorted(_VALID_VF_MODELS))
def test_every_advertised_model_maps_to_a_net_config(model):
    net_config = _vf_net_config_from_model(model)
    assert isinstance(net_config, _VectorFieldNetConfigBase)
    if model == "transformer_cross_attn":
        assert net_config.is_x_emb_seq


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown vector field model"):
        _vf_net_config_from_model("not_a_model")


@pytest.mark.parametrize("net_cls", NET_CONFIGS)
def test_net_config_builds_the_network_alone(net_cls, batches):
    assert isinstance(net_cls().build(*batches), nn.Module)


def test_cross_attention_takes_a_sequence_condition():
    theta, x_seq = torch.randn(32, 2), torch.randn(32, 5, 4)
    estimator = FlowMatchingConfig(net=TransformerConfig(is_x_emb_seq=True)).build(
        theta, x_seq
    )
    assert estimator.condition_shape == torch.Size([5, 4])


def test_custom_network_module_is_accepted(batches):
    theta, x = batches

    class CustomNet(nn.Module):
        def forward(self, input, condition, time):
            return torch.zeros_like(input)

    custom = CustomNet()
    assert FlowMatchingConfig(net=custom).build(theta, x).net is custom


def test_estimator_config_rejects_an_invalid_network():
    with pytest.raises(TypeError, match="nn.Module"):
        FlowMatchingConfig(net="mlp")


def test_net_config_rejects_embedding_net_in_extra_kwargs():
    with pytest.raises(ValueError, match="embedding_net"):
        MLPConfig(extra_kwargs={"embedding_net": nn.Identity()})


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


def test_compose_standardization_rejects_the_gaussian_baseline(batches):
    with pytest.raises(ValueError, match="gaussian_baseline"):
        FlowMatchingConfig(compose_standardization=True, gaussian_baseline=True).build(
            *batches
        )


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
        (NPSE, VEScoreConfig, VEScoreEstimator),
        (NPSE, VPScoreConfig, VPScoreEstimator),
        (NPSE, SubVPScoreConfig, SubVPScoreEstimator),
    ],
)
def test_trainer_trains_with_a_config(
    trainer_cls, config_cls, estimator_cls, gaussian_sims
):
    prior, theta, x = gaussian_sims
    config = config_cls(net=MLPConfig(hidden_features=16, num_layers=2))
    trainer = trainer_cls(prior, config, show_progress_bars=False)
    estimator = trainer.append_simulations(theta, x).train(
        max_num_epochs=1, training_batch_size=100
    )
    assert isinstance(estimator, estimator_cls)


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


@pytest.mark.parametrize(
    "trainer_cls, config_cls",
    [(FMPE, FlowMatchingConfig), (NPSE, VEScoreConfig)],
)
def test_trainer_rejects_a_config_class(trainer_cls, config_cls, gaussian_sims):
    prior, _, _ = gaussian_sims
    with pytest.raises(TypeError, match="not an instance"):
        trainer_cls(prior, config_cls, show_progress_bars=False)


@pytest.mark.parametrize(
    "sde_type, estimator_cls",
    [
        ("ve", VEScoreEstimator),
        ("vp", VPScoreEstimator),
        ("subvp", SubVPScoreEstimator),
    ],
)
def test_npse_sde_type_selects_the_config(sde_type, estimator_cls, gaussian_sims):
    prior, theta, x = gaussian_sims
    trainer = NPSE(prior, sde_type=sde_type, show_progress_bars=False)
    trainer.append_simulations(theta, x)
    assert isinstance(trainer._build_neural_net(theta, x), estimator_cls)


def test_npse_rejects_sde_type_together_with_a_config(gaussian_sims):
    prior, _, _ = gaussian_sims
    with pytest.raises(ValueError, match="already selects the SDE"):
        NPSE(prior, VEScoreConfig(), sde_type="vp", show_progress_bars=False)


@pytest.mark.parametrize("input_kind", ["default", "config", "callable"])
@pytest.mark.parametrize(
    "trainer_cls, config", [(FMPE, FlowMatchingConfig()), (NPSE, VEScoreConfig())]
)
def test_supported_estimator_inputs_do_not_warn(
    trainer_cls, config, input_kind, gaussian_sims
):
    prior, _, _ = gaussian_sims
    estimators = {
        "default": None,
        "config": config,
        "callable": lambda theta, x: None,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        trainer_cls(
            prior,
            vf_estimator=estimators[input_kind],
            show_progress_bars=False,
        )


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


@pytest.mark.parametrize("trainer_cls", [FMPE, NPSE])
def test_role_shapes_are_not_swapped(trainer_cls):
    prior = MultivariateNormal(zeros(2), torch.eye(2))
    theta, x = prior.sample((100,)), torch.randn(100, 5)
    trainer = trainer_cls(prior, show_progress_bars=False)
    trainer.append_simulations(theta, x)
    estimator = trainer._build_neural_net(theta, x)

    assert estimator.input_shape == torch.Size([2])
    assert estimator.condition_shape == torch.Size([5])


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


@pytest.mark.parametrize("model", sorted(_VALID_VF_MODELS - {"transformer_cross_attn"}))
@pytest.mark.parametrize(
    "factory_fn", [posterior_flow_nn, posterior_score_nn], ids=["flow", "score"]
)
def test_factory_builds_every_model(factory_fn, model, batches):
    assert factory_fn(model=model)(*batches) is not None


@pytest.mark.parametrize(
    "factory_fn, sde_kwargs, estimator_cls",
    [
        (posterior_flow_nn, {}, FlowMatchingEstimator),
        (posterior_score_nn, {"sde_type": "ve"}, VEScoreEstimator),
        (posterior_score_nn, {"sde_type": "vp"}, VPScoreEstimator),
        (posterior_score_nn, {"sde_type": "subvp"}, SubVPScoreEstimator),
    ],
)
def test_factory_sde_type_maps_to_the_config(
    factory_fn, sde_kwargs, estimator_cls, batches
):
    assert isinstance(factory_fn(**sde_kwargs)(*batches), estimator_cls)


def test_factory_routes_settings_to_the_axis_that_owns_them(batches):
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
    "factory_fn, factory_kwargs, config_cls",
    [
        (posterior_flow_nn, {}, FlowMatchingConfig),
        (posterior_score_nn, {"sde_type": "ve"}, VEScoreConfig),
        (posterior_score_nn, {"sde_type": "vp"}, VPScoreConfig),
        (posterior_score_nn, {"sde_type": "subvp"}, SubVPScoreConfig),
    ],
    ids=["flow", "ve", "vp", "subvp"],
)
def test_estimator_config_defaults_match_the_factory(
    factory_fn, factory_kwargs, config_cls, batches
):
    torch.manual_seed(0)
    from_factory = factory_fn(**factory_kwargs)(*batches)
    torch.manual_seed(0)
    from_config = config_cls().build(*batches)

    _assert_same_state(from_factory, from_config)


@pytest.mark.parametrize(
    "model, net_config, sequence_condition",
    [
        ("mlp", MLPConfig(), False),
        ("ada_mlp", AdaMLPConfig(), False),
        ("transformer", TransformerConfig(), False),
        (
            "transformer_cross_attn",
            TransformerConfig(is_x_emb_seq=True),
            True,
        ),
    ],
)
def test_network_config_defaults_match_the_factory(
    model, net_config, sequence_condition, batches
):
    theta, condition = batches
    if sequence_condition:
        condition = torch.randn(32, 5, 3)

    torch.manual_seed(0)
    from_factory = posterior_flow_nn(model=model)(theta, condition)
    torch.manual_seed(0)
    from_config = FlowMatchingConfig(net=net_config).build(theta, condition)

    _assert_same_state(from_factory, from_config)


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
    "trainer_cls, config_cls",
    [(FMPE, FlowMatchingConfig), (NPSE, VEScoreConfig)],
)
def test_trainer_default_matches_the_config_default(trainer_cls, config_cls, batches):
    theta, x = batches
    prior = MultivariateNormal(zeros(2), torch.eye(2))
    trainer = trainer_cls(prior, show_progress_bars=False)
    trainer.append_simulations(theta, x)

    from_trainer = trainer._build_neural_net(theta, x)
    assert type(from_trainer) is type(config_cls().build(theta, x))
    assert isinstance(from_trainer.net, VectorFieldMLP)


def test_extra_kwargs_has_one_bucket_per_axis(batches):
    estimator = VEScoreConfig(extra_kwargs={"t_max": 0.9}).build(*batches)
    assert estimator.t_max == 0.9


@pytest.mark.parametrize("config_cls", ALL_CONFIGS + NET_CONFIGS)
def test_extra_kwargs_rejects_a_name_that_is_a_field(config_cls):
    name = dc_fields(config_cls)[0].name
    with pytest.raises(ValueError, match="Pass the"):
        config_cls(extra_kwargs={name: None})
