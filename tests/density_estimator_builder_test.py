# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import inspect
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from sbi.neural_nets import likelihood_nn
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.net_builders.estimator_configs import (
    _CLASSIFIER_CONFIGS,
    _DENSITY_CONFIGS,
    ClassifierConfigBase,
    DensityConfigBase,
    LinearClassifierConfig,
    MAFConfig,
    MDNConfig,
    MLPClassifierConfig,
    MixedConfig,
    NSFConfig,
    PretrainedConfigBase,
    ResNetClassifierConfig,
    ZukoNCSFConfig,
    ZukoNSFConfig,
    _NFlowsFlowConfigBase,
    _UnconstrainedCapableConfigBase,
    _ZukoDensityConfigBase,
)
from sbi.neural_nets.net_builders.flow import build_zuko_flow, get_base_dist
from sbi.neural_nets.ratio_estimators import RatioEstimator
from sbi.utils import BoxUniform

DENSITY_MODELS = sorted(_DENSITY_CONFIGS)
# TabPFN needs the optional dependency and a fitted regressor, so it takes part
# in the introspection tests but not in the ones that build a network.
BUILDABLE_DENSITY_MODELS = [m for m in DENSITY_MODELS if m != "tabpfn"]
CLASSIFIER_MODELS = sorted(_CLASSIFIER_CONFIGS)


@pytest.fixture
def batches():
    return torch.randn(100, 3), torch.randn(100, 5)


def _params(fn) -> dict:
    return inspect.signature(fn).parameters


def _defaults_agree(value, default) -> bool:
    """Whether a field default matches a build function's, modules by type."""
    if isinstance(default, nn.Module):
        return type(value) is type(default)
    return value == default


@pytest.mark.parametrize("model", BUILDABLE_DENSITY_MODELS)
def test_density_config_builds(model, batches):
    """Every density config must build a working estimator with the right roles."""
    theta, x = batches
    estimator = _DENSITY_CONFIGS[model]().build(batch_input=theta, batch_condition=x)

    assert isinstance(estimator, ConditionalDensityEstimator)
    assert estimator.input_shape == theta[0].shape
    assert estimator.condition_shape == x[0].shape
    loss = estimator.loss(torch.randn(10, 3), condition=torch.randn(10, 5))
    assert loss.shape == (10,)
    assert torch.isfinite(loss).all()


@pytest.mark.parametrize("model", CLASSIFIER_MODELS)
def test_classifier_config_builds(model, batches):
    """Every classifier config must build a working ratio estimator."""
    theta, x = batches
    estimator = _CLASSIFIER_CONFIGS[model]().build(batch_input=theta, batch_condition=x)

    assert isinstance(estimator, RatioEstimator)
    log_ratios = estimator.unnormalized_log_ratio(theta[:10], x[:10])
    assert log_ratios.shape == (10,)
    assert torch.isfinite(log_ratios).all()


@pytest.mark.parametrize(
    "config_cls,kwargs",
    [
        (MAFConfig, {"num_bins": 20}),
        (NSFConfig, {"randperm": True}),
        (MDNConfig, {"num_transforms": 3}),
        (ZukoNSFConfig, {"tails": "linear"}),
        (ZukoNSFConfig, {"num_blocks": 3}),
        (LinearClassifierConfig, {"hidden_features": 64}),
        (MLPClassifierConfig, {"num_blocks": 3}),
        (ResNetClassifierConfig, {"norm_layer": nn.LayerNorm}),
    ],
    ids=[
        "maf-num_bins",
        "nsf-randperm",
        "mdn-num_transforms",
        "zuko_nsf-tails",
        "zuko_nsf-num_blocks",
        "linear-hidden_features",
        "mlp-num_blocks",
        "resnet-norm_layer",
    ],
)
def test_setting_a_model_does_not_use_raises(config_cls, kwargs):
    """A setting a model does not use must not be constructible.

    This is what the per-model configs buy over one flat builder: the setting is
    not a field, so Python rejects it instead of the value being dropped on the
    way to the network.
    """
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        config_cls(**kwargs)


@pytest.mark.parametrize("model", DENSITY_MODELS)
def test_no_density_field_is_dropped_before_the_network(model):
    """Every field must be named by a build function on the way down.

    The Zuko wrappers forward what they do not name to `build_zuko_flow`, so a
    field is safe if either signature has it. Anything else ends up in a
    `**kwargs` that drops it.
    """
    config_cls = _DENSITY_CONFIGS[model]
    accepted = set(_params(config_cls._BUILD_FN))
    if issubclass(config_cls, _ZukoDensityConfigBase):
        accepted |= set(_params(build_zuko_flow))
    if issubclass(config_cls, _NFlowsFlowConfigBase):
        accepted |= set(_params(get_base_dist))

    for name in config_cls()._build_kwargs():
        assert name in accepted, f"`{name}` is not accepted by {model}"


@pytest.mark.parametrize("model", DENSITY_MODELS)
def test_density_defaults_match_the_build_functions(model):
    """Each config must default to what its build function uses.

    `_build_kwargs()` is the dict that actually reaches the build function, so
    checking it also covers the `z_score_input` / `z_score_condition` renames.
    """
    config_cls = _DENSITY_CONFIGS[model]
    params = dict(_params(config_cls._BUILD_FN))
    if issubclass(config_cls, _ZukoDensityConfigBase):
        # The per-model wrapper wins where both name a setting: it is the one
        # the config calls.
        params = {**_params(build_zuko_flow), **params}
    if issubclass(config_cls, _NFlowsFlowConfigBase):
        params = {**_params(get_base_dist), **params}

    for name, value in config_cls()._build_kwargs().items():
        assert _defaults_agree(value, params[name].default), f"{model}.{name} drifted"


@pytest.mark.parametrize("model", CLASSIFIER_MODELS)
def test_classifier_defaults_match_the_build_functions(model):
    """Each classifier config must default to what its build function uses."""
    config_cls = _CLASSIFIER_CONFIGS[model]
    params = _params(config_cls._BUILD_FN)

    for name, value in config_cls()._build_kwargs().items():
        assert name in params, f"`{name}` is not accepted by {model}"
        assert _defaults_agree(value, params[name].default), f"{model}.{name} drifted"


def test_mixed_defaults_match_the_factory():
    """Mixed is the family where a default drift shipped once.

    The builder path gave `num_transforms=2` and `num_bins=5` against the
    factory's 5 and 10, so compare the two complete networks.
    """
    theta = torch.randn(100, 3)
    mixed_x = torch.cat(
        [torch.randn(100, 2), torch.randint(0, 3, (100, 1)).float()], dim=-1
    )

    config = MixedConfig()
    assert isinstance(config.continuous, NSFConfig)
    torch.manual_seed(0)
    configured = config.build(mixed_x, theta)
    torch.manual_seed(0)
    from_factory = likelihood_nn("mnle")(theta, mixed_x)

    assert configured.state_dict().keys() == from_factory.state_dict().keys()
    for name, value in configured.state_dict().items():
        assert torch.equal(value, from_factory.state_dict()[name]), name

    assert config.continuous.tail_bound == 10.0
    assert config.dropout_probability == 0.0


def test_every_classifier_model_has_a_config():
    from sbi.neural_nets.net_builders.classifier import (
        build_linear_classifier,
        build_mlp_classifier,
        build_resnet_classifier,
    )

    assert {c._BUILD_FN for c in _CLASSIFIER_CONFIGS.values()} == {
        build_linear_classifier,
        build_mlp_classifier,
        build_resnet_classifier,
    }


@pytest.mark.parametrize(
    "config_cls,expected",
    [(MAFConfig, {"z_score_x", "z_score_y"}), (NSFConfig, {"z_score_x", "z_score_y"})],
    ids=["maf", "nsf"],
)
def test_z_score_names_are_translated(config_cls, expected):
    """The user-facing names must reach the build functions as x/y."""
    kwargs = config_cls(
        z_score_input="none", z_score_condition="structured"
    )._build_kwargs()

    assert expected <= set(kwargs)
    assert kwargs["z_score_x"] == "none"
    assert kwargs["z_score_y"] == "structured"
    assert not {"z_score_input", "z_score_condition"} & set(kwargs)


def test_classifier_embedding_nets_keep_their_user_facing_meaning(batches):
    """`embedding_net_theta` embeds theta, `embedding_net_x` embeds the data.

    The build functions use positional x/y naming, in which their
    `embedding_net_x` applies to theta, so the config has to swap them.
    """
    theta, x = batches
    estimator = ResNetClassifierConfig(
        embedding_net_theta=nn.Linear(3, 7),
        embedding_net_x=nn.Linear(5, 4),
    ).build(batch_input=theta, batch_condition=x)

    assert estimator.embedding_net_theta[-1].out_features == 7
    assert estimator.embedding_net_x[-1].out_features == 4


@pytest.mark.parametrize("model", BUILDABLE_DENSITY_MODELS)
def test_custom_hidden_features_reach_the_network(model, batches):
    """A changed width must produce a different network."""
    theta, x = batches
    config_cls = _DENSITY_CONFIGS[model]

    torch.manual_seed(0)
    default = config_cls().build(batch_input=theta, batch_condition=x)
    torch.manual_seed(0)
    wider = config_cls(hidden_features=80).build(batch_input=theta, batch_condition=x)

    assert sum(p.numel() for p in default.parameters()) != sum(
        p.numel() for p in wider.parameters()
    )


@pytest.mark.parametrize("model", DENSITY_MODELS)
def test_only_the_right_models_offer_the_unconstrained_transform(model):
    """`transform_to_unconstrained` must be offered exactly where it works.

    It derives the reparametrization from a distribution's support instead of
    from batch statistics, and only the Zuko flows and the MDN implement it. On
    the others `z_score_parser` would silently make it a no-op.
    """
    config_cls = _DENSITY_CONFIGS[model]
    implements_it = issubclass(config_cls, _UnconstrainedCapableConfigBase)

    if implements_it:
        config_cls(
            z_score_input="transform_to_unconstrained",
            x_dist=BoxUniform(torch.zeros(3), torch.ones(3)),
        )
    else:
        with pytest.raises(ValueError, match="Invalid value"):
            config_cls(z_score_input="transform_to_unconstrained")


@pytest.mark.parametrize("config_cls", [ZukoNSFConfig, MDNConfig], ids=["zuko", "mdn"])
def test_unconstrained_transform_builds(config_cls, batches):
    """The unconstrained transform must survive an actual build."""
    theta, x = batches
    estimator = config_cls(
        z_score_input="transform_to_unconstrained",
        x_dist=BoxUniform(-3 * torch.ones(3), 3 * torch.ones(3)),
    ).build(batch_input=theta, batch_condition=x)

    assert isinstance(estimator, ConditionalDensityEstimator)


def test_x_dist_requires_the_unconstrained_transform():
    """`x_dist` is read only by that mode, so setting it alone is a mistake."""
    with pytest.raises(ValueError, match="x_dist"):
        ZukoNSFConfig(x_dist=BoxUniform(torch.zeros(2), torch.ones(2)))


def test_condition_side_never_offers_the_unconstrained_transform():
    """The condition side never applies it, so accepting it would be a no-op."""
    with pytest.raises(ValueError, match="Invalid value.*z_score_condition"):
        ZukoNSFConfig(z_score_condition="transform_to_unconstrained")


@pytest.mark.parametrize(
    "config_cls", [MAFConfig, ZukoNSFConfig, ResNetClassifierConfig]
)
def test_invalid_z_score_value_raises(config_cls):
    """`Literal` values are not checked by Python, so the config must check them."""
    with pytest.raises(ValueError, match="Invalid value.*z_score_input"):
        config_cls(z_score_input="typo")


@pytest.mark.parametrize("config_cls", [MAFConfig, ResNetClassifierConfig])
def test_configs_are_immutable(config_cls):
    with pytest.raises(FrozenInstanceError):
        config_cls().hidden_features = 42


@pytest.mark.parametrize(
    "base,example",
    [
        (DensityConfigBase, "NSFConfig"),
        (PretrainedConfigBase, "TabPFNConfig"),
        (ClassifierConfigBase, "ResNetClassifierConfig"),
    ],
    ids=["density", "pretrained", "classifier"],
)
def test_role_bases_are_not_usable_on_their_own(base, example):
    """A role base holds no model, so it must not pass as a config."""
    with pytest.raises(TypeError, match=f"per-model config.*{example}"):
        base()


def test_repr_shows_only_the_settings_that_differ_from_the_defaults():
    assert repr(MAFConfig()) == "MAFConfig()"
    assert repr(MAFConfig(hidden_features=64)) == "MAFConfig(hidden_features=64)"
    # An untouched embedding net does not compare equal to a fresh one, so the
    # repr has to recognise it by type.
    assert "embedding_net" not in repr(NSFConfig())
    assert "embedding_net" in repr(NSFConfig(embedding_net=nn.Linear(3, 3)))
    assert repr(MixedConfig()) == "MixedConfig()"


def test_extra_kwargs_reaches_the_build_function(batches):
    """`extra_kwargs` must forward settings that have no field of their own."""
    theta, x = batches

    torch.manual_seed(0)
    default = ZukoNSFConfig().build(batch_input=theta, batch_condition=x)
    torch.manual_seed(0)
    with pytest.warns(UserWarning, match="Unknown `extra_kwargs`"):
        tweaked = ZukoNSFConfig(extra_kwargs={"passes": 2}).build(
            batch_input=theta, batch_condition=x
        )

    left, right = default.state_dict(), tweaked.state_dict()
    assert any(not torch.equal(left[k], right[k]) for k in left)


def test_extra_kwargs_typo_is_not_silent(batches):
    theta, x = batches
    with (
        pytest.warns(UserWarning, match="Unknown `extra_kwargs`"),
        pytest.raises(TypeError, match="unexpected keyword argument"),
    ):
        ZukoNSFConfig(extra_kwargs={"binz": 20}).build(
            batch_input=theta, batch_condition=x
        )


@pytest.mark.parametrize(
    "config",
    [
        MDNConfig(extra_kwargs={"num_component": 1}),
        MDNConfig(extra_kwargs={"kwargs": 1}),
        LinearClassifierConfig(extra_kwargs={"hidden_feature": 2}),
    ],
    ids=["density", "var-keyword-name", "classifier"],
)
def test_swallowed_extra_kwargs_warn_at_build_time(config, batches):
    """Builders with permissive **kwargs must still expose likely typos."""
    theta, x = batches
    with pytest.warns(UserWarning, match="Unknown `extra_kwargs`"):
        config.build(batch_input=theta, batch_condition=x)


def test_extra_kwargs_cannot_shadow_a_field():
    """Fields set through `extra_kwargs` would bypass the config's validation."""
    with pytest.raises(ValueError, match="are fields of"):
        ZukoNSFConfig(extra_kwargs={"num_bins": 20})


@pytest.mark.parametrize(
    "config_cls,extra_kwargs",
    [
        (MAFConfig, {"z_score_x": "none"}),
        (ResNetClassifierConfig, {"embedding_net_y": nn.Identity()}),
        (ZukoNCSFConfig, {"bins": 3}),
        (ZukoNSFConfig, {"bins": 3}),
    ],
    ids=[
        "z-score-alias",
        "classifier-embedding-alias",
        "zuko-ncsf-alias",
        "zuko-nsf-alias",
    ],
)
def test_extra_kwargs_cannot_shadow_a_downstream_alias(config_cls, extra_kwargs):
    """Aliases must not bypass validation of their user-facing fields."""
    with pytest.raises(ValueError, match="fields|duplicates"):
        config_cls(extra_kwargs=extra_kwargs)


def test_nflows_dtype_reaches_the_base_distribution(batches):
    theta, x = batches
    estimator = MAFConfig(dtype=torch.float64).build(
        batch_input=theta, batch_condition=x
    )

    assert estimator.net._distribution._log_z.dtype == torch.float64
