# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import inspect
import warnings
from dataclasses import FrozenInstanceError, fields
from typing import get_args

import pytest
import torch
import zuko
from torch import Tensor

from sbi.inference import MarginalTrainer
from sbi.neural_nets.estimators import UnconditionalDensityEstimator
from sbi.neural_nets.factory import ZukoFlowType
from sbi.neural_nets.net_builders.estimator_configs import (
    MARGINAL_MODELS,
    _BUILD_KWARG_ALIASES,
    _MARGINAL_CONFIGS,
    DensityEstimatorBuilder,
    MarginalBPFConfig,
    MarginalConfigBase,
    MarginalGFConfig,
    MarginalMAFConfig,
    MarginalNAFConfig,
    MarginalNCSFConfig,
    MarginalNICEConfig,
    MarginalNSFConfig,
    MarginalSOSPFConfig,
)
from sbi.neural_nets.net_builders.flow import (
    build_zuko_bpf,
    build_zuko_flow,
    build_zuko_gf,
    build_zuko_maf,
    build_zuko_naf,
    build_zuko_ncsf,
    build_zuko_nice,
    build_zuko_nsf,
    build_zuko_sospf,
    build_zuko_unaf,
    build_zuko_unconditional_flow,
    nflow_specific_kwargs,
)

# The conditional build functions carry the defaults sbi has chosen for each
# Zuko flow. The marginal configs must reproduce them, under Zuko's own
# parameter names (`num_bins` is sbi's name for Zuko's `bins`).
_CONDITIONAL_BUILD_FNS = {
    "bpf": build_zuko_bpf,
    "gf": build_zuko_gf,
    "maf": build_zuko_maf,
    "naf": build_zuko_naf,
    "ncsf": build_zuko_ncsf,
    "nice": build_zuko_nice,
    "nsf": build_zuko_nsf,
    "sospf": build_zuko_sospf,
    "unaf": build_zuko_unaf,
}

_ZUKO_NAMES = {"bins": "num_bins"}

_SHARED_FIELDS = {f.name for f in fields(MarginalConfigBase)}

# The same fields under the names `_build_kwargs()` produces, so that the two
# hops of the build can be told apart by the key alone.
_SHARED_BUILD_KWARGS = {
    _BUILD_KWARG_ALIASES.get(name, name)
    for name in _SHARED_FIELDS
    if name != "extra_kwargs"
}

MODELS = sorted(_MARGINAL_CONFIGS)


def _model_specific_fields(config_cls) -> list:
    """Return the fields a config adds on top of the shared marginal ones."""
    return [f for f in fields(config_cls) if f.name not in _SHARED_FIELDS]


def _differs(first: UnconditionalDensityEstimator, second) -> bool:
    """Whether two estimators differ in parameter shapes or values."""
    left, right = first.state_dict(), second.state_dict()
    if left.keys() != right.keys():
        return True
    return any(
        left[k].shape != right[k].shape or not torch.equal(left[k], right[k])
        for k in left
    )


def _build_seeded(config: MarginalConfigBase, batch_x: Tensor):
    """Build from a fixed seed, so two builds differ only through their config."""
    torch.manual_seed(0)
    return config.build(batch_x)


@pytest.fixture
def batch_x() -> Tensor:
    return torch.randn(100, 3)


def _samples_in_domain_of(model: str) -> Tensor:
    """Return samples from the domain the model is defined on."""
    if model == "ncsf":
        return torch.pi * (2 * torch.rand(100, 3) - 1)
    return torch.randn(100, 3)


@pytest.mark.parametrize(
    "config_cls,kwargs",
    [
        (MarginalMAFConfig, {"bins": 20}),
        (MarginalNSFConfig, {"randmask": True}),
        (MarginalNICEConfig, {"signal": 8}),
        (MarginalGFConfig, {"degree": 4}),
        (MarginalBPFConfig, {"polynomials": 3}),
        (MarginalNAFConfig, {"components": 4}),
        (MarginalNCSFConfig, {"randmask": True}),
    ],
    ids=[
        "maf-bins",
        "nsf-randmask",
        "nice-signal",
        "gf-degree",
        "bpf-poly",
        "naf-components",
        "ncsf-randmask",
    ],
)
def test_setting_a_model_does_not_use_raises(config_cls, kwargs):
    """A setting a model does not expose must not be constructible.

    This is what the per-model configs buy over one flat builder: the setting
    is not a field, so Python rejects it instead of the value being dropped on
    the way to the flow.
    """
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        config_cls(**kwargs)


@pytest.mark.parametrize("model", MODELS)
def test_no_field_is_dropped_before_the_flow(model):
    """Every field must be a parameter of what consumes it.

    `build_zuko_unconditional_flow` consumes the shared fields itself, drops the
    names in `nflow_specific_kwargs`, and forwards the rest to the Zuko flow
    class. A field that survives neither hop is silently ignored, which is what
    `num_bins` used to be.
    """
    config_cls = _MARGINAL_CONFIGS[model]
    build_params = inspect.signature(build_zuko_unconditional_flow).parameters
    zuko_params = inspect.signature(
        getattr(zuko.flows, config_cls._WHICH_NF).__init__
    ).parameters

    for name in config_cls()._build_kwargs():
        assert name not in nflow_specific_kwargs, (
            f"`{name}` is filtered out before the flow is constructed"
        )
        expected = build_params if name in _SHARED_BUILD_KWARGS else zuko_params
        assert name in expected, f"`{name}` is not accepted by {model}"


@pytest.mark.parametrize("model", MODELS)
def test_defaults_match_the_conditional_build_functions(model):
    """Marginal defaults must equal the conditional ones for the same flow.

    Marginal flows have no build function of their own to read defaults from,
    so the conditional counterpart is the single source of truth.
    """
    build_params = inspect.signature(_CONDITIONAL_BUILD_FNS[model]).parameters

    for f in _model_specific_fields(_MARGINAL_CONFIGS[model]):
        param = build_params[_ZUKO_NAMES.get(f.name, f.name)]
        assert f.default == param.default, f"{model}.{f.name} drifted"


def test_shared_defaults_match_the_build_functions():
    """The shared fields must default to what the build functions use."""
    unconditional = inspect.signature(build_zuko_unconditional_flow).parameters
    conditional = inspect.signature(build_zuko_flow).parameters
    defaults = {f.name: f.default for f in fields(MarginalConfigBase)}

    assert defaults["hidden_features"] == unconditional["hidden_features"].default
    assert defaults["num_transforms"] == unconditional["num_transforms"].default
    # `build_zuko_unconditional_flow` requires z_score_x, so the conditional
    # builder is the reference for its default.
    assert defaults["z_score_input"] == conditional["z_score_x"].default


@pytest.mark.parametrize(
    "model,field_name",
    [
        (model, f.name)
        for model in MODELS
        for f in _model_specific_fields(_MARGINAL_CONFIGS[model])
    ],
)
def test_changing_a_field_changes_the_network(model, field_name, batch_x):
    """Every model-specific setting must reach the flow it configures.

    Two builds from the same seed differ only through the config, so an
    unchanged network means the setting never arrived.
    """
    config_cls = _MARGINAL_CONFIGS[model]
    default = config_cls.__dataclass_fields__[field_name].default
    other = (not default) if isinstance(default, bool) else default + 2

    assert _differs(
        _build_seeded(config_cls(), batch_x),
        _build_seeded(config_cls(**{field_name: other}), batch_x),
    )


def test_marginal_nsf_defaults_to_ten_bins(batch_x):
    """The marginal NSF must use 10 bins, like the conditional one.

    `build_zuko_unconditional_flow` drops `num_bins`, so marginal NSFs used to
    fall back to Zuko's default of 8 bins whatever the user asked for.
    """
    assert MarginalNSFConfig().bins == 10

    torch.manual_seed(0)
    zuko_default = build_zuko_unconditional_flow("NSF", batch_x, "independent")
    torch.manual_seed(0)
    ten_bins = build_zuko_unconditional_flow("NSF", batch_x, "independent", bins=10)

    assert _differs(_build_seeded(MarginalNSFConfig(), batch_x), zuko_default)
    assert not _differs(_build_seeded(MarginalNSFConfig(), batch_x), ten_bins)


@pytest.mark.parametrize("model", MODELS)
def test_build_returns_a_usable_estimator(model):
    """Every model must build an estimator that models the given samples."""
    batch_x = _samples_in_domain_of(model)
    estimator = _MARGINAL_CONFIGS[model]().build(batch_x)
    log_prob = estimator.log_prob(batch_x)
    assert isinstance(estimator, UnconditionalDensityEstimator)
    assert estimator.input_shape == batch_x[0].shape
    assert log_prob.shape == (batch_x.shape[0],)
    assert torch.isfinite(log_prob).all()
    assert estimator.sample(torch.Size((5,))).shape == (5, *batch_x[0].shape)


@pytest.mark.parametrize("value", ["typo", "transform_to_unconstrained"])
def test_invalid_z_score_value_raises(value):
    """`Literal` values are not checked by Python, so the config must check them.

    `transform_to_unconstrained` is a valid mode for conditional flows but not
    for unconditional ones, so it must be rejected here.
    """
    with pytest.raises(ValueError, match="Invalid value"):
        MarginalNSFConfig(z_score_input=value)


def test_configs_are_immutable():
    config = MarginalNSFConfig()
    with pytest.raises(FrozenInstanceError):
        config.bins = 20


def test_base_config_is_not_usable_on_its_own():
    """The base holds no model, so it must not pass as a config."""
    with pytest.raises(TypeError, match="per-model config"):
        MarginalConfigBase()


def test_repr_shows_only_the_settings_that_differ_from_the_defaults():
    assert repr(MarginalNSFConfig()) == "MarginalNSFConfig()"
    assert (
        repr(MarginalNSFConfig(bins=20, num_transforms=3))
        == "MarginalNSFConfig(num_transforms=3, bins=20)"
    )
    assert (
        repr(MarginalMAFConfig(extra_kwargs={"passes": 2}))
        == "MarginalMAFConfig(extra_kwargs={'passes': 2})"
    )


def test_extra_kwargs_reaches_the_flow(batch_x):
    """`extra_kwargs` must forward settings that have no field of their own."""
    assert _differs(
        _build_seeded(MarginalMAFConfig(), batch_x),
        _build_seeded(MarginalMAFConfig(extra_kwargs={"passes": 2}), batch_x),
    )


def test_extra_kwargs_typo_is_not_silent(batch_x):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        MarginalNSFConfig(extra_kwargs={"binz": 20}).build(batch_x)


@pytest.mark.parametrize(
    "kwargs",
    [{"hidden_features": 200}, {"extra_kwargs": {"activation": torch.nn.ELU}}],
    ids=["hidden_features", "extra_kwargs"],
)
def test_gf_rejects_the_settings_it_would_drop(kwargs):
    """GF takes no network without a condition, so these never arrive.

    Zuko forwards them to an element-wise transform that builds nothing when
    `context=0`, which a marginal flow always has. They are not filtered by
    `nflow_specific_kwargs` and Zuko raises nothing, so only the config can
    catch them.
    """
    with pytest.raises(ValueError, match="GF does not use"):
        MarginalGFConfig(**kwargs)


def test_gf_still_accepts_the_settings_that_work(batch_x):
    """The guard must not block the two knobs GF does read."""
    assert _differs(
        _build_seeded(MarginalGFConfig(), batch_x),
        _build_seeded(MarginalGFConfig(num_transforms=8, components=16), batch_x),
    )


def test_extra_kwargs_cannot_shadow_a_field():
    """Fields set through `extra_kwargs` would bypass the config's validation."""
    with pytest.raises(ValueError, match="are fields of"):
        MarginalNSFConfig(extra_kwargs={"bins": 20})


@pytest.mark.parametrize("key", sorted(nflow_specific_kwargs))
def test_extra_kwargs_rejects_the_names_the_flow_never_sees(key):
    """build_zuko_unconditional_flow drops these before building the flow."""
    with pytest.raises(ValueError, match="never reach the flow"):
        MarginalNSFConfig(extra_kwargs={key: 20})


def test_every_model_has_a_config():
    """The registry, the Literal and the enum must all name the same models.

    `MARGINAL_MODELS` and `_MARGINAL_CONFIGS` are written out separately, so
    nothing but this stops them drifting apart.
    """
    assert set(get_args(MARGINAL_MODELS)) == set(_MARGINAL_CONFIGS)
    assert set(_MARGINAL_CONFIGS) == {flow.value for flow in ZukoFlowType}

    for model, config_cls in _MARGINAL_CONFIGS.items():
        assert model.upper() == config_cls._WHICH_NF
        assert hasattr(zuko.flows, config_cls._WHICH_NF)


@pytest.mark.parametrize(
    "density_estimator",
    [None, MarginalNSFConfig(), lambda x: None],
    ids=["default", "config", "callable"],
)
def test_no_warning_for_valid_inputs(density_estimator):
    """None, a config, and a callable are the supported ways in."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        MarginalTrainer(density_estimator=density_estimator)


@pytest.mark.parametrize(
    "density_estimator", ["nsf", ZukoFlowType.NSF], ids=["string", "enum"]
)
def test_deprecated_model_arguments_warn(density_estimator):
    with pytest.warns(FutureWarning, match="from sbi.neural_nets import"):
        MarginalTrainer(density_estimator=density_estimator)


@pytest.mark.parametrize("model", MODELS)
def test_deprecated_string_builds_the_default_config(model, batch_x):
    """The deprecated path must resolve to the config for the same model."""
    with pytest.warns(FutureWarning):
        trainer = MarginalTrainer(density_estimator=model)

    torch.manual_seed(0)
    from_string = trainer._build_neural_net(batch_x)

    assert not _differs(from_string, _build_seeded(_MARGINAL_CONFIGS[model](), batch_x))


def test_unknown_model_string_raises():
    with (
        pytest.raises(ValueError, match="Unknown marginal model"),
        pytest.warns(FutureWarning),
    ):
        MarginalTrainer(density_estimator="not_a_flow")


def test_conditional_builder_raises():
    """A builder for a conditional estimator cannot build a marginal one."""
    with pytest.raises(TypeError, match="marginal config"):
        MarginalTrainer(density_estimator=DensityEstimatorBuilder(model="maf"))


def test_config_class_instead_of_instance_raises():
    """A forgotten `()` would otherwise pass the callable branch."""
    with pytest.raises(TypeError, match="not an instance"):
        MarginalTrainer(density_estimator=MarginalNSFConfig)


def test_non_callable_density_estimator_raises():
    with pytest.raises(ValueError, match="marginal config"):
        MarginalTrainer(density_estimator=42)


def test_built_module_is_rejected():
    """A module is callable, so only `check_estimator_arg` catches it.

    The trainer needs a function that builds the estimator from a batch, not
    the estimator itself.
    """
    with pytest.raises(TypeError, match="function returning a nn.Module"):
        MarginalTrainer(density_estimator=torch.nn.Linear(3, 3))


def test_default_estimator_is_the_marginal_nsf(batch_x):
    """`MarginalTrainer()` must keep building an NSF, as it did before."""
    torch.manual_seed(0)
    default = MarginalTrainer()._build_neural_net(batch_x)

    assert not _differs(default, _build_seeded(MarginalNSFConfig(), batch_x))


@pytest.mark.filterwarnings("ignore:Maximum number of epochs")
@pytest.mark.parametrize(
    "config",
    [MarginalNSFConfig(hidden_features=16, num_transforms=2), MarginalSOSPFConfig()],
    ids=["nsf", "sospf"],
)
def test_train_with_config(config):
    """End-to-end: a config trains and the trained estimator samples."""
    x = torch.randn(200, 2)

    trainer = MarginalTrainer(density_estimator=config, show_progress_bars=False)
    trainer.append_samples(x)
    estimator = trainer.train(max_num_epochs=2, stop_after_epochs=1)

    assert isinstance(estimator, UnconditionalDensityEstimator)
    assert estimator.sample(torch.Size((10,))).shape == (10, 2)


def test_callable_density_estimator_is_used_as_is(batch_x):
    """A user-supplied build function must be called with the sample batch."""
    seen = {}

    def build_fn(x: Tensor) -> UnconditionalDensityEstimator:
        seen["batch_x"] = x
        return MarginalNSFConfig().build(x)

    trainer = MarginalTrainer(density_estimator=build_fn)
    trainer._build_neural_net(batch_x)

    assert seen["batch_x"] is batch_x
