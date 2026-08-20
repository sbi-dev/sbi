# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Tests for how the factory functions construct per-model configs.

The factories span a whole family, so their named arguments cover settings a
given model may not have. A name the model does not know is dropped while it
still holds the factory's own default, and rejected once the caller has set it.
An unrecognised name keeps the old warn-and-forward behaviour, so that
library-specific parameters (e.g. Zuko flow kwargs) pass through.
"""

import inspect
import warnings
from dataclasses import fields as dc_fields

import pytest
import torch

from sbi.neural_nets.estimators import MixedDensityEstimator
from sbi.neural_nets.factory import (
    _CLASSIFIER_FACTORY_FIELDS,
    _LIKELIHOOD_FACTORY_FIELDS,
    _POSTERIOR_FACTORY_FIELDS,
    ZukoFlowType,
    classifier_nn,
    likelihood_nn,
    marginal_nn,
    posterior_flow_nn,
    posterior_nn,
    posterior_score_nn,
)
from sbi.neural_nets.net_builders.estimator_configs import (
    _CLASSIFIER_CONFIGS,
    _DENSITY_CONFIGS,
    MixedConfig,
    NSFConfig,
)
from sbi.neural_nets.net_builders.mixed_nets import build_mnle

THETA, X = torch.randn(100, 3), torch.randn(100, 5)


@pytest.mark.parametrize(
    "factory_fn,fields",
    [
        (posterior_nn, _POSTERIOR_FACTORY_FIELDS),
        (likelihood_nn, _LIKELIHOOD_FACTORY_FIELDS),
        (classifier_nn, _CLASSIFIER_FACTORY_FIELDS),
    ],
    ids=["posterior", "likelihood", "classifier"],
)
def test_factory_field_maps_name_real_parameters(factory_fn, fields):
    """The maps decide which defaults are read, so they must not drift."""
    params = inspect.signature(factory_fn).parameters
    assert set(fields.values()) <= set(params)


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


@pytest.mark.parametrize(
    "factory_fn,model,kwarg",
    [
        (posterior_nn, "mdn", {"num_bins": 20}),
        (posterior_nn, "mdn", {"num_transforms": 3}),
        (likelihood_nn, "made", {"num_components": 5}),
        (posterior_nn, "maf", {"num_bins": 20}),
        (classifier_nn, "linear", {"hidden_features": 64}),
        (posterior_nn, "mdn", {"num_blocks": 7}),
        (classifier_nn, "linear", {"num_blocks": 7}),
    ],
    ids=[
        "mdn-bins",
        "mdn-transforms",
        "made-components",
        "maf-bins",
        "linear-width",
        "mdn-other-model-field",
        "linear-other-model-field",
    ],
)
def test_factory_rejects_a_setting_the_model_does_not_use(factory_fn, model, kwarg):
    """A setting the model never reads used to be forwarded and dropped.

    Routing the factories through the per-model configs turns that silent
    ignore into an error, which is the point of the per-model design.
    """
    with pytest.raises(ValueError, match="would be silently ignored"):
        factory_fn(model, **kwarg)


@pytest.mark.parametrize(
    "factory_fn,model",
    [(fn, m) for fn in (posterior_nn, likelihood_nn) for m in sorted(_DENSITY_CONFIGS)]
    + [(classifier_nn, m) for m in sorted(_CLASSIFIER_CONFIGS)],
)
def test_factory_defaults_configure_every_model(factory_fn, model):
    """The family-wide defaults must not reject the models that ignore them.

    `posterior_nn("mdn")` passes `num_bins=10` that MDN has no field for, and
    `tabpfn` narrows `z_score_input` to `none`. Both are the factory's own
    defaults, so they are dropped rather than written over the config's.

    Construction is what fails, so `tabpfn` needs neither data nor its
    optional dependency here.
    """
    build_fn = factory_fn(model)
    assert build_fn is not None
    if model != "tabpfn":
        assert build_fn(THETA, X) is not None


def test_num_bins_still_reaches_the_zuko_flows():
    """The Zuko configs name it `bins`, so the factory name has to be mapped."""
    torch.manual_seed(0)
    default = posterior_nn("zuko_nsf")(THETA, X)
    torch.manual_seed(0)
    coarse = posterior_nn("zuko_nsf", num_bins=3)(THETA, X)

    assert sum(p.numel() for p in default.parameters()) != sum(
        p.numel() for p in coarse.parameters()
    )


@pytest.mark.parametrize(
    "factory_fn,input_dim,condition_dim",
    [(posterior_nn, 3, 5), (likelihood_nn, 5, 3)],
    ids=["posterior", "likelihood"],
)
def test_factories_keep_their_roles(factory_fn, input_dim, condition_dim):
    """`posterior_nn` models theta given x, `likelihood_nn` the other way."""
    estimator = factory_fn("maf")(THETA, X)

    assert estimator.input_shape == torch.Size([input_dim])
    assert estimator.condition_shape == torch.Size([condition_dim])


def test_posterior_nn_accepts_valid_extra_kwargs():
    """An unknown name is still forwarded, so library kwargs keep working."""
    with pytest.warns(UserWarning, match="Unknown kwargs"):
        build_fn = posterior_nn("zuko_maf", passes=2)
    with pytest.warns(UserWarning, match="Unknown `extra_kwargs`"):
        assert build_fn(THETA, X) is not None


def test_posterior_nn_accepts_nflows_dtype_without_warning():
    """The nflows base-distribution dtype is a supported transitive setting."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        estimator = posterior_nn("maf", dtype=torch.float64)(THETA, X)

    assert estimator.net._distribution._log_z.dtype == torch.float64


def test_mdn_snpe_a_rejects_num_components():
    """NPE-A owns the component count, so setting it on the factory is an error."""
    with pytest.raises(ValueError, match="num_components"):
        posterior_nn("mdn_snpe_a", num_components=20)


def test_mdn_snpe_a_takes_num_components_at_call_time():
    """NPE-A overrides the count per round, so it stays a call-time argument."""
    estimator = posterior_nn("mdn_snpe_a")(THETA, X, num_components=3)
    assert estimator.net._num_components == 3


def test_unknown_model_raises():
    """An unknown name still fails where it always has, at build time."""
    build_fn = posterior_nn("not_a_model")
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_fn(THETA, X)


@pytest.mark.parametrize(
    "factory_fn,model",
    [(posterior_nn, "mnpe"), (likelihood_nn, "mnle")],
    ids=["posterior", "likelihood"],
)
def test_mixed_string_path_builds_the_typed_config(factory_fn, model):
    """Deprecated mixed strings must route their settings through MixedConfig."""
    mixed = torch.cat(
        [torch.rand(100, 2), torch.randint(0, 3, (100, 1)).float()], dim=-1
    )
    config = MixedConfig(
        continuous=NSFConfig(
            z_score_input="none",
            hidden_features=16,
            num_transforms=2,
            num_bins=4,
            tail_bound=10.0,
        ),
        z_score_condition="none",
        log_transform_x=True,
    )
    factory_args = (mixed, X) if factory_fn is posterior_nn else (THETA, mixed)
    config_args = (mixed, X) if factory_fn is posterior_nn else (mixed, THETA)

    torch.manual_seed(0)
    expected = config.build(*config_args)
    torch.manual_seed(0)
    actual = factory_fn(
        model,
        z_score_theta="none",
        z_score_x="none",
        hidden_features=16,
        num_transforms=2,
        num_bins=4,
        log_transform_x=True,
    )(*factory_args)

    assert isinstance(actual, MixedDensityEstimator)
    assert expected.state_dict().keys() == actual.state_dict().keys()
    for name, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[name]), name


_NONE_IS_UNSET = [
    "flow_model",
    "hidden_features",
    "num_transforms",
    "num_bins",
    "continuous_hidden_features",
    "discrete_hidden_features",
    "combined_embedding_features",
    "num_categories_per_variable",
    "combined_embedding_net",
]


def _mixed_batches():
    mixed = torch.cat(
        [torch.rand(100, 2), torch.randint(0, 3, (100, 1)).float()], dim=-1
    )
    return mixed, THETA


def _assert_same_net(expected, actual):
    """Whether two built estimators carry the same parameters, name by name."""
    assert expected.state_dict().keys() == actual.state_dict().keys()
    for key, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key]), key


@pytest.mark.parametrize(
    "flow_model,name",
    [
        ("mdn", "num_blocks"),
        ("mdn", "tail_bound"),
        ("zuko_maf", "num_blocks"),
        ("nsf", "num_components"),
    ],
)
def test_mixed_treats_another_models_field_set_to_none_as_unset(flow_model, name):
    """The flat path dropped a None for any name the family knows."""
    mixed, theta = _mixed_batches()

    torch.manual_seed(0)
    omitted = build_mnle(mixed, theta, flow_model=flow_model)
    torch.manual_seed(0)
    explicit_none = build_mnle(mixed, theta, flow_model=flow_model, **{name: None})

    _assert_same_net(omitted, explicit_none)


@pytest.mark.parametrize(
    "flow_model,kwarg",
    [
        ("mdn", {"num_blocks": 3}),
        ("zuko_maf", {"num_blocks": 3}),
        ("nsf", {"num_components": 5}),
    ],
)
def test_mixed_still_rejects_another_models_field_with_a_value(flow_model, kwarg):
    """Only a None is unset. A real value the model cannot read still raises."""
    mixed, theta = _mixed_batches()

    with pytest.raises(ValueError, match="would be silently ignored"):
        build_mnle(mixed, theta, flow_model=flow_model, **kwarg)


@pytest.mark.parametrize("value", [64, None], ids=["value", "none"])
def test_mixed_still_warns_on_an_unknown_name(value):
    """Dropping a recognised None must not swallow a typo that carries one."""
    with pytest.warns(UserWarning, match="Unknown kwargs"):
        likelihood_nn("mnle", hiden_features=value)


@pytest.mark.parametrize("name", _NONE_IS_UNSET)
@pytest.mark.parametrize(
    "factory_fn,model",
    [(posterior_nn, "mnpe"), (likelihood_nn, "mnle")],
    ids=["posterior", "likelihood"],
)
def test_mixed_factory_treats_none_as_unset(factory_fn, model, name):
    """The flat path dropped every None, so None must still mean omitted."""
    mixed, theta = _mixed_batches()
    args = (mixed, X) if factory_fn is posterior_nn else (theta, mixed)

    torch.manual_seed(0)
    omitted = factory_fn(model)(*args)
    torch.manual_seed(0)
    explicit_none = factory_fn(model, **{name: None})(*args)

    _assert_same_net(omitted, explicit_none)


@pytest.mark.parametrize("name", _NONE_IS_UNSET)
def test_mixed_builder_treats_none_as_unset(name):
    """`build_mnle` reads the same flat arguments as the factories."""
    mixed, theta = _mixed_batches()

    torch.manual_seed(0)
    omitted = build_mnle(mixed, theta)
    torch.manual_seed(0)
    explicit_none = build_mnle(mixed, theta, **{name: None})

    _assert_same_net(omitted, explicit_none)


def test_mixed_none_width_does_not_change_the_other_width():
    """A None width must not make a sibling width fall back to a new source.

    With `continuous_hidden_features` set, the categorical net keeps falling
    back to `hidden_features`, whether the discrete width is omitted or None.
    """
    mixed, theta = _mixed_batches()

    torch.manual_seed(0)
    omitted = build_mnle(mixed, theta, continuous_hidden_features=16)
    torch.manual_seed(0)
    explicit_none = build_mnle(
        mixed, theta, continuous_hidden_features=16, discrete_hidden_features=None
    )

    _assert_same_net(omitted, explicit_none)


_TAIL_BOUND_MODELS = sorted(
    name
    for name, cls in _DENSITY_CONFIGS.items()
    if "tail_bound" in {f.name for f in dc_fields(cls)}
)


@pytest.mark.parametrize("flow_model", _TAIL_BOUND_MODELS)
def test_mixed_keeps_the_flat_tail_bound_for_every_model(flow_model):
    """The flat mixed API passed `tail_bound` to whichever model reads it.

    Those models default to a narrower bound on their own, and the value does
    not change the parameter count, so only reading it off the built net keeps
    the deprecated path honest.
    """
    mixed = torch.cat(
        [torch.rand(100, 2), torch.randint(0, 3, (100, 1)).float()], dim=-1
    )
    estimator = build_mnle(mixed, THETA, flow_model=flow_model)

    bounds = {
        float(m.tail_bound)
        for m in estimator.continuous_net.modules()
        if hasattr(m, "tail_bound")
    }
    assert bounds == {10.0}


@pytest.mark.parametrize(
    "factory_fn,z_kwargs",
    [
        (posterior_nn, {"z_score_theta": None, "z_score_x": None}),
        (likelihood_nn, {"z_score_theta": None, "z_score_x": None}),
    ],
    ids=["posterior", "likelihood"],
)
def test_none_means_no_z_scoring(factory_fn, z_kwargs):
    """The factories document `None` as "do not z-score", so it must not.

    The configs do not accept `None`, so it is translated in the factory. No
    test asserted the effect on the built network before.
    """
    explicit = {k: "none" for k in z_kwargs}
    from_none = factory_fn("nsf", **z_kwargs)(THETA, X)
    from_none_str = factory_fn("nsf", **explicit)(THETA, X)
    z_scored = factory_fn("nsf")(THETA, X)

    def standardizing(net):
        return [
            type(m).__name__
            for m in net.modules()
            if type(m).__name__ in ("Standardize", "PointwiseAffineTransform")
        ]

    assert standardizing(from_none) == []
    assert standardizing(from_none) == standardizing(from_none_str)
    assert standardizing(z_scored) != []
