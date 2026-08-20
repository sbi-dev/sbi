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
    ConditionalFlowConfig,
)

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


def test_legacy_config_still_validates_the_mixed_string_path():
    """MNLE and MNPE still reach `build_mnle` / `build_mnpe` with flat kwargs."""
    cfg = ConditionalFlowConfig(hidden_features=64)
    assert cfg.to_dict() == {"hidden_features": 64}

    # The modeled variable carries the discrete column: theta for MNPE, x for MNLE.
    mixed = torch.cat(
        [torch.randn(100, 2), torch.randint(0, 3, (100, 1)).float()], dim=-1
    )
    assert isinstance(posterior_nn("mnpe")(mixed, X), MixedDensityEstimator)
    assert isinstance(likelihood_nn("mnle")(THETA, mixed), MixedDensityEstimator)


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
