# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import pytest
import torch
from torch import nn

from sbi.neural_nets.build_context import BuildContext
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.net_builders.estimator_configs import (
    _VALID_DENSITY_MODELS,
    DensityEstimatorBuilder,
)


def test_density_estimator_builder_defaults():
    builder = DensityEstimatorBuilder()
    assert builder.model == "maf"
    assert builder.hidden_features is None
    assert builder.embedding_net is None


@pytest.mark.parametrize("model", sorted(_VALID_DENSITY_MODELS))
def test_density_estimator_builder_accepts_valid_models(model):
    builder = DensityEstimatorBuilder(model=model)
    assert builder.model == model


def test_density_estimator_builder_invalid_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        DensityEstimatorBuilder(model="invalid_model")


def test_density_estimator_builder_to_dict_only_set_fields():
    builder = DensityEstimatorBuilder(model="nsf", hidden_features=64)
    d = builder.to_dict()
    assert d == {"model": "nsf", "hidden_features": 64}
    assert "num_transforms" not in d


def test_density_estimator_builder_to_dict_with_extra_kwargs():
    with pytest.warns(UserWarning, match="Unknown kwargs"):
        builder = DensityEstimatorBuilder.from_kwargs(
            model="zuko_maf", hidden_features=64, some_library_param=True
        )
    d = builder.to_dict()
    assert d["some_library_param"] is True
    assert d["hidden_features"] == 64


def test_density_estimator_builder_build_kwargs_excludes_model():
    builder = DensityEstimatorBuilder(model="maf", hidden_features=64)
    kwargs = builder._build_kwargs()
    assert "model" not in kwargs
    assert kwargs["hidden_features"] == 64


_QUICK_MODELS = [
    "mdn",
    "made",
    "maf",
    "maf_rqs",
    "nsf",
    "zuko_nice",
    "zuko_maf",
    "zuko_nsf",
    "zuko_ncsf",
    "zuko_sospf",
    "zuko_naf",
    "zuko_unaf",
    "zuko_gf",
    "zuko_bpf",
]


@pytest.mark.parametrize("model", _QUICK_MODELS)
def test_density_estimator_builder_build(model):
    """Test that build() returns a ConditionalDensityEstimator."""
    builder = DensityEstimatorBuilder(model=model)
    theta = torch.randn(100, 5)
    x = torch.randn(100, 3)
    ctx = BuildContext.from_data(theta, x)
    estimator = builder.build(ctx, batch_theta=theta, batch_x=x)
    assert isinstance(estimator, ConditionalDensityEstimator)


@pytest.mark.parametrize("model", ["maf", "nsf", "zuko_maf", "zuko_nsf", "mdn"])
def test_density_estimator_builder_build_with_custom_features(model):
    """Test that custom hidden_features are forwarded to the build function."""
    builder = DensityEstimatorBuilder(model=model, hidden_features=32)
    theta = torch.randn(100, 4)
    x = torch.randn(100, 2)
    ctx = BuildContext.from_data(theta, x)
    estimator = builder.build(ctx, batch_theta=theta, batch_x=x)
    assert isinstance(estimator, ConditionalDensityEstimator)


@pytest.mark.parametrize("model", ["maf", "zuko_nsf"])
def test_density_estimator_builder_build_with_embedding_net(model):
    emb = nn.Linear(3, 10)
    builder = DensityEstimatorBuilder(model=model, embedding_net=emb)
    theta = torch.randn(100, 5)
    x = torch.randn(100, 3)
    ctx = BuildContext.from_data(theta, x)
    estimator = builder.build(ctx, batch_theta=theta, batch_x=x)
    assert isinstance(estimator, ConditionalDensityEstimator)


@pytest.mark.parametrize("model", ["maf", "zuko_maf"])
def test_density_estimator_builder_loss_computable(model):
    """Test that the built estimator can compute a finite loss."""
    builder = DensityEstimatorBuilder(model=model, hidden_features=16, num_transforms=2)
    theta = torch.randn(100, 3)
    x = torch.randn(100, 2)
    ctx = BuildContext.from_data(theta, x)
    estimator = builder.build(ctx, batch_theta=theta, batch_x=x)

    # Evaluate on a fresh batch.
    batch_theta_eval = torch.randn(10, 3)
    batch_x_eval = torch.randn(10, 2)
    loss = estimator.loss(batch_theta_eval, condition=batch_x_eval)
    assert loss.shape == (10,)
    assert torch.isfinite(loss).all()


def test_density_estimator_builder_z_score_none():
    """Test that z_score_x='none' disables z-scoring."""
    builder = DensityEstimatorBuilder(
        model="maf", z_score_x="none", z_score_y="none", num_transforms=2
    )
    theta = torch.randn(100, 3)
    x = torch.randn(100, 2)
    ctx = BuildContext.from_data(theta, x)
    estimator = builder.build(ctx, batch_theta=theta, batch_x=x)
    assert isinstance(estimator, ConditionalDensityEstimator)
