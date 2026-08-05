# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import pytest
import torch
from torch import nn

from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.net_builders.estimator_configs import (
    _VALID_DENSITY_MODELS,
    DensityEstimatorBuilder,
    MixedDensityEstimatorBuilder,
    RatioEstimatorBuilder,
)


def test_density_estimator_builder_invalid_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        DensityEstimatorBuilder(model="invalid_model")


@pytest.mark.parametrize(
    "builder_cls,discriminator,model",
    [
        (DensityEstimatorBuilder, "model", "nsf"),
        (MixedDensityEstimatorBuilder, "continuous_model", "mdn"),
        (RatioEstimatorBuilder, "model", "mlp"),
    ],
)
def test_first_positional_argument_is_the_model(builder_cls, discriminator, model):
    """The first positional argument must bind to the model discriminator.

    `extra_kwargs` is declared on the shared base class, so without
    ``kw_only`` it would capture the model name and leave the discriminator
    at its default. That builds the wrong network without raising.
    """
    builder = builder_cls(model)
    assert getattr(builder, discriminator) == model
    assert builder.extra_kwargs == {}


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


@pytest.mark.parametrize("model", sorted(_VALID_DENSITY_MODELS))
def test_density_estimator_builder_build(model):
    """Test that build() returns correct estimator type for every supported model."""
    builder = DensityEstimatorBuilder(model=model)
    theta = torch.randn(100, 5)
    x = torch.randn(100, 3)
    estimator = builder.build(batch_input=theta, batch_condition=x)
    assert isinstance(estimator, ConditionalDensityEstimator)


@pytest.mark.parametrize("model", ["maf", "nsf", "zuko_maf", "zuko_nsf", "mdn"])
def test_density_estimator_builder_build_with_custom_features(model):
    """Test that custom hidden_features are forwarded and wired into the estimator."""
    custom_features = 32
    builder = DensityEstimatorBuilder(model=model, hidden_features=custom_features)
    theta = torch.randn(100, 4)
    x = torch.randn(100, 2)
    estimator = builder.build(batch_input=theta, batch_condition=x)
    assert isinstance(estimator, ConditionalDensityEstimator)
    # Verify that at least one layer has the requested hidden size.
    param_shapes = [p.shape for p in estimator.parameters()]
    assert any(custom_features in s for s in param_shapes), (
        f"No parameter with hidden_features={custom_features} found in {param_shapes}"
    )


@pytest.mark.parametrize("model", ["maf", "zuko_nsf"])
def test_density_estimator_builder_build_with_embedding_net(model):
    """Test that a custom embedding_net is wired into the built estimator."""
    emb = nn.Linear(3, 10)
    builder = DensityEstimatorBuilder(model=model, embedding_net=emb)
    theta = torch.randn(100, 5)
    x = torch.randn(100, 3)
    estimator = builder.build(batch_input=theta, batch_condition=x)
    assert isinstance(estimator, ConditionalDensityEstimator)
    # The supplied module itself must be installed in the estimator.
    assert any(module is emb for module in estimator.modules())


@pytest.mark.parametrize("model", ["maf", "zuko_maf"])
def test_density_estimator_builder_loss_computable(model):
    """Test that the built estimator can compute a finite loss."""
    builder = DensityEstimatorBuilder(model=model, hidden_features=16, num_transforms=2)
    theta = torch.randn(100, 3)
    x = torch.randn(100, 2)
    estimator = builder.build(batch_input=theta, batch_condition=x)

    # Evaluate on a fresh batch.
    batch_theta_eval = torch.randn(10, 3)
    batch_x_eval = torch.randn(10, 2)
    loss = estimator.loss(batch_theta_eval, condition=batch_x_eval)
    assert loss.shape == (10,)
    assert torch.isfinite(loss).all()


@pytest.mark.parametrize(
    "z_score_input, z_score_condition",
    [
        ("none", "none"),
        ("independent", "independent"),
        ("none", "independent"),
    ],
)
def test_density_estimator_builder_z_score_modes(z_score_input, z_score_condition):
    """Test that z_score fields are forwarded and the estimator builds successfully."""
    builder = DensityEstimatorBuilder(
        model="maf",
        z_score_input=z_score_input,
        z_score_condition=z_score_condition,
        num_transforms=2,
    )
    theta = torch.randn(100, 3)
    x = torch.randn(100, 2)
    estimator = builder.build(batch_input=theta, batch_condition=x)
    assert isinstance(estimator, ConditionalDensityEstimator)
    # When z_score is "none" for both, the estimator should still produce
    # a finite loss (just without standardization).
    loss = estimator.loss(theta[:10], condition=x[:10])
    assert torch.isfinite(loss).all()
