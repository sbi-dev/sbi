# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings

import pytest
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import BNRE, NRE_A, NRE_B, NRE_C
from sbi.neural_nets import classifier_nn
from sbi.neural_nets.net_builders.estimator_configs import (
    DensityEstimatorBuilder,
    RatioEstimatorBuilder,
)
from sbi.neural_nets.ratio_estimators import RatioEstimator
from sbi.utils.user_input_checks import check_estimator_arg

_NRE_TRAINERS = [NRE_A, NRE_B, NRE_C, BNRE]


@pytest.mark.parametrize(
    "trainer_cls", _NRE_TRAINERS, ids=["nre_a", "nre_b", "nre_c", "bnre"]
)
def test_no_warning_for_valid_inputs(trainer_cls):
    """None default, builder, and callable should not emit FutureWarning."""
    prior = MultivariateNormal(zeros(2), eye(2))
    builder = RatioEstimatorBuilder(model="resnet")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        trainer_cls(prior, show_progress_bars=False)
        trainer_cls(prior, classifier=builder, show_progress_bars=False)
        trainer_cls(
            prior,
            classifier=classifier_nn(model="resnet"),
            show_progress_bars=False,
        )


@pytest.mark.parametrize(
    "trainer_cls", _NRE_TRAINERS, ids=["nre_a", "nre_b", "nre_c", "bnre"]
)
def test_string_emits_deprecation_warning(trainer_cls):
    """Passing a string to classifier should emit FutureWarning."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.warns(FutureWarning, match="deprecated"):
        trainer_cls(prior, classifier="resnet", show_progress_bars=False)


@pytest.mark.parametrize(
    "trainer_cls", _NRE_TRAINERS, ids=["nre_a", "nre_b", "nre_c", "bnre"]
)
def test_wrong_builder_type_raises(trainer_cls):
    """Passing a DensityEstimatorBuilder should raise TypeError early."""
    prior = MultivariateNormal(zeros(2), eye(2))
    wrong_builder = DensityEstimatorBuilder(model="maf")
    with pytest.raises(TypeError, match="RatioEstimatorBuilder"):
        trainer_cls(prior, classifier=wrong_builder, show_progress_bars=False)


def test_builder_invalid_model():
    """Invalid model should raise ValueError at construction."""
    with pytest.raises(ValueError, match="Unknown model"):
        RatioEstimatorBuilder(model="invalid")


@pytest.mark.parametrize(
    "trainer_cls",
    [NRE_A, NRE_B],
    ids=["nre_a", "nre_b"],
)
@pytest.mark.parametrize("model", ("resnet", "mlp", "linear"))
def test_train_with_builder(trainer_cls, model):
    """Train with a RatioEstimatorBuilder end-to-end."""
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    builder = RatioEstimatorBuilder(model=model, hidden_features=16)
    inference = trainer_cls(prior, classifier=builder, show_progress_bars=False)

    theta = prior.sample((200,))
    x = theta + 0.1 * torch.randn_like(theta)
    ratio_estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=2, training_batch_size=100
    )
    assert isinstance(ratio_estimator, RatioEstimator)

    # Verify the trained estimator produces finite log-ratios on fresh data.
    fresh_theta = prior.sample((10,))
    fresh_x = fresh_theta + 0.1 * torch.randn_like(fresh_theta)
    log_ratios = ratio_estimator.unnormalized_log_ratio(fresh_theta, fresh_x)
    assert log_ratios.shape == (10,)
    assert torch.isfinite(log_ratios).all()

    # Posterior should be constructable and produce correct-shaped samples.
    posterior = inference.build_posterior()
    x_o = zeros(1, num_dim)
    samples = posterior.sample((10,), x=x_o)
    assert samples.shape == (10, num_dim)


@pytest.mark.parametrize(
    "trainer_cls",
    [NRE_A, NRE_B],
    ids=["nre_a", "nre_b"],
)
def test_builder_role_shapes(trainer_cls):
    """Verify input_shape=theta and condition_shape=x with asymmetric dims."""
    num_dim_theta = 2
    num_dim_x = 5
    prior = MultivariateNormal(zeros(num_dim_theta), eye(num_dim_theta))
    builder = RatioEstimatorBuilder(model="resnet", hidden_features=16)
    inference = trainer_cls(prior, classifier=builder, show_progress_bars=False)

    theta = prior.sample((200,))
    x = torch.randn(200, num_dim_x)
    estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=2, training_batch_size=100
    )

    assert estimator.input_shape == torch.Size([num_dim_theta])
    assert estimator.condition_shape == torch.Size([num_dim_x])


@pytest.mark.parametrize(
    "estimator",
    (
        RatioEstimatorBuilder(model="resnet"),
        RatioEstimatorBuilder(model="mlp"),
        RatioEstimatorBuilder(model="linear"),
        "resnet",
        classifier_nn(model="resnet"),
    ),
    ids=["resnet_builder", "mlp_builder", "linear_builder", "string", "callable"],
)
def test_check_estimator_arg_accepts_valid_inputs(estimator):
    """check_estimator_arg accepts ratio builders, strings, and callables."""
    check_estimator_arg(estimator)
