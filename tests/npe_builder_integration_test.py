# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import NPE_C
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.neural_nets.net_builders.estimator_configs import (
    DensityEstimatorBuilder,
)
from sbi.utils.user_input_checks import check_estimator_arg


def test_npe_no_warning_for_valid_inputs():
    """Passing a builder, callable, or using the default should not warn."""
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        NPE_C(prior, show_progress_bars=False)
        builder = DensityEstimatorBuilder(model="maf")
        NPE_C(prior, density_estimator=builder, show_progress_bars=False)
        build_fn = posterior_nn(model="maf")
        NPE_C(prior, density_estimator=build_fn, show_progress_bars=False)


def test_npe_string_emits_deprecation_warning():
    """Passing a string to density_estimator should emit FutureWarning."""
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    with pytest.warns(FutureWarning, match="deprecated"):
        NPE_C(prior, density_estimator="maf", show_progress_bars=False)


@pytest.mark.parametrize("model", ("maf", "nsf", "mdn"))
def test_npe_train_with_builder(model):
    """Train NPE_C with a DensityEstimatorBuilder and verify the result."""
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    builder = DensityEstimatorBuilder(model=model, hidden_features=16, num_transforms=2)
    inference = NPE_C(prior, density_estimator=builder, show_progress_bars=False)

    theta = prior.sample((200,))
    x = theta + 0.1 * torch.randn_like(theta)
    density_estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=2, training_batch_size=100
    )
    assert isinstance(density_estimator, ConditionalDensityEstimator)
    # Verify the trained estimator produces finite loss on a fresh batch.
    fresh_theta = prior.sample((10,))
    fresh_x = fresh_theta + 0.1 * torch.randn_like(fresh_theta)
    loss = density_estimator.loss(fresh_theta, condition=fresh_x)
    assert loss.shape == (10,)
    assert torch.isfinite(loss).all()

    # Also verify that a posterior can be constructed and sampled from.
    posterior = inference.build_posterior()
    x_o = zeros(1, num_dim)
    samples = posterior.sample((10,), x=x_o)
    assert samples.shape == (10, num_dim)


@pytest.mark.parametrize(
    "estimator",
    (DensityEstimatorBuilder(model="maf"), "maf", posterior_nn(model="maf")),
    ids=["builder", "string", "callable"],
)
def test_check_estimator_arg_accepts_valid_inputs(estimator):
    """check_estimator_arg should accept builders, strings, and callables."""
    check_estimator_arg(estimator)


def test_check_estimator_arg_rejects_module():
    """check_estimator_arg should reject raw nn.Module instances."""
    with pytest.raises(TypeError):
        check_estimator_arg(torch.nn.Linear(3, 3))
