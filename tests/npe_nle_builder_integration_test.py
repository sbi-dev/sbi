# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings

import pytest
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import MNLE, MNPE, NLE_A, NPE_C
from sbi.neural_nets import likelihood_nn, posterior_nn
from sbi.neural_nets.net_builders.estimator_configs import (
    DensityEstimatorBuilder,
)
from sbi.utils.user_input_checks import check_estimator_arg

_TRAINERS = [(NPE_C, posterior_nn, "theta"), (NLE_A, likelihood_nn, "x")]


@pytest.mark.parametrize(
    "trainer_cls,factory_fn",
    [(t, f) for t, f, _ in _TRAINERS],
    ids=["npe", "nle"],
)
def test_no_warning_for_valid_inputs(trainer_cls, factory_fn):
    """None default, builder, and callable should not emit FutureWarning."""
    prior = MultivariateNormal(zeros(2), eye(2))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        trainer_cls(prior, show_progress_bars=False)
        trainer_cls(
            prior,
            density_estimator=DensityEstimatorBuilder(model="maf"),
            show_progress_bars=False,
        )
        trainer_cls(
            prior,
            density_estimator=factory_fn(model="maf"),
            show_progress_bars=False,
        )


@pytest.mark.parametrize(
    "trainer_cls",
    [t for t, _, _ in _TRAINERS],
    ids=["npe", "nle"],
)
def test_string_emits_deprecation_warning(trainer_cls):
    """Passing a string to density_estimator should emit FutureWarning."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.warns(FutureWarning, match="deprecated"):
        trainer_cls(prior, density_estimator="maf", show_progress_bars=False)


@pytest.mark.parametrize(
    "trainer_cls,factory_fn,input_var",
    _TRAINERS,
    ids=["npe", "nle"],
)
@pytest.mark.parametrize("model", ("maf", "nsf"))
def test_train_with_builder(trainer_cls, factory_fn, input_var, model):
    """Train with a DensityEstimatorBuilder, verify loss and posterior sampling."""
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    builder = DensityEstimatorBuilder(model=model, hidden_features=16, num_transforms=2)
    inference = trainer_cls(prior, density_estimator=builder, show_progress_bars=False)

    theta = prior.sample((200,))
    x = theta + 0.1 * torch.randn_like(theta)
    density_estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=2, training_batch_size=100
    )

    # Verify finite loss on a fresh batch with correct role order.
    fresh_theta = prior.sample((10,))
    fresh_x = fresh_theta + 0.1 * torch.randn_like(fresh_theta)
    if input_var == "theta":
        # NPE: loss(input=θ, condition=x)
        loss = density_estimator.loss(fresh_theta, condition=fresh_x)
    else:
        # NLE: loss(input=x, condition=θ)
        loss = density_estimator.loss(fresh_x, condition=fresh_theta)
    assert loss.shape == (10,)
    assert torch.isfinite(loss).all()

    # Posterior should be constructable and produce correct-shaped samples.
    posterior = inference.build_posterior()
    x_o = zeros(1, num_dim)
    samples = posterior.sample((10,), x=x_o)
    assert samples.shape == (10, num_dim)


@pytest.mark.parametrize(
    "trainer_cls,input_var",
    [(t, v) for t, _, v in _TRAINERS],
    ids=["npe", "nle"],
)
def test_builder_role_shapes(trainer_cls, input_var):
    """Verify input_shape matches the modeled variable, not the condition."""
    num_dim_theta = 2
    num_dim_x = 5
    prior = MultivariateNormal(zeros(num_dim_theta), eye(num_dim_theta))
    builder = DensityEstimatorBuilder(model="maf", hidden_features=16, num_transforms=2)
    inference = trainer_cls(prior, density_estimator=builder, show_progress_bars=False)

    theta = prior.sample((200,))
    x = torch.randn(200, num_dim_x)
    estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=2, training_batch_size=100
    )

    if input_var == "theta":
        # NPE models p(θ|x): input=θ, condition=x
        assert estimator.input_shape == torch.Size([num_dim_theta])
        assert estimator.condition_shape == torch.Size([num_dim_x])
    else:
        # NLE models p(x|θ): input=x, condition=θ
        assert estimator.input_shape == torch.Size([num_dim_x])
        assert estimator.condition_shape == torch.Size([num_dim_theta])


@pytest.mark.parametrize("trainer_cls", [MNLE, MNPE], ids=["mnle", "mnpe"])
def test_mixed_trainer_default_no_warning(trainer_cls):
    """MNLE/MNPE default must not emit FutureWarning (string converted to callable)."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        trainer_cls(prior, show_progress_bars=False)


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
