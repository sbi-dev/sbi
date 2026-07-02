# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings

import pytest
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import MNLE, MNPE, NLE_A, NPE_C
from sbi.neural_nets import likelihood_nn, posterior_nn
from sbi.neural_nets.estimators import MixedDensityEstimator
from sbi.neural_nets.net_builders.estimator_configs import (
    DensityEstimatorBuilder,
    MixedDensityEstimatorBuilder,
)
from sbi.utils import BoxUniform
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
def test_mixed_no_warning_for_valid_inputs(trainer_cls):
    """None default and builder should not emit FutureWarning."""
    prior = MultivariateNormal(zeros(2), eye(2))
    builder = MixedDensityEstimatorBuilder(continuous_model="nsf")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        trainer_cls(prior, show_progress_bars=False)
        trainer_cls(prior, density_estimator=builder, show_progress_bars=False)


@pytest.mark.parametrize(
    "trainer_cls,string",
    [(MNLE, "mnle"), (MNPE, "mnpe")],
    ids=["mnle", "mnpe"],
)
def test_mixed_string_emits_deprecation_warning(trainer_cls, string):
    """Passing 'mnle'/'mnpe' string should emit FutureWarning."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.warns(FutureWarning, match="deprecated"):
        trainer_cls(prior, density_estimator=string, show_progress_bars=False)


@pytest.mark.parametrize(
    "trainer_cls,wrong_string",
    [(MNLE, "maf"), (MNPE, "nsf")],
    ids=["mnle", "mnpe"],
)
def test_mixed_wrong_string_raises(trainer_cls, wrong_string):
    """Passing a non-mnle/mnpe string should raise ValueError."""
    prior = MultivariateNormal(zeros(2), eye(2))
    with pytest.raises(ValueError, match="supports only"):
        trainer_cls(prior, density_estimator=wrong_string, show_progress_bars=False)


def test_mixed_builder_invalid_continuous_model():
    """Invalid continuous_model should raise ValueError."""
    with pytest.raises(ValueError, match="continuous_model"):
        MixedDensityEstimatorBuilder(continuous_model="invalid")


@pytest.mark.parametrize(
    "trainer_cls,make_data,input_var",
    [
        (
            MNLE,
            # MNLE: theta=continuous params, x=mixed observations
            lambda n: (
                BoxUniform(low=zeros(2), high=torch.ones(2)).sample((n,)),
                torch.cat(
                    [torch.randn(n, 3), torch.randint(0, 3, (n, 2)).float()], dim=-1
                ),
            ),
            "x",
        ),
        (
            MNPE,
            # MNPE: theta=mixed params, x=continuous observations
            lambda n: (
                torch.cat(
                    [torch.randn(n, 2), torch.randint(0, 3, (n, 1)).float()], dim=-1
                ),
                torch.randn(n, 4),
            ),
            "theta",
        ),
    ],
    ids=["mnle", "mnpe"],
)
def test_mixed_train_with_builder(trainer_cls, make_data, input_var):
    """Train MNLE/MNPE with MixedDensityEstimatorBuilder end-to-end."""
    builder = MixedDensityEstimatorBuilder(
        continuous_model="nsf", hidden_features=16, num_transforms=2
    )
    trainer = trainer_cls(density_estimator=builder, show_progress_bars=False)

    theta, x = make_data(200)
    estimator = trainer.append_simulations(theta, x).train(
        max_num_epochs=2, training_batch_size=100
    )
    assert isinstance(estimator, MixedDensityEstimator)

    # verify the estimator actually evaluates with finite loss
    theta_fresh, x_fresh = make_data(10)
    if input_var == "theta":
        loss = estimator.loss(theta_fresh, condition=x_fresh)
    else:
        loss = estimator.loss(x_fresh, condition=theta_fresh)
    assert loss.shape == (10,)
    assert torch.isfinite(loss).all()


@pytest.mark.parametrize(
    "estimator",
    (
        DensityEstimatorBuilder(model="maf"),
        MixedDensityEstimatorBuilder(continuous_model="nsf"),
        "maf",
        posterior_nn(model="maf"),
    ),
    ids=["density_builder", "mixed_builder", "string", "callable"],
)
def test_check_estimator_arg_accepts_valid_inputs(estimator):
    """check_estimator_arg should accept builders, strings, and callables."""
    check_estimator_arg(estimator)


def test_check_estimator_arg_rejects_module():
    """check_estimator_arg should reject raw nn.Module instances."""
    with pytest.raises(TypeError):
        check_estimator_arg(torch.nn.Linear(3, 3))
