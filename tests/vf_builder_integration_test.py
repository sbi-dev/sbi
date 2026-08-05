# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from dataclasses import fields as dc_fields

import pytest
import torch
from torch import zeros
from torch.distributions import MultivariateNormal

from sbi.inference import FMPE, NPSE
from sbi.neural_nets.net_builders.estimator_configs import (
    _FLOW_ONLY_FIELDS,
    _SCORE_ONLY_FIELDS,
    DensityEstimatorBuilder,
    VectorFieldEstimatorBuilder,
)


@pytest.fixture
def gaussian_sims():
    prior = MultivariateNormal(zeros(2), torch.eye(2))
    theta = prior.sample((200,))
    x = theta + 0.1 * torch.randn_like(theta)
    return prior, theta, x


@pytest.mark.parametrize("trainer_cls", [FMPE, NPSE])
def test_no_warning_for_valid_inputs(trainer_cls, gaussian_sims):
    """None, builder, and callable should not emit FutureWarning."""
    prior, _, _ = gaussian_sims
    for inp in [None, VectorFieldEstimatorBuilder(model="mlp"), lambda t, x: None]:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            trainer_cls(prior=prior, vf_estimator=inp)


@pytest.mark.parametrize(
    "trainer_cls,kwarg,match",
    [
        (NPSE, {"score_estimator": "mlp"}, "score_estimator"),
        (FMPE, {"density_estimator": lambda t, x: None}, "density_estimator"),
    ],
    ids=["npse-score", "fmpe-density"],
)
def test_legacy_kwarg_warns(trainer_cls, kwarg, match, gaussian_sims):
    prior, _, _ = gaussian_sims
    with pytest.warns(FutureWarning, match=match):
        trainer_cls(prior=prior, **kwarg)


@pytest.mark.parametrize("trainer_cls", [FMPE, NPSE])
def test_wrong_builder_type_raises(trainer_cls, gaussian_sims):
    prior, _, _ = gaussian_sims
    with pytest.raises(TypeError, match="VectorFieldEstimatorBuilder"):
        trainer_cls(prior=prior, vf_estimator=DensityEstimatorBuilder(model="maf"))


def test_builder_invalid_model():
    with pytest.raises(ValueError, match="Unknown model"):
        VectorFieldEstimatorBuilder(model="invalid")


@pytest.mark.parametrize(
    "est_type,bad_field",
    [("flow", {"sigma_min": 0.01}), ("score", {"gaussian_baseline": True})],
    ids=["flow-rejects-score", "score-rejects-flow"],
)
def test_estimator_type_field_guard(est_type, bad_field):
    with pytest.raises(ValueError, match="do not apply"):
        VectorFieldEstimatorBuilder(estimator_type=est_type, **bad_field)


def test_estimator_type_none_skips_field_guard():
    """When estimator_type is None, flow/score field guard is skipped."""
    # Should NOT raise even though sigma_min is a score-only field,
    # because the guard is deferred to the trainer.
    builder = VectorFieldEstimatorBuilder(model="mlp", sigma_min=0.01)
    assert builder.estimator_type is None
    assert builder.sigma_min == 0.01


def test_estimator_type_none_build_raises():
    """build() must raise if estimator_type is still None."""
    builder = VectorFieldEstimatorBuilder(model="mlp")
    with pytest.raises(ValueError, match="estimator_type is None"):
        builder.build(
            batch_input=torch.randn(10, 2),
            batch_condition=torch.randn(10, 2),
        )


@pytest.mark.parametrize(
    "trainer_cls,expected_type",
    [(FMPE, "FlowMatchingEstimator"), (NPSE, "ScoreEstimator")],
    ids=["fmpe-resolves-flow", "npse-resolves-score"],
)
def test_trainer_resolves_none_estimator_type(
    trainer_cls, expected_type, gaussian_sims
):
    """Trainers must resolve estimator_type=None to the correct type."""
    prior, theta, x = gaussian_sims
    builder = VectorFieldEstimatorBuilder(model="mlp")
    assert builder.estimator_type is None
    trainer = trainer_cls(prior=prior, vf_estimator=builder)
    trainer.append_simulations(theta, x)
    est = trainer.train(max_num_epochs=1, training_batch_size=100)
    assert expected_type in type(est).__name__


@pytest.mark.parametrize(
    "trainer_cls,wrong_type,match",
    [
        (FMPE, "score", "flow-matching"),
        (NPSE, "flow", "score estimators"),
    ],
    ids=["fmpe-rejects-score", "npse-rejects-flow"],
)
def test_trainer_rejects_wrong_estimator_type(
    trainer_cls, wrong_type, match, gaussian_sims
):
    """Trainer must raise when given a builder with the wrong estimator_type."""
    prior, _, _ = gaussian_sims
    with pytest.raises(ValueError, match=match):
        trainer_cls(
            prior=prior,
            vf_estimator=VectorFieldEstimatorBuilder(
                model="mlp", estimator_type=wrong_type
            ),
        )


def test_per_arch_validation_rejects_num_heads_on_mlp():
    """num_heads is transformer-only; must be rejected for model='mlp'."""
    with pytest.raises(ValueError, match="num_heads"):
        VectorFieldEstimatorBuilder(model="mlp", num_heads=8)


def test_score_accepts_score_fields():
    builder = VectorFieldEstimatorBuilder(
        estimator_type="score",
        sde_type="ve",
        sigma_min=0.01,
        sigma_max=50.0,
    )
    assert builder.sigma_min == 0.01


@pytest.mark.parametrize("trainer_cls", [FMPE, NPSE])
def test_train_with_builder(trainer_cls, gaussian_sims):
    """End-to-end: train with VectorFieldEstimatorBuilder and sample."""
    prior, theta, x = gaussian_sims
    est_type = "score" if trainer_cls is NPSE else "flow"
    builder_kwargs = {"model": "mlp", "estimator_type": est_type}
    if trainer_cls is NPSE:
        builder_kwargs["sde_type"] = "ve"

    trainer = trainer_cls(
        prior=prior,
        vf_estimator=VectorFieldEstimatorBuilder(**builder_kwargs),
    )
    trainer.append_simulations(theta, x)
    estimator = trainer.train(max_num_epochs=2, training_batch_size=100)

    assert estimator is not None
    posterior = trainer.build_posterior(estimator)
    samples = posterior.sample((10,), x=torch.randn(1, 2))
    assert samples.shape == (10, 2)


@pytest.mark.parametrize("trainer_cls", [FMPE, NPSE])
def test_builder_role_shapes(trainer_cls):
    """Asymmetric dims catch silent role swaps (Decision 10/13)."""
    prior = MultivariateNormal(zeros(2), torch.eye(2))
    theta = prior.sample((200,))
    x = theta.sum(dim=-1, keepdim=True) + 0.1 * torch.randn(200, 5)

    est_type = "score" if trainer_cls is NPSE else "flow"
    builder = VectorFieldEstimatorBuilder(model="mlp", estimator_type=est_type)
    trainer = trainer_cls(prior=prior, vf_estimator=builder)
    trainer.append_simulations(theta, x)
    estimator = trainer.train(max_num_epochs=1, training_batch_size=100)

    assert estimator.input_shape == torch.Size([2])
    assert estimator.condition_shape == torch.Size([5])


@pytest.mark.parametrize("trainer_cls", [FMPE, NPSE])
def test_warning_includes_import_path(trainer_cls, gaussian_sims):
    prior, _, _ = gaussian_sims
    with pytest.warns(FutureWarning, match="from sbi.neural_nets import"):
        trainer_cls(prior=prior, vf_estimator="mlp")


def test_score_only_fields_match_config():
    """_SCORE_ONLY_FIELDS must stay in sync with ScoreEstimatorConfig."""
    from sbi.neural_nets.net_builders.vector_field_nets import (
        ScoreEstimatorConfig,
        _VectorFieldBaseConfig,
    )

    score_diff = {f.name for f in dc_fields(ScoreEstimatorConfig)} - {
        f.name for f in dc_fields(_VectorFieldBaseConfig)
    }
    assert score_diff | {"sde_type"} == _SCORE_ONLY_FIELDS


def test_flow_only_fields_match_config():
    """_FLOW_ONLY_FIELDS must stay in sync with FlowEstimatorConfig."""
    from sbi.neural_nets.net_builders.vector_field_nets import (
        FlowEstimatorConfig,
        _VectorFieldBaseConfig,
    )

    flow_diff = {f.name for f in dc_fields(FlowEstimatorConfig)} - {
        f.name for f in dc_fields(_VectorFieldBaseConfig)
    }
    assert flow_diff == _FLOW_ONLY_FIELDS


@pytest.mark.parametrize(
    "trainer_cls,kwarg",
    [
        (NPSE, {"score_estimator": "mlp"}),
        (NPSE, {"density_estimator": lambda t, x: None}),
        (FMPE, {"density_estimator": lambda t, x: None}),
    ],
    ids=["npse-score", "npse-density", "fmpe-density"],
)
def test_legacy_and_vf_estimator_conflict(trainer_cls, kwarg, gaussian_sims):
    """Passing a deprecated kwarg alongside vf_estimator should raise."""
    prior, _, _ = gaussian_sims
    est_type = "score" if trainer_cls is NPSE else "flow"
    with pytest.raises(ValueError, match="Cannot pass both"):
        trainer_cls(
            prior=prior,
            vf_estimator=VectorFieldEstimatorBuilder(estimator_type=est_type),
            **kwarg,
        )


def test_npse_sde_type_conflict_with_builder(gaussian_sims):
    """Builder's sde_type is authoritative; trainer-level conflicts raise."""
    prior, _, _ = gaussian_sims
    with pytest.raises(ValueError, match="sde_type"):
        NPSE(
            prior=prior,
            vf_estimator=VectorFieldEstimatorBuilder(
                estimator_type="score",
                sde_type="ve",
            ),
            sde_type="vp",
        )


@pytest.mark.parametrize("est_type", ["flow", "score"])
def test_default_builder_applies_z_scoring(est_type):
    """Default builder must z-score inputs, matching posterior_flow_nn defaults."""
    theta = torch.randn(200, 2) + 5.0
    x = theta + 0.1 * torch.randn_like(theta)
    builder = VectorFieldEstimatorBuilder(model="mlp", estimator_type=est_type)
    estimator = builder.build(batch_input=theta, batch_condition=x)
    mean = estimator.mean_0
    std = estimator.std_0
    assert not torch.equal(mean, torch.zeros_like(mean)), (
        "mean_0 is all zeros: z-scoring was not applied to the input"
    )
    assert not torch.equal(std, torch.ones_like(std)), (
        "std_0 is all ones: z-scoring was not applied to the input"
    )

    from sbi.utils.sbiutils import Standardize

    has_standardize = any(
        isinstance(m, Standardize) for m in estimator.embedding_net.modules()
    )
    assert has_standardize, (
        "embedding_net has no Standardize layer: z-scoring was not applied to y"
    )
