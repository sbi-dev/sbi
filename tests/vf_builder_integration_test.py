# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import inspect
import warnings
from dataclasses import fields as dc_fields
from typing import get_args

import pytest
import torch
from torch import zeros
from torch.distributions import MultivariateNormal

from sbi.inference import FMPE, NPSE
from sbi.neural_nets.factory import posterior_flow_nn, posterior_score_nn
from sbi.neural_nets.net_builders.estimator_configs import (
    _FLOW_ONLY_FIELDS,
    _SCORE_ONLY_FIELDS,
    MAFConfig,
    VectorFieldEstimatorBuilder,
)


@pytest.mark.parametrize(
    "factory_fn", [posterior_flow_nn, posterior_score_nn], ids=["flow", "score"]
)
def test_advertised_time_emb_types_all_build(factory_fn):
    """Every value the factory's `time_emb_type` Literal advertises must build.

    The annotation advertised `"fourier"` while the networks only accept
    `"random_fourier"`, so the annotated value crashed at build time and the
    working value failed type checking.
    """
    annotation = inspect.signature(factory_fn).parameters["time_emb_type"].annotation
    values = get_args(annotation)
    assert values, "time_emb_type lost its Literal annotation"
    for value in values:
        builder = factory_fn(model="mlp", time_emb_type=value)
        builder(torch.randn(10, 2), torch.randn(10, 3))


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
        trainer_cls(prior=prior, vf_estimator=MAFConfig())


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


@pytest.mark.parametrize(
    "builder_sde,trainer_sde,should_raise,expected_cls",
    [
        # values agree then no error, uses the agreed value
        ("ve", "ve", False, "VEScoreEstimator"),
        # values conflict then must raise
        ("ve", "vp", True, None),
        # builder only (trainer omits) then no error, builder's value wins
        ("vp", None, False, "VPScoreEstimator"),
        # builder sde_type unset, trainer explicit then trainer's value forwarded
        (None, "vp", False, "VPScoreEstimator"),
    ],
    ids=["agree", "conflict", "builder-only", "trainer-only"],
)
def test_npse_sde_type_interactions(
    builder_sde, trainer_sde, should_raise, expected_cls, gaussian_sims
):
    """sde_type must raise only when both are supplied and they disagree."""
    prior, theta, x = gaussian_sims

    builder_kwargs = {"estimator_type": "score"}
    if builder_sde is not None:
        builder_kwargs["sde_type"] = builder_sde

    trainer_kwargs = {"prior": prior}
    if trainer_sde is not None:
        trainer_kwargs["sde_type"] = trainer_sde

    if should_raise:
        with pytest.raises(ValueError, match="sde_type"):
            NPSE(
                vf_estimator=VectorFieldEstimatorBuilder(**builder_kwargs),
                **trainer_kwargs,
            )
    else:
        trainer = NPSE(
            vf_estimator=VectorFieldEstimatorBuilder(**builder_kwargs),
            **trainer_kwargs,
        )
        trainer.append_simulations(theta, x)
        estimator = trainer.train(max_num_epochs=1, training_batch_size=100)
        assert type(estimator).__name__ == expected_cls, (
            f"Expected {expected_cls}, got {type(estimator).__name__}"
        )


@pytest.mark.parametrize("est_type", ["flow", "score"])
def test_default_builder_matches_factory_z_scoring(est_type):
    """Builder and factory must produce identical z-scoring buffers."""
    theta = torch.randn(200, 2) + 5.0
    x = theta + 0.1 * torch.randn_like(theta)

    # Build via builder.
    builder = VectorFieldEstimatorBuilder(model="mlp", estimator_type=est_type)
    est_builder = builder.build(batch_input=theta, batch_condition=x)

    # Build via factory.
    if est_type == "flow":
        from sbi.neural_nets.factory import posterior_flow_nn

        factory_fn = posterior_flow_nn(model="mlp")
    else:
        from sbi.neural_nets.factory import posterior_score_nn

        factory_fn = posterior_score_nn(model="mlp")
    est_factory = factory_fn(theta, x)

    # Compare z-scoring buffers on the input (theta) side.
    assert torch.allclose(est_builder.mean_0, est_factory.mean_0), (
        "mean_0 mismatch between builder and factory"
    )
    assert torch.allclose(est_builder.std_0, est_factory.std_0), (
        "std_0 mismatch between builder and factory"
    )

    # Compare z-scoring on the condition (x) side.
    from sbi.utils.sbiutils import Standardize

    def _get_standardize(module):
        for m in module.modules():
            if isinstance(m, Standardize):
                return m
        return None

    std_builder = _get_standardize(est_builder.embedding_net)
    std_factory = _get_standardize(est_factory.embedding_net)
    assert std_builder is not None, "Builder embedding_net missing Standardize"
    assert std_factory is not None, "Factory embedding_net missing Standardize"
    assert torch.allclose(std_builder.mean, std_factory.mean), (
        "embedding_net Standardize mean mismatch"
    )
    assert torch.allclose(std_builder.std, std_factory.std), (
        "embedding_net Standardize std mismatch"
    )


@pytest.mark.parametrize(
    "net,batch_y_3d,is_x_emb_seq_kwarg",
    [
        ("mlp", False, None),
        ("ada_mlp", False, None),
        ("transformer", False, None),
        ("transformer_cross_attn", True, None),
        # net="transformer" plus explicit is_x_emb_seq=True via kwargs
        ("transformer", True, True),
    ],
    ids=["mlp", "ada_mlp", "transformer", "cross_attn", "transformer+is_x_emb_seq"],
)
def test_all_architectures_build(net, batch_y_3d, is_x_emb_seq_kwarg):
    """All four architectures must build without error."""
    from sbi.neural_nets.net_builders.vector_field_nets import (
        build_vector_field_estimator,
    )

    batch_x = torch.randn(10, 3)
    # Cross-attention needs sequence-shaped conditioning.
    batch_y = torch.randn(10, 4, 8) if batch_y_3d else torch.randn(10, 5)

    extra = {}
    if is_x_emb_seq_kwarg is not None:
        extra["is_x_emb_seq"] = is_x_emb_seq_kwarg

    estimator = build_vector_field_estimator(
        batch_x=batch_x,
        batch_y=batch_y,
        net=net,
        estimator_type="flow",
        **extra,
    )
    assert estimator is not None
    assert estimator.input_shape == torch.Size([3])
