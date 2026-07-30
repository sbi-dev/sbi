# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Fast contract tests for composed vector-field standardization."""

import pickle

import pytest
import torch
from torch.distributions import Independent, Normal

from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior
from sbi.inference.potentials.vector_field_potential import VectorFieldBasedPotential
from sbi.neural_nets.factory import posterior_flow_nn, posterior_score_nn
from sbi.neural_nets.net_builders.vector_field_nets import (
    build_vector_field_estimator,
)
from sbi.utils.sbiutils import z_standardization

NUM_DIM = 3
_COMPOSE_KEYS = ("_theta_shift", "_theta_scale", "_compose_standardization")


def _batches(dim=NUM_DIM):
    torch.manual_seed(0)
    return 100.0 + 5.0 * torch.randn(32, dim), torch.randn(32, dim)


def _build(compose=True, estimator_type="flow", **kwargs):
    theta, x = _batches()
    return build_vector_field_estimator(
        theta,
        x,
        estimator_type=estimator_type,
        z_score_x="independent",
        compose_standardization=compose,
        **kwargs,
    )


def _prior():
    return Independent(Normal(torch.zeros(NUM_DIM), 200.0 * torch.ones(NUM_DIM)), 1)


def _posterior(estimator_type="flow", sample_with="ode", **kwargs):
    return VectorFieldPosterior(
        vector_field_estimator=_build(estimator_type=estimator_type, **kwargs),
        prior=_prior(),
        sample_with=sample_with,
    )


@pytest.mark.parametrize(
    "estimator_type,kwargs",
    [("flow", {}), ("score", {"sde_type": "ve"})],
)
def test_build_compose_on_unit_stats_and_affine(estimator_type, kwargs):
    theta, _ = _batches()
    estimator = _build(estimator_type=estimator_type, **kwargs)
    shift, scale = z_standardization(theta, structured_dims=False)

    assert estimator.compose_enabled
    assert torch.equal(estimator.mean_0, torch.zeros_like(estimator.mean_0))
    assert torch.equal(estimator.std_0, torch.ones_like(estimator.std_0))
    assert torch.allclose(estimator._theta_shift.flatten(), shift.float())
    assert torch.allclose(
        estimator._theta_scale.flatten(), scale.clamp_min(1e-20).float()
    )


@pytest.mark.parametrize(
    "estimator_type,kwargs",
    [("flow", {}), ("score", {"sde_type": "ve"})],
)
def test_build_compose_off_has_identity_affine(estimator_type, kwargs):
    estimator = _build(compose=False, estimator_type=estimator_type, **kwargs)

    assert not estimator.compose_enabled
    assert torch.equal(estimator._theta_shift, torch.zeros_like(estimator._theta_shift))
    assert torch.equal(estimator._theta_scale, torch.ones_like(estimator._theta_scale))


def test_build_rejects_compose_plus_baseline():
    theta, x = _batches()
    with pytest.raises(ValueError, match="cannot be used together"):
        build_vector_field_estimator(
            theta,
            x,
            estimator_type="flow",
            z_score_x="independent",
            gaussian_baseline=True,
            compose_standardization=True,
        )


def test_affine_contract():
    estimator = _build()
    estimator._theta_scale.copy_(torch.tensor([[2.0, 3.0, 5.0]]))
    theta = 100.0 + 5.0 * torch.randn(10, NUM_DIM)

    assert torch.allclose(estimator.from_z(estimator.to_z(theta)), theta)
    assert torch.allclose(
        estimator.log_abs_det(), torch.log(torch.tensor([2.0, 3.0, 5.0])).sum()
    )


@pytest.mark.gpu
def test_affine_contract_cuda():
    estimator = _build().cuda()
    theta = (100.0 + 5.0 * torch.randn(10, NUM_DIM)).cuda()

    reconstructed = estimator.from_z(estimator.to_z(theta))
    assert reconstructed.is_cuda
    assert torch.allclose(reconstructed, theta, atol=1e-5)


class _WrappedEstimator(torch.nn.Module):
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator


@pytest.mark.parametrize(
    "case",
    ["legacy", "partial", "full", "prefixed", "baseline", "nonunit"],
)
def test_checkpoint_loading(case):
    source = _build(compose=case not in {"legacy", "prefixed"})

    if case == "prefixed":
        source = _WrappedEstimator(source)
        state_dict = source.state_dict()
        for key in list(state_dict):
            if key.endswith(_COMPOSE_KEYS):
                del state_dict[key]
        destination = _WrappedEstimator(_build(compose=True))
        destination.load_state_dict(state_dict)
        assert not destination.estimator.compose_enabled
        return

    state_dict = source.state_dict()
    if case == "legacy":
        for key in _COMPOSE_KEYS:
            state_dict.pop(key)
        destination = _build(compose=True)
        destination.load_state_dict(state_dict)
        assert not destination.compose_enabled
        return

    if case == "partial":
        state_dict.pop("_theta_scale")
        with pytest.raises(RuntimeError, match="_theta_scale"):
            _build(compose=False).load_state_dict(state_dict)
        return

    if case == "baseline":
        with pytest.raises(ValueError, match="cannot be used together"):
            _build(compose=False, gaussian_baseline=True).load_state_dict(state_dict)
        return

    if case == "nonunit":
        state_dict["mean_0"] = torch.ones_like(state_dict["mean_0"])
        state_dict["std_0"] = 2 * torch.ones_like(state_dict["std_0"])
        with pytest.raises(ValueError, match="not unit"):
            _build(compose=False).load_state_dict(state_dict)
        return

    destination = _build(compose=False)
    destination.load_state_dict(state_dict)
    assert destination.compose_enabled
    assert torch.equal(destination._theta_shift, source._theta_shift)
    assert torch.equal(destination._theta_scale, source._theta_scale)


@pytest.mark.parametrize(
    "entrypoint,error_match",
    [
        ("sample_iid", "iid"),
        ("sample_guidance", "guided"),
        ("log_prob_iid", "iid"),
        ("map", "MAP"),
        ("set_x_iid", "iid"),
        ("set_x_guidance", "guided"),
        ("sample_batched", "sample_batched"),
    ],
)
def test_unsupported_entrypoints_raise(entrypoint, error_match):
    posterior = _posterior()
    potential = posterior.potential_fn
    x_iid = torch.randn(2, NUM_DIM)

    with pytest.raises(NotImplementedError, match=error_match):
        if entrypoint == "sample_iid":
            posterior.sample((2,), x=x_iid, show_progress_bars=False)
        elif entrypoint == "sample_guidance":
            posterior.sample(
                (2,),
                x=torch.zeros(1, NUM_DIM),
                guidance_method="classifier_free",
                show_progress_bars=False,
            )
        elif entrypoint == "log_prob_iid":
            posterior.log_prob(torch.zeros(1, NUM_DIM), x=x_iid)
        elif entrypoint == "map":
            posterior.map(show_progress_bars=False)
        elif entrypoint == "set_x_iid":
            potential.set_x(x_iid, x_is_iid=True)
        elif entrypoint == "set_x_guidance":
            potential.set_x(
                torch.zeros(1, NUM_DIM),
                guidance_method="classifier_free",
            )
        else:
            posterior.sample_batched(torch.Size([2]), x=x_iid)


@pytest.mark.parametrize(
    "estimator_type,sample_with,kwargs",
    [
        ("flow", "ode", {}),
        ("score", "sde", {"sde_type": "ve"}),
    ],
)
def test_samples_are_returned_in_theta_space(estimator_type, sample_with, kwargs):
    posterior = _posterior(
        estimator_type=estimator_type,
        sample_with=sample_with,
        **kwargs,
    )
    samples = posterior.sample(
        (5,),
        x=torch.zeros(1, NUM_DIM),
        steps=3,
        reject_outside_prior=False,
        show_progress_bars=False,
    )

    assert samples.shape == (5, NUM_DIM)
    assert torch.isfinite(samples).all()
    assert samples.abs().mean() > 10.0


def test_single_observation_log_prob_jacobian_exact():
    dim = 2
    theta, x = _batches(dim)
    estimator = build_vector_field_estimator(
        theta,
        x,
        estimator_type="flow",
        z_score_x="independent",
        compose_standardization=True,
    )
    prior = Independent(Normal(torch.zeros(dim), 100.0 * torch.ones(dim)), 1)
    potential = VectorFieldBasedPotential(
        estimator, prior=prior, x_o=None, device="cpu"
    )
    potential.set_x(torch.zeros(1, dim), x_is_iid=False)
    estimator._theta_scale.copy_(torch.tensor([[2.0, 3.0]]))
    potential.flow.log_prob = lambda z: torch.full(z.shape[:-1], 7.0)

    expected = 7.0 - torch.log(torch.tensor([2.0, 3.0])).sum()
    assert torch.allclose(
        potential(torch.zeros(1, dim)).flatten(), expected.reshape(-1)
    )


def test_pre_compose_pickled_posterior_samples():
    estimator = _build(compose=False)
    posterior = VectorFieldPosterior(
        vector_field_estimator=estimator,
        prior=_prior(),
        sample_with="ode",
    )
    for key in _COMPOSE_KEYS:
        del estimator._buffers[key]

    restored = pickle.loads(pickle.dumps(posterior))
    samples = restored.sample(
        (2,),
        x=torch.zeros(1, NUM_DIM),
        reject_outside_prior=False,
        show_progress_bars=False,
    )

    assert not restored.vector_field_estimator.compose_enabled
    assert samples.shape == (2, NUM_DIM)


@pytest.mark.parametrize("factory", [posterior_flow_nn, posterior_score_nn])
@pytest.mark.parametrize(
    "z_score_theta", [None, "none", "structured", "transform_to_unconstrained"]
)
def test_compose_requires_independent_theta_z_score(factory, z_score_theta):
    with pytest.raises(ValueError, match="z_score_theta='independent'"):
        factory(
            compose_standardization=True,
            z_score_theta=z_score_theta,
        )
