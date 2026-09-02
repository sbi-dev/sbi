# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from torch import Tensor, eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import (
    ImportanceSamplingPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
)
from sbi.inference.posteriors.ensemble_posterior import EnsemblePotential
from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters
from sbi.inference.potentials.base_potential import (
    BasePotential,
    CustomPotentialWrapper,
)
from sbi.inference.potentials.posterior_based_potential import PosteriorBasedPotential
from sbi.inference.potentials.vector_field_potential import VectorFieldBasedPotential
from sbi.neural_nets import posterior_nn, posterior_score_nn
from sbi.utils import BoxUniform
from sbi.utils.conditional_density_utils import ConditionedPotential


@pytest.mark.parametrize(
    "sampling_method",
    [
        ImportanceSamplingPosterior,
        pytest.param(MCMCPosterior, marks=pytest.mark.mcmc),
        RejectionPosterior,
        VIPosterior,
    ],
)
def test_callable_potential(
    sampling_method, mcmc_params_accurate: MCMCPosteriorParameters
):
    """Test whether callable potentials can be used to sample from a Gaussian."""
    dim = 2
    mean = 2.5
    cov = 2.0
    x_o = 1 * ones((dim,))
    target_density = MultivariateNormal(mean * ones((dim,)), cov * eye(dim))

    def potential(theta, x_o):
        return target_density.log_prob(theta + x_o)

    proposal = MultivariateNormal(zeros((dim,)), 5 * eye(dim))

    if sampling_method == ImportanceSamplingPosterior:
        approx_density = sampling_method(
            potential_fn=potential, proposal=proposal, method="sir"
        )
        approx_samples = approx_density.sample((1024,), oversampling_factor=1024, x=x_o)
    elif sampling_method == MCMCPosterior:
        approx_density = sampling_method(potential_fn=potential, proposal=proposal)
        approx_samples = approx_density.sample(
            (1024,), x=x_o, **asdict(mcmc_params_accurate)
        )
    elif sampling_method == VIPosterior:
        approx_density = sampling_method(
            potential_fn=potential, prior=proposal
        ).set_default_x(x_o)
        approx_density = approx_density.train()
        approx_samples = approx_density.sample((1024,))
    elif sampling_method == RejectionPosterior:
        approx_density = sampling_method(
            potential_fn=potential, proposal=proposal
        ).set_default_x(x_o)
        approx_samples = approx_density.sample((1024,))

    sample_mean = torch.mean(approx_samples, dim=0)
    sample_std = torch.std(approx_samples, dim=0)
    assert torch.allclose(sample_mean, torch.as_tensor(mean) - x_o, atol=0.2)
    assert torch.allclose(sample_std, torch.sqrt(torch.as_tensor(cov)), atol=0.1)


@pytest.mark.parametrize(
    "condition",
    [
        torch.rand(1, 2),
        pytest.param(
            torch.rand(2, 2),
            marks=pytest.mark.xfail(
                raises=ValueError,
                match="Condition with batch size > 1 not supported",
            ),
        ),
    ],
)
def test_conditioned_potential(condition: Tensor):
    potential_fn = CustomPotentialWrapper(
        potential_fn=lambda theta, x_o: theta,
        prior=BoxUniform(low=zeros(2), high=ones(2)),
    )

    ConditionedPotential(potential_fn, condition=condition, dims_to_sample=[0])


def _build_potential(potential_type: str, x_o: Tensor | None = None) -> BasePotential:
    """Build a small untrained potential of the given type."""
    prior = BoxUniform(zeros(2), ones(2))
    theta = prior.sample((20,))
    x = torch.randn(20, 2)

    if potential_type == "posterior":
        return PosteriorBasedPotential(posterior_nn("mdn")(theta, x), prior, x_o=x_o)
    return VectorFieldBasedPotential(posterior_score_nn()(theta, x), prior, x_o=x_o)


@pytest.mark.parametrize("potential_type", ("posterior", "vector_field"))
def test_bind_preserves_estimator(potential_type: str):
    """Test that bind() returns a potential holding the same estimator object.

    The planned x_o NaN-tolerance derivation (ADR 0001) reads the embedding net
    that consumes x through the potential's estimator. A bind() that re-threads
    state by hand and drops or copies the estimator would silently change the
    derived tolerance on the bound potential.
    """
    potential = _build_potential(potential_type)
    bound = potential.bind(zeros(1, 2))

    if potential_type == "posterior":
        assert bound.posterior_estimator is potential.posterior_estimator
    else:
        assert bound.vector_field_estimator is potential.vector_field_estimator

    assert bound is not potential
    assert bound.x_o is not None


@pytest.mark.parametrize("potential_type", ("posterior", "vector_field"))
def test_potential_with_x_o_at_construction(potential_type: str):
    """Test that x_o passed to the constructor behaves like a later bind()."""
    potential = _build_potential(potential_type, x_o=zeros(1, 2))

    assert potential.x_is_iid is False
    assert torch.isfinite(potential(zeros(1, 2))).all()


def test_base_bind_default_for_custom_potentials():
    """Test that a subclass overriding only __call__ survives bind()."""

    class QuadraticPotential(BasePotential):
        def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
            return -(theta**2).sum(-1)

    potential = QuadraticPotential(prior=BoxUniform(zeros(2), ones(2)))
    bound = potential.bind(zeros(1, 2))

    assert bound is not potential
    assert bound.prior is potential.prior
    assert torch.equal(bound.x_o, zeros(1, 2))
    assert torch.equal(bound(ones(3, 2)), -2 * ones(3))


def test_bind_clears_guidance_by_default():
    """Test that bind() without guidance args clears guidance, like set_x did."""
    potential = _build_potential("vector_field")

    guided = potential.bind(
        zeros(1, 2), guidance_method="classifier_free", guidance_params={"scale": 2.0}
    )
    assert guided.guidance_method == "classifier_free"

    rebound = guided.bind(zeros(1, 2))
    assert rebound.guidance_method is None
    assert rebound.guidance_params is None


def test_ensemble_bind_and_construction_respect_components():
    """Test that the ensemble binds its components and keeps their iid defaults."""
    components = [_build_potential("posterior") for _ in range(2)]
    prior = components[0].prior
    ensemble = EnsemblePotential(
        potential_fns=components,
        weights=torch.tensor([0.5, 0.5]),
        prior=prior,
        x_o=None,
    )

    bound = ensemble.bind(zeros(1, 2))
    assert all(p.x_is_iid is False for p in bound.potential_fns)

    bound_iid = ensemble.bind(zeros(1, 2), x_is_iid=True)
    assert all(p.x_is_iid is True for p in bound_iid.potential_fns)

    constructed = EnsemblePotential(
        potential_fns=components,
        weights=torch.tensor([0.5, 0.5]),
        prior=prior,
        x_o=zeros(1, 2),
    )
    assert all(p.return_x_o() is not None for p in constructed.potential_fns)
    assert torch.isfinite(constructed(zeros(1, 2))).all()


def test_set_x_none_clears_observation():
    """Test that set_x(None) clears the observation, as it did before bind()."""
    bound = _build_potential("vector_field").bind(zeros(1, 2))
    assert bound.return_x_o() is not None

    with pytest.warns(FutureWarning):
        bound.set_x(None)
    assert bound.return_x_o() is None
