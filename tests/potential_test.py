# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pickle
from dataclasses import asdict

import pytest
import torch
from torch import Tensor, eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import (
    DirectPosterior,
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
from sbi.inference.potentials.likelihood_based_potential import (
    likelihood_estimator_based_potential,
)
from sbi.inference.potentials.posterior_based_potential import (
    PosteriorBasedPotential,
    posterior_estimator_based_potential,
)
from sbi.inference.potentials.ratio_based_potential import (
    ratio_estimator_based_potential,
)
from sbi.inference.potentials.vector_field_potential import VectorFieldBasedPotential
from sbi.neural_nets import (
    classifier_nn,
    likelihood_nn,
    posterior_nn,
    posterior_score_nn,
)
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
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


def _gaussian_iid_potential_and_posterior():
    """Return the potential of x_i ~ N(theta, I), theta ~ N(0, I) and its posterior."""
    prior = MultivariateNormal(zeros(2), eye(2))

    def potential(theta, x_o):
        likelihood = MultivariateNormal(theta.unsqueeze(1), eye(2))
        return likelihood.log_prob(x_o.unsqueeze(0)).sum(-1) + prior.log_prob(theta)

    def true_posterior(x_o):
        num_trials = x_o.shape[0]
        return MultivariateNormal(
            x_o.sum(0) / (num_trials + 1), eye(2) / (num_trials + 1)
        )

    return prior, potential, true_posterior


def test_importance_posterior_log_prob_after_changing_default_x():
    """The normalization constant must not be reused for another default x, also not
    for more trials of the same value."""
    torch.manual_seed(0)
    prior, potential, true_posterior = _gaussian_iid_potential_and_posterior()
    posterior = ImportanceSamplingPosterior(potential, proposal=prior)
    theta = zeros(1, 2)

    for x_o in [3 * ones(1, 2), zeros(1, 2), zeros(3, 2)]:
        log_prob = posterior.set_default_x(x_o).log_prob(theta)
        assert torch.allclose(log_prob, true_posterior(x_o).log_prob(theta), atol=0.2)


def test_importance_posterior_log_prob_with_many_iid_trials():
    """The normalization constant must not underflow for a sharp potential."""
    torch.manual_seed(0)
    prior, potential, true_posterior = _gaussian_iid_potential_and_posterior()
    x_o = torch.randn(500, 2) + 0.5
    posterior = ImportanceSamplingPosterior(potential, proposal=prior)
    theta = x_o.mean(0, keepdim=True)

    log_prob = posterior.set_default_x(x_o).log_prob(theta)
    assert torch.isfinite(log_prob).all()
    assert torch.allclose(log_prob, true_posterior(x_o).log_prob(theta), atol=0.5)


def test_importance_posterior_normalization_constant_is_estimated_at_given_x():
    """The public estimate must use the x and number of samples it is given, not
    those of the last estimate."""
    torch.manual_seed(0)
    prior, potential, _ = _gaussian_iid_potential_and_posterior()
    posterior = ImportanceSamplingPosterior(potential, proposal=prior)
    x_a, x_b = zeros(1, 2), 3 * ones(1, 2)

    posterior.log_prob(zeros(1, 2), x=x_a)
    posterior.estimate_normalization_constant(x_b, num_samples=1)
    log_z = torch.log(posterior.estimate_normalization_constant(x_b))
    true_log_z = MultivariateNormal(zeros(2), 2 * eye(2)).log_prob(x_b)
    assert torch.allclose(log_z, true_log_z, atol=0.2)


def _quadratic_potential(theta, x_o):
    return -((theta - x_o) ** 2).sum(-1)


def test_importance_posterior_pickled_before_log_space_cache():
    """Posteriors pickled by sbi 0.27 saved Z as `_normalization_constant`."""
    proposal = MultivariateNormal(zeros(2), eye(2))
    posterior = ImportanceSamplingPosterior(_quadratic_potential, proposal=proposal)
    del posterior._log_normalization_cache
    posterior._normalization_constant = None

    loaded = pickle.loads(pickle.dumps(posterior))
    assert torch.isfinite(loaded.log_prob(zeros(1, 2), x=zeros(1, 2))).all()


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


def _potential_with_trials_in_one_observation(method: str):
    """Untrained NPE or NRE potential whose observation holds 5 trials."""
    torch.manual_seed(0)
    prior = BoxUniform(-3 * ones(2), 3 * ones(2))
    theta = prior.sample((50,))
    x = theta.unsqueeze(1) + torch.randn(50, 5, 2)
    embedding = PermutationInvariantEmbedding(
        FCEmbedding(2, 8), trial_net_output_dim=8, output_dim=4
    )
    if method == "npe":
        estimator = posterior_nn("mdn", embedding_net=embedding)(theta, x)
        potential_fn, _ = posterior_estimator_based_potential(estimator, prior, None)
    else:
        estimator = classifier_nn("mlp", embedding_net_x=embedding)(theta, x)
        potential_fn, _ = ratio_estimator_based_potential(estimator, prior, None)
    return potential_fn, prior


@pytest.mark.parametrize("method", ["npe", "nre"])
def test_potential_counts_iid_trials_against_the_shape_of_one_observation(method):
    """An observation with several trials in its event shape is one iid trial.

    With a permutation-invariant embedding, one observation holds 5 trials. Without
    batch dimension, it must not be counted as 5 iid trials.
    """
    potential_fn, prior = _potential_with_trials_in_one_observation(method)

    theta_o = torch.tensor([[0.5, -0.5], [1.0, 1.0]])
    x_o = torch.stack([zeros(5, 2), ones(5, 2), -ones(5, 2)])

    def iid_potential(x_o: Tensor) -> Tensor:
        return potential_fn.bind(x_o, x_is_iid=True)(theta_o, track_gradients=False)

    singles = torch.stack([iid_potential(x_i.unsqueeze(0)) for x_i in x_o])
    assert torch.allclose(iid_potential(x_o[0]), singles[0])
    # iid trials: the sum of single-trial potentials, with the prior counted once.
    expected = singles.sum(0) - 2 * prior.log_prob(theta_o)
    assert torch.allclose(iid_potential(x_o), expected, atol=1e-5)


@pytest.mark.parametrize("method", ["npe", "nle", "nre"])
def test_x_that_contradicts_the_estimator_raises_a_clear_error(method):
    """A wrong shape of x, or theta and x batches that do not pair, raise."""
    prior = BoxUniform(-ones(2), ones(2))
    theta, x = prior.sample((20,)), prior.sample((20,))
    if method == "npe":
        estimator = posterior_nn("mdn")(theta, x)
        potential_fn, _ = posterior_estimator_based_potential(estimator, prior, None)
    elif method == "nle":
        estimator = likelihood_nn("mdn")(x, theta)
        potential_fn, _ = likelihood_estimator_based_potential(estimator, prior, None)
    else:
        estimator = classifier_nn("mlp")(theta, x)
        potential_fn, _ = ratio_estimator_based_potential(estimator, prior, None)
    posterior = RejectionPosterior(potential_fn, proposal=prior)

    for wrong_x in (zeros(1, 3), zeros(2, 3, 2)):
        with pytest.raises(ValueError, match=r"one observation has shape \(2,\)"):
            posterior.set_default_x(wrong_x)
    if method == "npe":
        with pytest.raises(ValueError, match=r"one observation has shape \(2,\)"):
            DirectPosterior(estimator, prior).sample_batched((2,), zeros(2, 3))
    unpaired = potential_fn.bind(zeros(3, 2), x_is_iid=False)
    with pytest.raises(ValueError, match="theta has 4 rows and x has 3 observations"):
        unpaired(zeros(4, 2))


@pytest.mark.mcmc
def test_mcmc_sample_batched_reads_one_observation_without_batch_dimension():
    """One observation with 5 trials is a batch of 1, not a batch of 5 trials."""
    potential_fn, prior = _potential_with_trials_in_one_observation("npe")
    posterior = MCMCPosterior(
        potential_fn, proposal=prior, method="slice_np_vectorized", warmup_steps=2
    )
    samples = posterior.sample_batched(
        (4,), x=zeros(5, 2), num_chains=2, show_progress_bars=False
    )
    assert samples.shape == (4, 1, 2)
