# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch.distributions import MultivariateNormal

from sbi.inference.posteriors import (
    DirectPosterior,
    EnsemblePosterior,
    ImportanceSamplingPosterior,
    RejectionPosterior,
    VIPosterior,
    VectorFieldPosterior,
)
from sbi.inference.potentials import likelihood_estimator_based_potential
from sbi.neural_nets import likelihood_nn, posterior_flow_nn, posterior_nn

MAP_KWARGS = dict(num_init_samples=20, num_to_optimize=5, num_iter=2)


@pytest.fixture
def toy():
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    theta = prior.sample((100,))
    x = theta + 0.1 * torch.randn_like(theta)
    return prior, theta, x


def _likelihood_potential(prior, theta, x):
    return likelihood_estimator_based_potential(
        likelihood_nn("mdn")(theta, x), prior, None
    )


def _untrained_posterior(kind, prior, theta, x):
    if kind == "direct":
        return DirectPosterior(posterior_nn("mdn")(theta, x), prior)
    if kind == "vector_field":
        return VectorFieldPosterior(posterior_flow_nn()(theta, x), prior)
    if kind == "ensemble":
        return EnsemblePosterior([
            _untrained_posterior("direct", prior, theta, x) for _ in range(2)
        ])
    potential_fn, theta_transform = _likelihood_potential(prior, theta, x)
    return RejectionPosterior(potential_fn, prior, theta_transform)


@pytest.mark.parametrize(
    "kind, init_method, match",
    [
        ("direct", "proposal", "no proposal distribution"),
        ("vector_field", "proposal", "no proposal distribution"),
        ("ensemble", "proposal", "no proposal distribution"),
        ("direct", "bogus", "Use 'posterior', or pass"),
        ("rejection", "bogus", "Use 'posterior', 'proposal', or pass"),
    ],
)
def test_map_init_method_errors_list_supported_options(kind, init_method, match, toy):
    posterior = _untrained_posterior(kind, *toy).set_default_x(torch.zeros(1, 2))
    with pytest.raises(ValueError, match=match):
        posterior.map(init_method=init_method, **MAP_KWARGS)


def test_importance_map_posterior_init_with_importance_method(toy):
    """`.sample()` returns (samples, weights) here, so the inits must come from SIR."""
    potential_fn, theta_transform = _likelihood_potential(*toy)
    posterior = ImportanceSamplingPosterior(
        potential_fn, toy[0], theta_transform, method="importance"
    ).set_default_x(torch.zeros(1, 2))
    theta_map = posterior.map(init_method="posterior", **MAP_KWARGS)
    assert theta_map.shape == (1, 2)


def test_amortized_vi_map_proposal_init_uses_amortized_q(toy):
    """The single-x `q` sits far from the prior; a MAP seeded from it stays there."""
    prior, theta, x = toy
    potential_fn, _ = _likelihood_potential(prior, theta, x)
    loc = torch.full((2,), 50.0, requires_grad=True)
    posterior = VIPosterior(
        potential_fn, prior, q=MultivariateNormal(loc, torch.eye(2)), parameters=[loc]
    )
    posterior.train_amortized(theta, x, max_num_iters=2, show_progress_bar=False)
    posterior.set_default_x(torch.zeros(1, 2))
    theta_map = posterior.map(init_method="proposal", **MAP_KWARGS)
    assert theta_map.abs().max() < 10
