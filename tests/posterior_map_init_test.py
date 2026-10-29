# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch.distributions import MultivariateNormal

from sbi.inference.posteriors import (
    DirectPosterior,
    EnsemblePosterior,
    RejectionPosterior,
    VectorFieldPosterior,
)
from sbi.inference.potentials import likelihood_estimator_based_potential
from sbi.neural_nets import likelihood_nn, posterior_flow_nn, posterior_nn

MAP_KWARGS = dict(num_init_samples=20, num_to_optimize=5, num_iter=2)


@pytest.fixture(scope="module")
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
        ("vector_field", "bogus", "Use 'posterior', or pass"),
        ("ensemble", "bogus", "Use 'posterior', or pass"),
        ("rejection", "bogus", "Use 'posterior', 'proposal', or pass"),
    ],
)
def test_map_init_method_errors_list_supported_options(kind, init_method, match, toy):
    posterior = _untrained_posterior(kind, *toy).set_default_x(torch.zeros(1, 2))
    with pytest.raises(ValueError, match=match):
        posterior.map(init_method=init_method, **MAP_KWARGS)
