# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from sbi.diagnostics import kl_divergence_mc
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.utils import BoxUniform

from .test_utils import PosteriorPotential


@pytest.mark.parametrize(
    "p, q",
    (
        (Normal(1.0, 0.5), Normal(0.0, 1.0)),
        (
            MultivariateNormal(ones(2), 0.5 * eye(2)),
            MultivariateNormal(zeros(2), eye(2)),
        ),
        (
            MultivariateNormal(zeros(3), 0.2 * eye(3)),
            MultivariateNormal(zeros(3), eye(3)),
        ),
    ),
)
def test_kl_divergence_mc_matches_analytic(p, q):
    estimate, sem = kl_divergence_mc(p, q, num_samples=20_000)

    assert sem > 0
    assert torch.abs(estimate - kl_divergence(p, q)) < 5 * sem


def test_kl_divergence_mc_is_zero_for_identical_distributions():
    p = MultivariateNormal(zeros(2), eye(2))
    estimate, sem = kl_divergence_mc(p, p, num_samples=1000)

    assert estimate == 0.0
    assert sem == 0.0


def test_kl_divergence_mc_raises_outside_support():
    p = MultivariateNormal(zeros(2), eye(2))
    q = BoxUniform(low=-0.01 * ones(2), high=0.01 * ones(2))

    with pytest.raises(ValueError, match="outside the support"):
        kl_divergence_mc(p, q, num_samples=500)


def test_kl_divergence_mc_refuses_batched_distribution():
    p = Normal(ones(2), 0.5 * ones(2))  # batch_shape (2,), not one joint distribution

    with pytest.raises(ValueError, match="Independent"):
        kl_divergence_mc(p, Normal(zeros(2), ones(2)), num_samples=100)


@pytest.mark.parametrize("unnormalized_arg", ("p", "q"))
def test_kl_divergence_mc_refuses_unnormalized_posterior(unnormalized_arg):
    prior = MultivariateNormal(zeros(2), eye(2))
    target = MultivariateNormal(ones(2), 0.5 * eye(2))
    unnormalized = MCMCPosterior(
        potential_fn=PosteriorPotential(target, prior), proposal=prior
    )
    p, q = (unnormalized, prior) if unnormalized_arg == "p" else (prior, unnormalized)

    with pytest.raises(NotImplementedError, match="c2st"):
        kl_divergence_mc(p, q, x=zeros(1, 2), num_samples=10)
