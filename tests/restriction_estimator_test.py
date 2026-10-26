# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import math

import torch
from torch.distributions import Independent, Normal

from sbi.utils.restriction_estimator import RestrictedPrior


def test_restricted_prior_sir_strictly_enforces_restriction():
    """Test that SIR sampling from RestrictedPrior strictly satisfies restriction."""
    torch.manual_seed(42)
    dim = 2
    prior = Independent(Normal(torch.zeros(dim), torch.ones(dim)), 1)
    proposal = Independent(Normal(torch.zeros(dim), torch.ones(dim)), 1)

    # Restriction: positive quadrant theta_0 >= 0 and theta_1 >= 0
    def accept_reject_fn(theta: torch.Tensor) -> torch.Tensor:
        return (theta[..., 0] >= 0.0) & (theta[..., 1] >= 0.0)

    restricted_prior = RestrictedPrior(
        prior, accept_reject_fn, posterior=proposal, sample_with="sir"
    )

    samples = restricted_prior.sample((200,), oversampling_factor=64)
    accepted = accept_reject_fn(samples)
    assert bool(torch.all(accepted)), (
        f"Returned SIR samples violate the restriction! "
        f"{(~accepted).sum().item()} violations found."
    )


def test_restricted_prior_sir_reweights_prior_correctly():
    """Test that SIR sampling incorporates prior log-prob and matches moments."""
    torch.manual_seed(42)
    dim = 2
    prior = Independent(Normal(torch.zeros(dim), torch.ones(dim)), 1)
    # Shifted proposal to ensure importance reweighting by prior/proposal is needed
    proposal = Independent(
        Normal(torch.tensor([0.5, 0.0]), torch.tensor([1.0, 1.0])), 1
    )

    # Restriction: theta_0 >= 0 (half-normal distribution on theta_0)
    def accept_reject_fn(theta: torch.Tensor) -> torch.Tensor:
        return theta[..., 0] >= 0.0

    restricted_prior = RestrictedPrior(
        prior, accept_reject_fn, posterior=proposal, sample_with="sir"
    )

    samples = restricted_prior.sample((5_000,), oversampling_factor=64)
    assert bool(torch.all(accept_reject_fn(samples))), (
        "Returned SIR samples violate restriction."
    )

    # Theoretical mean of standard half-normal is sqrt(2/pi) approx 0.7979
    expected_mean = math.sqrt(2.0 / math.pi)
    sample_mean = samples[:, 0].mean().item()
    assert abs(sample_mean - expected_mean) < 0.05, (
        f"Sample mean {sample_mean} deviates from expected truncated prior mean "
        f"{expected_mean}."
    )


def test_restricted_prior_sir_vectorized_accept_reject_shape():
    """Test that accept_reject_fn returning (N, 1) does not cause broadcasting bugs."""
    torch.manual_seed(42)
    dim = 2
    prior = Independent(Normal(torch.zeros(dim), torch.ones(dim)), 1)
    proposal = prior

    # Returns (N, 1) instead of (N,)
    def accept_reject_fn(theta: torch.Tensor) -> torch.Tensor:
        return theta[..., 0:1] >= 0.0

    restricted_prior = RestrictedPrior(
        prior, accept_reject_fn, posterior=proposal, sample_with="sir"
    )

    samples = restricted_prior.sample((50,), oversampling_factor=32)
    assert samples.shape == (50, 2)
    assert bool(torch.all(accept_reject_fn(samples))), (
        "Returned SIR samples violate restriction."
    )


def test_restricted_prior_sir_dtype_preservation():
    """Test that RestrictedPrior SIR works with float64 and preserves dtype."""
    torch.manual_seed(42)
    dim = 2
    prior = Independent(
        Normal(
            torch.zeros(dim, dtype=torch.float64),
            torch.ones(dim, dtype=torch.float64),
        ),
        1,
    )
    proposal = prior

    def accept_reject_fn(theta: torch.Tensor) -> torch.Tensor:
        return theta[..., 0] >= 0.0

    restricted_prior = RestrictedPrior(
        prior, accept_reject_fn, posterior=proposal, sample_with="sir"
    )

    samples = restricted_prior.sample((50,), oversampling_factor=32)
    assert samples.dtype == torch.float64
    assert bool(torch.all(accept_reject_fn(samples))), (
        "Returned SIR samples violate restriction."
    )


def test_restricted_prior_sir_uses_oversampling_factor():
    """Test that SIR sampling evaluates oversampling_factor candidates per sample."""
    prior = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)
    num_candidates = []

    def accept_reject_fn(theta: torch.Tensor) -> torch.Tensor:
        num_candidates.append(theta.shape[0])
        return torch.ones(theta.shape[0], dtype=torch.bool)

    restricted_prior = RestrictedPrior(
        prior, accept_reject_fn, posterior=prior, sample_with="sir"
    )
    restricted_prior.sample((10,), oversampling_factor=4)

    assert sum(num_candidates) == 40
