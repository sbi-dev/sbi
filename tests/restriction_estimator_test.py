# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributions import Independent, Normal

from sbi.samplers.importance.sir import sampling_importance_resampling
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


def test_restricted_prior_sir_all_candidates_rejected_terminates_with_error():
    """Test that all rejected candidates terminate within retry bound and raise."""
    prior = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)

    def always_reject(theta: torch.Tensor) -> torch.Tensor:
        return torch.zeros(theta.shape[0], dtype=torch.bool)

    restricted_prior = RestrictedPrior(
        prior, always_reject, posterior=prior, sample_with="sir"
    )

    with pytest.raises(
        RuntimeError, match="failed to produce valid samples after 3 attempts"
    ):
        restricted_prior.sample((5,), oversampling_factor=4, max_retries=3)


def test_sir_all_neginf_log_weights_avoids_nans_and_raises():
    """Test that potential returning -inf raises RuntimeError without producing NaNs."""
    proposal = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)

    def neginf_potential(theta: torch.Tensor) -> torch.Tensor:
        return torch.full((theta.shape[0],), float("-inf"))

    with pytest.raises(
        RuntimeError, match="failed to produce valid samples after 2 attempts"
    ):
        sampling_importance_resampling(
            potential_fn=neginf_potential,
            proposal=proposal,
            num_samples=4,
            num_candidate_samples=8,
            max_retries=2,
        )


def test_sir_invalid_max_retries():
    """Test that invalid max_retries raises ValueError."""
    proposal = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)

    with pytest.raises(ValueError, match="max_retries must be at least 1"):
        sampling_importance_resampling(
            potential_fn=proposal.log_prob,
            proposal=proposal,
            num_samples=1,
            max_retries=0,
        )


def test_restricted_prior_sir_partial_rejection_success():
    """Test that partial rejection discards invalid candidates and succeeds."""
    torch.manual_seed(42)
    dim = 2
    prior = Independent(Normal(torch.zeros(dim), torch.ones(dim)), 1)

    # 50% rejection rate
    def accept_reject_fn(theta: torch.Tensor) -> torch.Tensor:
        return theta[..., 0] >= 0.0

    restricted_prior = RestrictedPrior(
        prior, accept_reject_fn, posterior=prior, sample_with="sir"
    )

    samples = restricted_prior.sample((50,), oversampling_factor=8)
    assert samples.shape == (50, 2)
    assert bool(torch.all(accept_reject_fn(samples)))


def test_restricted_prior_sir_recovers_after_initial_empty_attempt():
    """Test that retry loop continues and succeeds when a subsequent attempt yields
    valid samples.
    """
    prior = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)
    call_count = 0

    def conditional_accept(theta: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        # Reject everything on the first call, accept all on subsequent calls
        if call_count == 1:
            return torch.zeros(theta.shape[0], dtype=torch.bool)
        return torch.ones(theta.shape[0], dtype=torch.bool)

    restricted_prior = RestrictedPrior(
        prior, conditional_accept, posterior=prior, sample_with="sir"
    )

    samples = restricted_prior.sample((10,), oversampling_factor=4, max_retries=5)
    assert samples.shape == (10, 2)
    assert call_count >= 2


def test_sir_progress_bar_closes_on_exception():
    """Test that the SIR progress bar is reliably closed when an exception occurs."""
    proposal = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)

    def failing_potential(theta: torch.Tensor) -> torch.Tensor:
        raise ValueError("Simulated failure during potential evaluation")

    with patch("sbi.samplers.importance.sir.tqdm") as mock_tqdm:
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        with pytest.raises(
            ValueError, match="Simulated failure during potential evaluation"
        ):
            sampling_importance_resampling(
                potential_fn=failing_potential,
                proposal=proposal,
                num_samples=2,
                show_progress_bars=True,
            )

        mock_pbar.close.assert_called_once()


def test_sir_max_sampling_time_timeout():
    """Test that SIR aborts and raises RuntimeError when max_sampling_time is
    exceeded.
    """
    proposal = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)

    with (
        patch("time.time", side_effect=[0.0, 10.0, 20.0]),
        pytest.raises(RuntimeError, match="exceeded max_sampling_time of 1.0s"),
    ):
        sampling_importance_resampling(
            potential_fn=proposal.log_prob,
            proposal=proposal,
            num_samples=10,
            max_sampling_time=1.0,
        )


def test_restricted_prior_sir_max_sampling_time_timeout():
    """Test that RestrictedPrior.sample respects max_sampling_time under SIR."""
    prior = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)
    restricted_prior = RestrictedPrior(
        prior,
        lambda theta: torch.ones(theta.shape[0], dtype=torch.bool),
        posterior=prior,
        sample_with="sir",
    )

    with (
        patch("time.time", side_effect=[0.0, 5.0, 10.0]),
        pytest.raises(RuntimeError, match="exceeded max_sampling_time of 0.5s"),
    ):
        restricted_prior.sample((10,), max_sampling_time=0.5)


def test_sir_invalid_max_sampling_time():
    """Test that non-positive max_sampling_time raises ValueError."""
    proposal = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)

    with pytest.raises(ValueError, match="max_sampling_time must be positive"):
        sampling_importance_resampling(
            potential_fn=proposal.log_prob,
            proposal=proposal,
            num_samples=1,
            max_sampling_time=0.0,
        )
