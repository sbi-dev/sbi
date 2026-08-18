# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Dict

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from sbi.diagnostics import SequentialConvergenceTracker, kl_divergence_mc
from sbi.inference import NPE_C
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.simulators.linear_gaussian import (
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils import BoxUniform

from .test_utils import PosteriorPotential, get_dkl_gaussian_prior


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
    """The MC estimate should agree with the closed-form KL for Gaussians."""
    estimate, sem = kl_divergence_mc(p, q, num_samples=20_000)
    exact = kl_divergence(p, q)

    assert torch.isfinite(estimate) and sem > 0
    assert torch.abs(estimate - exact) < 5 * sem, (
        f"MC estimate {estimate:.4f} +/- {sem:.4f} is too far from the exact "
        f"value {float(exact):.4f}."
    )


def test_kl_divergence_mc_is_exactly_zero_for_identical_distributions():
    """Evaluating both densities on the same samples cancels term by term.

    KL(p || p) is therefore exactly zero rather than zero up to MC noise, which
    is what makes the increment trustworthy when two rounds barely differ.
    """
    p = MultivariateNormal(zeros(2), eye(2))
    estimate, sem = kl_divergence_mc(p, p, num_samples=5000)

    assert estimate == pytest.approx(0.0, abs=1e-6)
    assert sem == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_mc_standard_error_shrinks_with_samples():
    """The standard error should fall roughly as 1/sqrt(num_samples)."""
    p = MultivariateNormal(ones(2), 0.5 * eye(2))
    q = MultivariateNormal(zeros(2), eye(2))

    _, sem_small = kl_divergence_mc(p, q, num_samples=1000)
    _, sem_large = kl_divergence_mc(p, q, num_samples=16_000)

    assert sem_large < sem_small


def test_kl_divergence_mc_reuses_provided_samples():
    """Passing `p_samples` should bypass sampling and be reproducible."""
    p = MultivariateNormal(zeros(2), eye(2))
    q = MultivariateNormal(ones(2), eye(2))
    samples = p.sample((2000,))

    first, _ = kl_divergence_mc(p, q, p_samples=samples)
    second, _ = kl_divergence_mc(p, q, p_samples=samples)

    assert torch.allclose(first, second)


def test_kl_divergence_mc_raises_outside_support():
    """Samples outside q's support make the divergence infinite."""
    p = MultivariateNormal(zeros(2), eye(2))
    q = BoxUniform(low=-0.01 * ones(2), high=0.01 * ones(2))

    with pytest.raises(ValueError, match="outside the support"):
        kl_divergence_mc(p, q, num_samples=500)


def _unnormalized_posterior(gaussian_setup: Dict, x_o):
    """An `MCMCPosterior` wrapping a closed-form target, for guard tests."""
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o,
        gaussian_setup["likelihood_shift"],
        gaussian_setup["likelihood_cov"],
        gaussian_setup["prior_mean"],
        gaussian_setup["prior_cov"],
    )
    return MCMCPosterior(
        potential_fn=PosteriorPotential(gt_posterior, gaussian_setup["prior"]),
        proposal=gaussian_setup["prior"],
    )


def test_kl_divergence_mc_raises_for_unnormalized_posterior(gaussian_setup: Dict):
    """Posteriors that only give the potential must be refused, not silently used.

    The guard keys on the warning these posteriors emit rather than on a list of
    classes, so this test is what pins that warning's wording in place. It is
    exercised at `log_prob`, which is where the guard lives -- sampling from an
    `MCMCPosterior` is neither needed nor cheap.
    """
    prior = gaussian_setup["prior"]
    x_o = zeros(1, gaussian_setup["num_dim"])
    unnormalized = _unnormalized_posterior(gaussian_setup, x_o)

    # As the denominator: sampling comes from the prior, so no MCMC is run.
    with pytest.raises(NotImplementedError, match="c2st"):
        kl_divergence_mc(prior, unnormalized, x_o=x_o, num_samples=10)

    # As the numerator, with samples supplied so that no MCMC is run.
    with pytest.raises(NotImplementedError, match="c2st"):
        kl_divergence_mc(unnormalized, prior, x_o=x_o, p_samples=prior.sample((10,)))


def test_tracker_raises_before_sampling_for_unnormalized_posterior(
    gaussian_setup: Dict,
):
    """The tracker probes on a cheap prior sample, so MCMC never starts."""
    prior = gaussian_setup["prior"]
    x_o = zeros(1, gaussian_setup["num_dim"])
    unnormalized = _unnormalized_posterior(gaussian_setup, x_o)

    tracker = SequentialConvergenceTracker(prior, x_o, num_samples=10)
    with pytest.raises(NotImplementedError, match="c2st"):
        tracker.update(unnormalized)

    assert tracker.history == []


def test_unnormalized_guard_fires_after_warning_already_shown(gaussian_setup: Dict):
    """The guard must survive Python's once-per-location warning registry.

    A plain `warnings.warn` is only shown the first time it is reached, so a
    guard that merely listened for the warning could silently stop firing once
    the user had called `log_prob()` themselves. Escalating via
    `filterwarnings` invalidates that registry, so the error is still raised.
    """
    prior = gaussian_setup["prior"]
    x_o = zeros(1, gaussian_setup["num_dim"])
    unnormalized = _unnormalized_posterior(gaussian_setup, x_o)

    # Trigger the warning once, exactly as a user calling `log_prob()` would.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unnormalized.log_prob(prior.sample((2,)), x=x_o)

    with pytest.raises(NotImplementedError, match="c2st"):
        kl_divergence_mc(prior, unnormalized, x_o=x_o, num_samples=10)


def test_tracker_first_round_has_no_increment(gaussian_setup: Dict):
    """Round 0 has no predecessor, so increment and ratio are undefined."""
    prior = gaussian_setup["prior"]
    x_o = zeros(1, gaussian_setup["num_dim"])
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o,
        gaussian_setup["likelihood_shift"],
        gaussian_setup["likelihood_cov"],
        gaussian_setup["prior_mean"],
        gaussian_setup["prior_cov"],
    )

    tracker = SequentialConvergenceTracker(prior, x_o, num_samples=2000)
    record = tracker.update(gt_posterior)

    assert record["round"] == 0
    assert record["increment"] is None
    assert record["ratio"] is None
    assert record["compression"] > 0
    assert not record["uninformative"]


def test_tracker_compression_matches_analytic(gaussian_setup: Dict):
    """Feeding the true posterior should recover the exact prior-to-posterior KL."""
    prior = gaussian_setup["prior"]
    x_o = zeros(1, gaussian_setup["num_dim"])
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o,
        gaussian_setup["likelihood_shift"],
        gaussian_setup["likelihood_cov"],
        gaussian_setup["prior_mean"],
        gaussian_setup["prior_cov"],
    )
    exact = float(kl_divergence(gt_posterior, prior))

    tracker = SequentialConvergenceTracker(prior, x_o, num_samples=20_000)
    record = tracker.update(gt_posterior)

    assert abs(record["compression"] - exact) < 5 * record["compression_sem"]


def test_tracker_increment_is_zero_for_repeated_posterior(gaussian_setup: Dict):
    """Passing the same posterior twice means the round changed nothing."""
    prior = gaussian_setup["prior"]
    x_o = zeros(1, gaussian_setup["num_dim"])
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o,
        gaussian_setup["likelihood_shift"],
        gaussian_setup["likelihood_cov"],
        gaussian_setup["prior_mean"],
        gaussian_setup["prior_cov"],
    )

    tracker = SequentialConvergenceTracker(prior, x_o, num_samples=5000)
    tracker.update(gt_posterior)
    record = tracker.update(gt_posterior)

    assert record["increment"] == pytest.approx(0.0, abs=1e-5)
    assert record["ratio"] == pytest.approx(0.0, abs=1e-5)
    assert len(tracker.history) == 2


def test_tracker_flags_uninformative_estimate(gaussian_setup: Dict):
    """An estimate indistinguishable from the prior is flagged, not thresholded."""
    prior = gaussian_setup["prior"]
    x_o = zeros(1, gaussian_setup["num_dim"])

    tracker = SequentialConvergenceTracker(prior, x_o, num_samples=5000)
    first = tracker.update(prior)
    second = tracker.update(prior)

    assert first["uninformative"]
    assert first["compression"] == pytest.approx(0.0, abs=1e-5)
    # The ratio has a vanishing denominator here, so it must not be reported.
    assert second["ratio"] != second["ratio"]  # NaN


@pytest.mark.slow
def test_tracker_tracks_true_error_across_rounds(gaussian_setup: Dict):
    """On a multi-round NPE run the diagnostics should be finite and informative."""
    prior = gaussian_setup["prior"]
    simulator = gaussian_setup["simulator"]
    num_dim = gaussian_setup["num_dim"]
    x_o = zeros(1, num_dim)

    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o,
        gaussian_setup["likelihood_shift"],
        gaussian_setup["likelihood_cov"],
        gaussian_setup["prior_mean"],
        gaussian_setup["prior_cov"],
    )
    exact_compression = float(kl_divergence(gt_posterior, prior))

    inference = NPE_C(prior=prior, show_progress_bars=False)
    tracker = SequentialConvergenceTracker(prior, x_o, num_samples=4000)
    proposal = prior
    true_errors = []

    for _ in range(3):
        theta = proposal.sample((1000,))
        x = simulator(theta)
        inference.append_simulations(theta, x, proposal=proposal).train(
            show_train_summary=False
        )
        posterior = inference.build_posterior().set_default_x(x_o)

        tracker.update(posterior)
        true_errors.append(
            float(
                get_dkl_gaussian_prior(
                    posterior,
                    x_o[0],
                    gaussian_setup["likelihood_shift"],
                    gaussian_setup["likelihood_cov"],
                    gaussian_setup["prior_mean"],
                    gaussian_setup["prior_cov"],
                    num_samples=200,
                )
            )
        )
        proposal = posterior

    assert len(tracker.history) == 3
    assert all(
        record["increment"] is None or record["increment"] >= 0
        for record in tracker.history
    )
    assert tracker.history[0]["increment"] is None
    assert not any(record["uninformative"] for record in tracker.history)

    # The estimated compression should be in the right ballpark throughout: the
    # posterior is learned, so allow a generous margin around the exact value.
    for compression in tracker.compressions:
        assert abs(compression - exact_compression) < 0.5, (
            f"compression {compression:.3f} far from exact {exact_compression:.3f}"
        )

    # The final estimate should be a decent posterior; if it is, the diagnostic
    # should not be reporting a large remaining increment.
    assert true_errors[-1] < 0.5
    assert tracker.ratios[-1] < 0.25
