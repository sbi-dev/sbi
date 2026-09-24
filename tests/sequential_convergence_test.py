# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Dict

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from sbi.diagnostics import kl_divergence_mc
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.simulators.linear_gaussian import (
    true_posterior_linear_gaussian_mvn_prior,
)
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
