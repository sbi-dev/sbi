from typing import List

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import NPSE
from sbi.simulators import linear_gaussian
from sbi.simulators.linear_gaussian import (
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)

from .test_utils import check_c2st, get_dkl_gaussian_prior


# We always test num_dim and sample_with with defaults and mark the rests as slow.
@pytest.mark.parametrize(
    "sde_type, num_dim, prior_str, sample_with",
    [
        ("vp", 1, "gaussian", ["sde", "ode"]),
        ("vp", 3, "uniform", ["sde", "ode"]),
        ("vp", 3, "gaussian", ["sde", "ode"]),
        ("ve", 3, "uniform", ["sde", "ode"]),
        ("subvp", 3, "uniform", ["sde", "ode"]),
    ],
)
def test_c2st_npse_on_linearGaussian(
    sde_type, num_dim: int, prior_str: str, sample_with: List[str]
):
    """Test whether NPSE infers well a simple example with available ground truth."""

    x_o = zeros(1, num_dim)
    num_samples = 1000
    num_simulations = 10_000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    if prior_str == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    else:
        prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o,
            likelihood_shift,
            likelihood_cov,
            prior=prior,
            num_samples=num_samples,
        )

    inference = NPSE(prior, sde_type=sde_type, show_progress_bars=True)

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    score_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=100
    )
    # amortize the training when testing sample_with.
    for method in sample_with:
        posterior = inference.build_posterior(score_estimator, sample_with=method)
        posterior.set_default_x(x_o)
        samples = posterior.sample((num_samples,))

        # Compute the c2st and assert it is near chance level of 0.5.
        check_c2st(
            samples,
            target_samples,
            alg=f"npse-{sde_type or 'vp'}-{prior_str}-{num_dim}D-{method}",
        )

    # Checks for log_prob()
    if prior_str == "gaussian":
        # For the Gaussian prior, we compute the KLd between ground truth and
        # posterior.
        dkl = get_dkl_gaussian_prior(
            posterior,
            x_o[0],
            likelihood_shift,
            likelihood_cov,
            prior_mean,
            prior_cov,
        )

        max_dkl = 0.15

        assert (
            dkl < max_dkl
        ), f"D-KL={dkl} is more than 2 stds above the average performance."


def test_c2st_npse_on_linearGaussian_different_dims():
    """Test NPE on linear Gaussian with different theta and x dimensionality."""

    theta_dim = 3
    x_dim = 2
    discard_dims = theta_dim - x_dim

    x_o = zeros(1, x_dim)
    num_samples = 1000
    num_simulations = 2000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(x_dim)
    likelihood_cov = 0.3 * eye(x_dim)

    prior_mean = zeros(theta_dim)
    prior_cov = eye(theta_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    target_samples = samples_true_posterior_linear_gaussian_mvn_prior_different_dims(
        x_o,
        likelihood_shift,
        likelihood_cov,
        prior_mean,
        prior_cov,
        num_discarded_dims=discard_dims,
        num_samples=num_samples,
    )

    def simulator(theta):
        return linear_gaussian(
            theta,
            likelihood_shift,
            likelihood_cov,
            num_discarded_dims=discard_dims,
        )

    # Test whether prior can be `None`.
    inference = NPSE(prior=None)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    # Test whether we can stop and resume.
    inference.append_simulations(theta, x).train(
        max_num_epochs=10, training_batch_size=100
    )
    inference.train(
        resume_training=True, force_first_round_loss=True, training_batch_size=100
    )
    posterior = inference.build_posterior().set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="npse_different_dims_and_resume_training")


@pytest.mark.xfail(
    reason="iid_bridge not working.",
    raises=NotImplementedError,
    strict=True,
    match="Score accumulation*",
)
@pytest.mark.parametrize("num_trials", [2, 10])
def test_npse_iid_inference(num_trials):
    """Test whether NPSE infers well a simple example with available ground truth."""

    num_dim = 2
    x_o = zeros(num_trials, num_dim)
    num_samples = 1000
    num_simulations = 3000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    inference = NPSE(prior, show_progress_bars=True)

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    score_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=100,
    )
    posterior = inference.build_posterior(score_estimator)
    posterior.set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(
        samples, target_samples, alg=f"npse-vp-gaussian-1D-{num_trials}iid-trials"
    )


@pytest.mark.slow
@pytest.mark.xfail(
    raises=NotImplementedError,
    reason="MAP optimization via score not working accurately.",
)
def test_npse_map():
    num_dim = 2
    x_o = zeros(num_dim)
    num_simulations = 3000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    inference = NPSE(prior, show_progress_bars=True)

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference.append_simulations(theta, x).train(
        training_batch_size=100, max_num_epochs=10
    )
    posterior = inference.build_posterior().set_default_x(x_o)

    map_ = posterior.map(show_progress_bars=True)

    assert torch.allclose(map_, gt_posterior.mean, atol=0.2), "MAP is not close to GT."
