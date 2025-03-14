from typing import List

import pytest
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
from sbi.utils import BoxUniform

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

        assert dkl < max_dkl, (
            f"D-KL={dkl} is more than 2 stds above the average performance."
        )


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


@pytest.fixture(scope="module", params=["vp", "ve", "subvp"])
def sde_type(request):
    """Module-scoped fixture for SDE type."""
    return request.param


@pytest.fixture(scope="module", params=["gaussian", "uniform", None])
def prior_type(request):
    """Module-scoped fixture for prior type."""
    return request.param


@pytest.fixture(scope="module")
def npse_trained_model(sde_type, prior_type):
    """Module-scoped fixture that trains a score estimator for NPSE tests."""
    num_dim = 2
    num_simulations = 5000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    if prior_type == "gaussian" or (prior_type is None):
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        prior_npse = prior if prior_type is None else None
    elif prior_type == "uniform":
        prior = BoxUniform(-2 * ones(num_dim), 2 * ones(num_dim))
        prior_npse = prior

    # This check that our method to handle "general" priors works.
    # i.e. if NPSE does not get a proper passed by the user.
    inference = NPSE(prior_npse, show_progress_bars=True, sde_type=sde_type)

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    score_estimator = inference.append_simulations(theta, x).train(
        stop_after_epochs=200
    )

    return {
        "score_estimator": score_estimator,
        "inference": inference,
        "prior": prior,
        "likelihood_shift": likelihood_shift,
        "likelihood_cov": likelihood_cov,
        "prior_mean": prior_mean
        if prior_type == "gaussian" or prior_type is None
        else None,
        "prior_cov": prior_cov
        if prior_type == "gaussian" or prior_type is None
        else None,
        "num_dim": num_dim,
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "iid_method, num_trial",
    [
        pytest.param("fnpe", 3, id="fnpe-2trials"),
        pytest.param("gauss", 3, id="gauss-6trials"),
        pytest.param("auto_gauss", 8, id="auto_gauss-8trials"),
        pytest.param("auto_gauss", 16, id="auto_gauss-16trials"),
        pytest.param("jac_gauss", 8, id="jac_gauss-8trials"),
    ],
)
def test_npse_iid_inference(
    npse_trained_model, iid_method, num_trial, sde_type, prior_type
):
    """Test whether NPSE infers well a simple example with available ground truth."""
    num_samples = 1000

    # Extract data from fixture
    score_estimator = npse_trained_model["score_estimator"]
    inference = npse_trained_model["inference"]
    prior = npse_trained_model["prior"]
    likelihood_shift = npse_trained_model["likelihood_shift"]
    likelihood_cov = npse_trained_model["likelihood_cov"]
    prior_mean = npse_trained_model["prior_mean"]
    prior_cov = npse_trained_model["prior_cov"]
    num_dim = npse_trained_model["num_dim"]

    x_o = zeros(num_trial, num_dim)
    posterior = inference.build_posterior(score_estimator)
    posterior.set_default_x(x_o)
    samples = posterior.sample((num_samples,), iid_method=iid_method)

    if prior_type == "gaussian" or (prior_type is None):
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    elif prior_type == "uniform":
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o,
            likelihood_shift,
            likelihood_cov,
            prior,  # type: ignore
        )

    # Compute the c2st and assert it is near chance level of 0.5.
    # Some degradation is expected, also because posterior get tighter which
    # usually makes the c2st worse.
    check_c2st(
        samples,
        target_samples,
        alg=f"npse-{sde_type}-{prior_type}-{num_dim}-{iid_method}-{num_trial}iid-trials",
        tol=0.05 * min(num_trial, 8),
    )


@pytest.mark.slow
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

    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior().set_default_x(x_o)

    map_ = posterior.map(show_progress_bars=True, num_iter=5)

    assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5, "MAP is not close to GT."
