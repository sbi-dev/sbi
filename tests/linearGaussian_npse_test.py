from typing import List

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import NPSE
from sbi.simulators import linear_gaussian
from sbi.simulators.linear_gaussian import (
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)

from .test_utils import check_c2st, get_dkl_gaussian_prior


def _initialize_experiment(
    num_dim: int, num_samples: int, num_simulations: int, prior_type: str
):
    """Set up priors, generate target samples, and create simulation data."""
    if prior_type in {"gaussian", None}:
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    elif prior_type == "uniform":
        prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
        prior_mean, prior_cov = None, None
    else:
        raise NotImplementedError(f"Unsupported prior type: {prior_type}")

    x_o = zeros((1, num_dim))
    likelihood_shift = -ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    if prior_type in {"gaussian", None}:
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    else:  # prior_type == "uniform"
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o, likelihood_shift, likelihood_cov, prior=prior, num_samples=num_samples
        )

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)
    prior = None if prior_type is None else prior

    return (
        prior,
        target_samples,
        theta,
        x,
        x_o,
        likelihood_shift,
        likelihood_cov,
        prior_mean,
        prior_cov,
    )


@pytest.fixture(scope="module")
def npse_trained_model(request):
    """Module-scoped fixture that trains a score estimator for NPSE tests."""
    num_dim, num_simulations, num_samples = 2, 5_000, 1_000
    sde_type, prior_type = request.param

    (
        prior,
        target_samples,
        theta,
        x,
        x_o,
        likelihood_shift,
        likelihood_cov,
        prior_mean,
        prior_cov,
    ) = _initialize_experiment(num_dim, num_samples, num_simulations, prior_type)

    inference = NPSE(prior, show_progress_bars=True, sde_type=sde_type)

    score_estimator = inference.append_simulations(theta, x).train(
        stop_after_epochs=200
    )

    return {
        "score_estimator": score_estimator,
        "inference": inference,
        "prior": prior,
        "likelihood_shift": likelihood_shift,
        "likelihood_cov": likelihood_cov,
        "prior_mean": prior_mean,
        "prior_cov": prior_cov,
        "num_dim": num_dim,
        "x_o": x_o,
        "sde_type": sde_type,
        "prior_type": prior_type,
        "target_samples": target_samples,
        "num_samples": num_samples,
    }


# We always test num_dim and sample_with with defaults and mark the rests as slow.
@pytest.mark.slow
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
    num_samples = 1000
    num_simulations = 10_000

    (
        prior,
        target_samples,
        theta,
        x,
        x_o,
        likelihood_shift,
        likelihood_cov,
        prior_mean,
        prior_cov,
    ) = _initialize_experiment(num_dim, num_samples, num_simulations, prior_str)

    inference = NPSE(prior, sde_type=sde_type, show_progress_bars=True)

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


@pytest.mark.parametrize(
    "npse_trained_model",
    [
        pytest.param(("ve", None), id="ve-None"),
        pytest.param(("vp", "gaussian"), id="vp-gaussian"),
        pytest.param(("vp", "uniform"), id="vp-uniform", marks=pytest.mark.slow),
        pytest.param(("vp", None), id="vp-None", marks=pytest.mark.slow),
        pytest.param(("ve", "gaussian"), id="ve-gaussian", marks=pytest.mark.slow),
        pytest.param(("ve", "uniform"), id="ve-uniform", marks=pytest.mark.slow),
        pytest.param(
            ("subvp", "gaussian"), id="subvp-gaussian", marks=pytest.mark.slow
        ),
        pytest.param(("subvp", "uniform"), id="subvp-uniform", marks=pytest.mark.slow),
        pytest.param(("subvp", None), id="subvp-None", marks=pytest.mark.slow),
    ],
    indirect=True,  # So pytest knows to pass to the fixture
)
@pytest.mark.parametrize(
    "iid_method, num_trial",
    [
        pytest.param("fnpe", 3, id="fnpe-2trials", marks=pytest.mark.slow),
        pytest.param("gauss", 3, id="gauss-6trials", marks=pytest.mark.slow),
        pytest.param("auto_gauss", 8, id="auto_gauss-8trials"),
        pytest.param(
            "auto_gauss", 16, id="auto_gauss-16trials", marks=pytest.mark.slow
        ),
        pytest.param("jac_gauss", 8, id="jac_gauss-8trials", marks=pytest.mark.slow),
    ],
)
def test_npse_iid_inference(npse_trained_model, iid_method, num_trial):
    """Test whether NPSE infers well a simple example with available ground truth."""

    score_estimator = npse_trained_model["score_estimator"]
    inference = npse_trained_model["inference"]
    num_dim = npse_trained_model["num_dim"]
    sde_type = npse_trained_model["sde_type"]
    prior_type = npse_trained_model["prior_type"]
    target_samples = npse_trained_model["target_samples"]
    x_o = npse_trained_model["x_o"]
    num_samples = npse_trained_model["num_samples"]

    posterior = inference.build_posterior(score_estimator)
    posterior.set_default_x(x_o)
    samples = posterior.sample((num_samples,), iid_method=iid_method)

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
@pytest.mark.parametrize("npse_trained_model", [("vp", "gaussian")], indirect=True)
def test_npse_map(npse_trained_model):
    x_o = npse_trained_model["x_o"]
    inference = npse_trained_model["inference"]
    prior_mean = npse_trained_model["prior_mean"]
    prior_cov = npse_trained_model["prior_cov"]
    likelihood_shift = npse_trained_model["likelihood_shift"]
    likelihood_cov = npse_trained_model["likelihood_cov"]

    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )

    posterior = inference.build_posterior().set_default_x(x_o)

    map_ = posterior.map(show_progress_bars=True, num_iter=5)

    assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5, "MAP is not close to GT."
