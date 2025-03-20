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

IID_METHODS = ["fnpe", "gauss", "auto_gauss", "jac_gauss"]
SAMPLING_METHODS = ["sde", "ode"]
SDE_TYPE = ["vp", "ve", "subvp"]
PRIOR_TYPE = ["gaussian", "uniform", None]
NUM_TRIAL = [3, 8, 16]
NUM_DIM = [1, 2, 3]


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


@pytest.mark.parametrize("sde_type", SDE_TYPE)
@pytest.mark.parametrize("prior_type", PRIOR_TYPE)
@pytest.mark.parametrize("iid_method", IID_METHODS)
@pytest.mark.parametrize("num_trial", NUM_TRIAL)
@pytest.mark.parametrize("num_dim", NUM_DIM)
@pytest.mark.parametrize("sample_with", SAMPLING_METHODS)
def test_npse_snapshot(
    sde_type,
    prior_type,
    iid_method,
    num_trial,
    num_dim,
    sample_with,
    ndarrays_regression,
):
    num_simulations, num_samples = 10, 10
    stop_after_epochs = 1

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
        stop_after_epochs=stop_after_epochs
    )

    posterior = inference.build_posterior(score_estimator, sample_with=sample_with)
    posterior.set_default_x(x_o)
    samples = posterior.sample((num_samples,), iid_method=iid_method)
    ndarrays_regression.check(
        {'values': samples.numpy()}, default_tolerance=dict(atol=1e-3, rtol=1e-2)
    )
