from dataclasses import dataclass
from itertools import product
from typing import Optional

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import NPSE, ScorePosterior
from sbi.simulators import linear_gaussian
from sbi.simulators.linear_gaussian import (
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)

from .test_utils import check_c2st, get_dkl_gaussian_prior

IID_METHODS = ["fnpe", "gauss", "auto_gauss", "jac_gauss"]
SAMPLING_METHODS = ["sde", "ode"]


@dataclass
class NpseTrainingTestCase:
    num_dim: int
    prior_type: Optional[str]
    sde_type: str


@dataclass
class NpseSamplingTestCase:
    iid_method: str
    sampling_method: str
    num_trials: int


def _get_npse_prior(prior_type, num_dim):
    prior_mean = None
    prior_cov = None
    if prior_type == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    elif prior_type == "uniform":
        prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
    else:
        raise NotImplementedError(f"Unsupported prior type: {prior_type}")

    return prior, prior_mean, prior_cov


def _get_npse_training_data(num_simulations, num_dim, prior):
    likelihood_shift = -ones(num_dim)
    likelihood_cov = eye(num_dim)
    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)
    return theta, x, likelihood_shift, likelihood_cov


def _get_npse_target_samples(
    num_samples, num_dim, prior_type, prior, prior_mean, prior_cov, x_o
):
    likelihood_shift = -ones(num_dim)
    likelihood_cov = eye(num_dim)
    if prior_type == "gaussian":
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o,
            likelihood_shift,
            likelihood_cov,
            prior_mean,
            prior_cov,
        )
        target_samples = gt_posterior.sample((num_samples,))
    elif prior_type == "uniform":
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o,
            likelihood_shift,
            likelihood_cov,
            prior=prior,
            num_samples=num_samples,
        )
    else:
        raise NotImplementedError(f"Unsupported prior type: {prior_type}")
    return target_samples, likelihood_shift, likelihood_cov


def _train_npse(
    sde_type,
    num_dim,
    num_simulations,
    prior,
    training_batch_size=None,
    stop_after_epochs=None,
    max_num_epochs=None,
):
    inference = NPSE(show_progress_bars=True, sde_type=sde_type)
    theta, x, _, _ = _get_npse_training_data(
        num_simulations=num_simulations, num_dim=num_dim, prior=prior
    )

    kwargs = {}
    if stop_after_epochs is not None:
        kwargs["stop_after_epochs"] = stop_after_epochs
    if training_batch_size is not None:
        kwargs["training_batch_size"] = training_batch_size
    if max_num_epochs is not None:
        kwargs["max_num_epochs"] = max_num_epochs

    return inference.append_simulations(theta, x).train(**kwargs)


training_test_cases_gaussian = [
    pytest.param(
        NpseTrainingTestCase(1, "gaussian", "vp"), id="gaussian_prior-dim_1-vp"
    ),
    pytest.param(
        NpseTrainingTestCase(1, "gaussian", "ve"), id="gaussian_prior-dim_1-ve"
    ),
    pytest.param(
        NpseTrainingTestCase(2, "gaussian", "ve"), id="gaussian_prior-dim_2-ve"
    ),
    pytest.param(
        NpseTrainingTestCase(2, "gaussian", "vp"), id="gaussian_prior-dim_2-vp"
    ),
    pytest.param(
        NpseTrainingTestCase(2, "gaussian", "subvp"), id="gaussian_prior-dim_2-subvp"
    ),
]

training_test_cases_uniform = [
    pytest.param(NpseTrainingTestCase(3, "uniform", "ve"), id="uniform_prior-dim_3-ve"),
    pytest.param(NpseTrainingTestCase(3, "uniform", "vp"), id="uniform_prior-dim_3-vp"),
    pytest.param(
        NpseTrainingTestCase(3, "uniform", "subvp"), id="uniform_prior-dim_3-subvp"
    ),
]


training_test_cases_all = training_test_cases_gaussian + training_test_cases_uniform

sampling_test_cases_1_trial = [
    pytest.param(
        NpseSamplingTestCase(iid, sampling, 1), id=f"{iid}-{sampling}-trials_1"
    )
    for iid, sampling in product(IID_METHODS, SAMPLING_METHODS)
]

sampling_test_cases_n_trials = [
    pytest.param(NpseSamplingTestCase("fnpe", "sde", 3), id="fnpe-sde-trials_3"),
    pytest.param(NpseSamplingTestCase("gauss", "sde", 3), id="gauss-sde-trials_3"),
    pytest.param(
        NpseSamplingTestCase("auto_gauss", "sde", 3), id="auto_gauss-sde-trials_3"
    ),
    pytest.param(
        NpseSamplingTestCase("jac_gauss", "sde", 3), id="jac_gauss-sde-trials_3"
    ),
    pytest.param(NpseSamplingTestCase("fnpe", "sde", 8), id="fnpe-sde-trials_8"),
    pytest.param(NpseSamplingTestCase("gauss", "sde", 8), id="gauss-sde-trials_8"),
    pytest.param(
        NpseSamplingTestCase("auto_gauss", "sde", 8), id="auto_gauss-sde-trials_8"
    ),
    pytest.param(
        NpseSamplingTestCase("jac_gauss", "sde", 8), id="jac_gauss-sde-trials_8"
    ),
]

sampling_test_cases_all = sampling_test_cases_1_trial + sampling_test_cases_n_trials


@pytest.fixture(scope="module")
def npse_trained_model(request):
    num_simulations = 5_000
    stop_after_epochs = 200
    training_batch_size = 100
    test_case: NpseTrainingTestCase = request.param
    prior, _, _ = _get_npse_prior(
        prior_type=test_case.prior_type, num_dim=test_case.num_dim
    )
    score_estimator = _train_npse(
        test_case,
        num_simulations=num_simulations,
        stop_after_epochs=stop_after_epochs,
        training_batch_size=training_batch_size,
        prior=prior,
    )
    return score_estimator, test_case


@pytest.mark.slow
@pytest.mark.parametrize("npse_trained_model", training_test_cases_all, indirect=True)
@pytest.mark.parametrize("sampling_test_case", sampling_test_cases_all)
def test_c2st(npse_trained_model, sampling_test_case: NpseSamplingTestCase):
    num_samples = 10_000
    score_estimator, test_case = npse_trained_model
    prior, prior_mean, prior_cov = _get_npse_prior(
        prior_type=test_case.prior_type, num_dim=test_case.num_dim
    )

    x_o = zeros(sampling_test_case.num_trials, test_case.num_dim)

    posterior = ScorePosterior(
        score_estimator,
        prior=prior,
        sample_with=sampling_test_case.sampling_method,
    )
    posterior.set_default_x(x_o)

    npse_samples = posterior.sample(
        (num_samples,), iid_method=sampling_test_case.iid_method
    )

    target_samples, likelihood_shift, likelihood_cov = _get_npse_target_samples(
        num_samples=num_samples,
        num_dim=test_case.num_dim,
        prior_type=test_case.prior_type,
        prior=prior,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        x_o=x_o,
    )

    check_c2st(
        npse_samples,
        target_samples,
        alg=f"npse-{test_case.sde_type or 'vp'}-{test_case.prior_type}"
        f"-{test_case.num_dim}D-{sampling_test_case.sampling_method}",
        tol=0.05 * min(sampling_test_case.num_trials, 8),
    )
    if test_case.prior_type == "gaussian":
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o,
            likelihood_shift,
            likelihood_cov,
            prior_mean,
            prior_cov,
        )

        map_ = posterior.map(show_progress_bars=True, num_iter=5)
        assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5, "MAP is not close to GT."


@pytest.mark.slow
@pytest.mark.parametrize(
    "npse_trained_model", training_test_cases_gaussian, indirect=True
)
def test_kld_gaussian(npse_trained_model):
    # For the Gaussian prior, we compute the KLd between ground truth and
    # posterior.
    score_estimator, test_case = npse_trained_model
    prior, prior_mean, prior_cov = _get_npse_prior(
        prior_type=test_case.prior_type, num_dim=test_case.num_dim
    )

    x_o = zeros(1, test_case.num_dim)
    posterior = ScorePosterior(score_estimator, prior=prior)
    posterior.set_default_x(x_o)

    _, likelihood_shift, likelihood_cov = _get_npse_target_samples(
        num_samples=1,
        num_dim=test_case.num_dim,
        prior_type=test_case.prior_type,
        prior=prior,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        x_o=x_o,
    )
    dkl = get_dkl_gaussian_prior(
        posterior,
        x_o[0],
        likelihood_shift,
        likelihood_cov,
        prior_mean,
        prior_cov,
    )
    max_dkl = 0.15
    assert dkl < max_dkl, f"D-KL={dkl} is more than 2std above the average performance."


@pytest.mark.parametrize("sampling_test_case", sampling_test_cases_1_trial)
@pytest.mark.parametrize("training_test_case", training_test_cases_all)
def test_npse_snapshot(
    sampling_test_case: NpseSamplingTestCase,
    training_test_case: NpseTrainingTestCase,
    ndarrays_regression,
):
    num_simulations = 5
    num_samples = 10
    steps = 5

    prior, _, _ = _get_npse_prior(
        training_test_case.prior_type, training_test_case.num_dim
    )
    score_estimator = _train_npse(
        sde_type=training_test_case.sde_type,
        num_dim=training_test_case.num_dim,
        prior=prior,
        num_simulations=num_simulations,
        training_batch_size=None,
        stop_after_epochs=1,
        max_num_epochs=1,
    )

    x_o = zeros((sampling_test_case.num_trials, training_test_case.num_dim))

    posterior = ScorePosterior(
        score_estimator,
        prior=prior,
        sample_with=sampling_test_case.sampling_method,
    )
    posterior.set_default_x(x_o)

    samples = posterior.sample(
        (num_samples,), iid_method=sampling_test_case.iid_method, steps=steps
    )

    ndarrays_regression.check(
        {'values': samples.numpy()}, default_tolerance=dict(atol=1e-3, rtol=1e-2)
    )
