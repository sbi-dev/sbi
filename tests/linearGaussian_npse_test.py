from dataclasses import dataclass
from itertools import product
from typing import Optional

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

IID_METHODS = ["fnpe", "gauss", "auto_gauss", "jac_gauss"]
SAMPLING_METHODS = ["sde", "ode"]


@dataclass
class NpseTrainingTestCase:
    num_dim: int
    prior_type: Optional[str]
    sde_type: str

    @property
    def likelihood_shift(self):
        return -ones(self.num_dim)

    @property
    def likelihood_cov(self):
        return 0.3 * eye(self.num_dim)

    @property
    def prior_mean(self):
        if self.prior_type in {"gaussian", None}:
            return zeros(self.num_dim)
        else:
            return None

    @property
    def prior_cov(self):
        if self.prior_type in {"gaussian", None}:
            return eye(self.num_dim)
        else:
            return None

    @property
    def prior(self):
        if self.prior_type in {"gaussian", None}:
            prior = MultivariateNormal(
                loc=self.prior_mean, covariance_matrix=self.prior_cov
            )
        elif self.prior_type == "uniform":
            prior = utils.BoxUniform(
                -2.0 * ones(self.num_dim), 2.0 * ones(self.num_dim)
            )
        else:
            raise NotImplementedError(f"Unsupported prior type: {self.prior_type}")
        return prior

    def get_training_data(self, num_simulations: int):
        theta = self.prior.sample((num_simulations,))
        x = linear_gaussian(theta, self.likelihood_shift, self.likelihood_cov)
        return theta, x

    def get_target_samples(self, num_samples: int, x_o):
        if self.prior_type in {"gaussian", None}:
            gt_posterior = true_posterior_linear_gaussian_mvn_prior(
                x_o,
                self.likelihood_shift,
                self.likelihood_cov,
                self.prior_mean,
                self.prior_cov,
            )
            target_samples = gt_posterior.sample((num_samples,))
        else:  # prior_type == "uniform"
            target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
                x_o,
                self.likelihood_shift,
                self.likelihood_cov,
                prior=self.prior,
                num_samples=num_samples,
            )
        return target_samples


@dataclass
class NpseSamplingTestCase:
    iid_method: str
    sampling_method: str
    num_trials: int


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

training_test_cases_none = [
    pytest.param(NpseTrainingTestCase(2, None, "ve"), id="None_prior-dim_3-ve"),
    pytest.param(NpseTrainingTestCase(2, None, "vp"), id="None_prior-dim_3-vp"),
    pytest.param(NpseTrainingTestCase(2, None, "subvp"), id="None_prior-dim_3-subvp"),
]

training_test_cases_all = (
    training_test_cases_gaussian
    + training_test_cases_uniform
    + training_test_cases_none
)

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


def _train_npse(
    test_case: NpseTrainingTestCase,
    num_simulations,
    stop_after_epochs,
    training_batch_size,
):
    inference = NPSE(
        test_case.prior, show_progress_bars=True, sde_type=test_case.sde_type
    )
    theta, x = test_case.get_training_data(num_simulations)

    kwargs = {}
    if stop_after_epochs is not None:
        kwargs["stop_after_epochs"] = stop_after_epochs
    if training_batch_size is not None:
        kwargs["training_batch_size"] = training_batch_size

    score_estimator = inference.append_simulations(theta, x).train(**kwargs)
    return inference, score_estimator


@pytest.fixture(scope="module")
def npse_trained_model(request):
    num_simulations = 5_000
    stop_after_epochs = 200
    training_batch_size = 100
    test_case: NpseTrainingTestCase = request.param
    inference = NPSE(
        test_case.prior, sde_type=test_case.sde_type, show_progress_bars=True
    )
    theta, x = test_case.get_training_data(num_simulations)
    inference.append_simulations(theta, x)
    inference, score_estimator = _train_npse(
        test_case, num_simulations, stop_after_epochs, training_batch_size
    )
    return inference, score_estimator, test_case


@pytest.mark.slow
@pytest.mark.parametrize("npse_trained_model", training_test_cases_all, indirect=True)
@pytest.mark.parametrize("sampling_test_case", sampling_test_cases_all)
def test_c2st(npse_trained_model, sampling_test_case: NpseSamplingTestCase):
    num_samples = 1_000
    inference, score_estimator, test_case = npse_trained_model

    x_o = zeros(sampling_test_case.num_trials, test_case.num_dim)
    posterior = inference.build_posterior(
        score_estimator, sample_with=sampling_test_case.sampling_method
    )
    posterior.set_default_x(x_o)
    npse_samples = posterior.sample(
        (num_samples,), iid_method=sampling_test_case.iid_method
    )
    check_c2st(
        npse_samples,
        test_case.get_target_samples(npse_samples.shape[0], x_o),
        alg=f"npse-{test_case.sde_type or 'vp'}-{test_case.prior_type}"
        f"-{test_case.num_dim}D-{sampling_test_case.sampling_method}",
        tol=0.05 * min(sampling_test_case.num_trials, 8),
    )
    if test_case.prior_type == "gaussian":
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o,
            test_case.likelihood_shift,
            test_case.likelihood_cov,
            test_case.prior_mean,
            test_case.prior_cov,
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
    inference, score_estimator, test_case = npse_trained_model
    x_o = zeros(1, test_case.num_dim)
    posterior = inference.build_posterior(score_estimator)
    posterior.set_default_x(x_o)
    dkl = get_dkl_gaussian_prior(
        posterior,
        x_o[0],
        test_case.likelihood_shift,
        test_case.likelihood_cov,
        test_case.prior_mean,
        test_case.prior_cov,
    )
    max_dkl = 0.15
    assert dkl < max_dkl, f"D-KL={dkl} is more than 2std above the average performance."


@pytest.mark.skip
@pytest.mark.parametrize("sampling_test_case", sampling_test_cases_all)
@pytest.mark.parametrize("test_case", training_test_cases_none)
def test_npse_snapshot(
    test_case: NpseTrainingTestCase, sampling_test_case: NpseSamplingTestCase, snapshot
):
    num_simulations = 5
    num_samples = 3
    stop_after_epochs = 2
    steps = 7
    training_batch_size = num_simulations
    inference, score_estimator = _train_npse(
        test_case, num_simulations, stop_after_epochs, training_batch_size
    )
    x_o = zeros(sampling_test_case.num_trials, test_case.num_dim)
    posterior = inference.build_posterior(
        score_estimator, sample_with=sampling_test_case.sampling_method
    )
    posterior.set_default_x(x_o)
    samples = posterior.sample(
        (num_samples,), iid_method=sampling_test_case.iid_method, steps=steps
    )
    assert snapshot == samples.tolist()
