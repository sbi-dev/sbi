from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Tuple

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
NUM_DIM = [1, 2, 3]
NUM_TRIAL = [1, 3, 8, 16]
PRIOR_TYPE = ["gaussian", "uniform", None]
SAMPLING_METHODS = ["sde", "ode"]
SDE_TYPE = ["vp", "ve", "subvp"]


@dataclass(frozen=True)
class NpseTrainingTestCase:
    num_dim: int
    prior_type: Optional[str]
    sde_type: str

    def __str__(self):
        return f"{self.prior_type}_prior-dim_{self.num_dim}-{self.sde_type}"

    @property
    def likelihood_shift(self):
        return -ones(self.num_dim)

    @property
    def likelihood_cov(self):
        return 0.3 * eye(self.num_dim)

    @property
    def prior_mean(self):
        if self.prior_type == "gaussian":
            return zeros(self.num_dim)
        else:
            return None

    @property
    def prior_cov(self):
        if self.prior_type == "gaussian":
            return eye(self.num_dim)
        else:
            return None

    @property
    def prior(self):
        if self.prior_type == "gaussian":
            prior = MultivariateNormal(
                loc=self.prior_mean, covariance_matrix=self.prior_cov
            )
        elif self.prior_type == "uniform":
            prior = utils.BoxUniform(
                -2.0 * ones(self.num_dim), 2.0 * ones(self.num_dim)
            )
        elif self.prior_type is None:
            prior = None
        else:
            raise NotImplementedError(f"Unsupported prior type: {self.prior_type}")
        return prior

    def _get_default_prior(self):
        if self.prior is not None:
            return self.prior

        return utils.BoxUniform(-2.0 * ones(self.num_dim), 2.0 * ones(self.num_dim))

    def get_training_data(self, num_simulations: int):
        theta = self._get_default_prior().sample((num_simulations,))
        x = linear_gaussian(theta, self.likelihood_shift, self.likelihood_cov)
        return theta, x

    def get_target_samples(self, num_samples: int, x_o):
        if self.prior_type == "gaussian":
            gt_posterior = true_posterior_linear_gaussian_mvn_prior(
                x_o,
                self.likelihood_shift,
                self.likelihood_cov,
                self.prior_mean,
                self.prior_cov,
            )
            target_samples = gt_posterior.sample((num_samples,))
        elif self.prior_type in {"uniform", None}:
            target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
                x_o,
                self.likelihood_shift,
                self.likelihood_cov,
                prior=self._get_default_prior(),
                num_samples=num_samples,
            )
        else:
            raise NotImplementedError(f"Unsupported prior type: {self.prior_type}")
        return target_samples


@dataclass(frozen=True)
class NpseSamplingTestCase:
    iid_method: str
    sampling_method: str
    num_trials: int

    def __str__(self):
        return f"{self.iid_method}-{self.sampling_method}-trials_{self.num_trials}"


training_test_cases_gaussian = [
    NpseTrainingTestCase(1, "gaussian", "vp"),
    NpseTrainingTestCase(1, "gaussian", "ve"),
    NpseTrainingTestCase(2, "gaussian", "ve"),
    NpseTrainingTestCase(2, "gaussian", "vp"),
    NpseTrainingTestCase(2, "gaussian", "subvp"),
]

training_test_cases_uniform = [
    NpseTrainingTestCase(2, "uniform", "ve"),
    NpseTrainingTestCase(2, "uniform", "vp"),
    NpseTrainingTestCase(2, "uniform", "subvp"),
    NpseTrainingTestCase(3, "uniform", "ve"),
    NpseTrainingTestCase(3, "uniform", "vp"),
    NpseTrainingTestCase(3, "uniform", "subvp"),
]


training_test_cases_all = training_test_cases_gaussian + training_test_cases_uniform

sampling_test_cases_1_trial = [
    NpseSamplingTestCase(iid, sampling, 1)
    for iid, sampling in product(IID_METHODS, SAMPLING_METHODS)
]

sampling_test_cases_n_trials = [
    NpseSamplingTestCase("fnpe", "sde", 3),
    NpseSamplingTestCase("gauss", "sde", 3),
    NpseSamplingTestCase("auto_gauss", "sde", 3),
    NpseSamplingTestCase("jac_gauss", "sde", 3),
    NpseSamplingTestCase("fnpe", "sde", 8),
    NpseSamplingTestCase("gauss", "sde", 8),
    NpseSamplingTestCase("auto_gauss", "sde", 8),
    NpseSamplingTestCase("jac_gauss", "sde", 8),
]

sampling_test_cases_all = sampling_test_cases_1_trial + sampling_test_cases_n_trials


def _get_regression_cases() -> List[Tuple[NpseTrainingTestCase, NpseSamplingTestCase]]:
    """
    # ToDO check if there is a bug for prior_type='uniform' and num_trial>1
    # ToDO validate bug for combination sde_type='ode' and num_trial>1
    # ToDO investigate non-determinism for prior_type=None, dim>= 2, ve, auto/jac_gauss
    To make the regression tests run fast enough, we exclude certain combinations
    :return: list of combinations of training and sampling test cases
    """
    all_train = [
        NpseTrainingTestCase(num_dim, prior_type, sde_type)
        for num_dim, prior_type, sde_type in product(NUM_DIM, PRIOR_TYPE, SDE_TYPE)
    ]
    all_sample = [
        NpseSamplingTestCase(iid_method, sampling_method, num_trials)
        for iid_method, sampling_method, num_trials in product(
            IID_METHODS, SAMPLING_METHODS, NUM_TRIAL
        )
    ]
    all_combinations = product(all_train, all_sample)

    is_uniform = lambda t: t.prior_type == "uniform"
    too_many_trial = lambda s: s.num_trials > 1
    is_ode = lambda s: s.sampling_method == "ode"

    is_non_deterministic = (
        lambda t, s: t.prior_type is None
        and t.sde_type == "ve"
        and t.num_dim >= 2
        and s.iid_method in {"auto_gauss", "jac_gauss"}
        and s.num_trials >= 16
    )

    exclude_cond = lambda t, s: (
        (is_uniform(t) or is_ode(s)) and too_many_trial(s)
    ) or is_non_deterministic(t, s)
    combinations = []
    for train_case, sampling_case in all_combinations:
        if not exclude_cond(train_case, sampling_case):
            combinations.append((train_case, sampling_case))
    return combinations


def _train_npse(
    test_case: NpseTrainingTestCase,
    num_simulations,
    stop_after_epochs=None,
    training_batch_size=None,
    max_num_epochs=None,
):
    inference = NPSE(
        prior=test_case.prior, show_progress_bars=True, sde_type=test_case.sde_type
    )
    theta, x = test_case.get_training_data(num_simulations)

    kwargs = {}
    if stop_after_epochs is not None:
        kwargs["stop_after_epochs"] = stop_after_epochs
    if training_batch_size is not None:
        kwargs["training_batch_size"] = training_batch_size
    if max_num_epochs is not None:
        kwargs["max_num_epochs"] = max_num_epochs

    score_estimator = inference.append_simulations(theta, x).train(**kwargs)
    return inference, score_estimator


def _build_posterior(inference, score_estimator, x_o, prior=None, sample_with=None):
    kwargs = {}
    if prior is not None:
        kwargs["prior"] = prior
    if sample_with is not None:
        kwargs["sample_with"] = sample_with

    posterior = inference.build_posterior(score_estimator, **kwargs)
    posterior.set_default_x(x_o)
    return posterior


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
@pytest.mark.parametrize(
    "npse_trained_model", training_test_cases_all, indirect=True, ids=str
)
@pytest.mark.parametrize("sampling_test_case", sampling_test_cases_all, ids=str)
def test_c2st(npse_trained_model, sampling_test_case: NpseSamplingTestCase):
    num_samples = 1_000
    inference, score_estimator, test_case = npse_trained_model
    x_o = zeros(sampling_test_case.num_trials, test_case.num_dim)
    posterior = _build_posterior(
        inference,
        score_estimator,
        x_o,
        prior=test_case.prior,
        sample_with=sampling_test_case.sampling_method,
    )
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
    "npse_trained_model", training_test_cases_gaussian, indirect=True, ids=str
)
def test_kld_gaussian(npse_trained_model):
    # For the Gaussian prior, we compute the KLd between ground truth and
    # posterior.
    inference, score_estimator, test_case = npse_trained_model
    x_o = zeros(1, test_case.num_dim)
    posterior = _build_posterior(inference, score_estimator, x_o)
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


@pytest.mark.parametrize(
    "training_test_case, sampling_test_case", _get_regression_cases(), ids=str
)
def test_npse_snapshot(
    sampling_test_case: NpseSamplingTestCase,
    training_test_case: NpseTrainingTestCase,
    ndarrays_regression,
):
    num_simulations = 5
    num_samples = 2
    steps = 5

    inference, score_estimator = _train_npse(
        training_test_case,
        num_simulations,
        stop_after_epochs=1,
        max_num_epochs=1,
        training_batch_size=None,
    )
    x_o = zeros(sampling_test_case.num_trials, training_test_case.num_dim)
    posterior = _build_posterior(
        inference,
        score_estimator,
        x_o,
        prior=training_test_case.prior,
        sample_with=sampling_test_case.sampling_method,
    )
    samples = posterior.sample(
        (num_samples,), iid_method=sampling_test_case.iid_method, steps=steps
    )
    ndarrays_regression.check(
        {'values': samples.numpy()}, default_tolerance=dict(atol=1e-3, rtol=1e-2)
    )
