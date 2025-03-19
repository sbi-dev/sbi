from dataclasses import dataclass
from typing import List, Optional

import pytest
import pytest_cases
from coverage.files import actual_path
from pyro.contrib.autoname import scope
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


@dataclass
class NpseTestCase:
    num_dim: int
    prior_type: str
    sde_type: str
    sample_with: Optional[str] = None
    iid_method: Optional[str] = None

    @property
    def likelihood_shift(self):
        return -ones(self.num_dim)

    @property
    def likelihood_cov(self):
        return 0.3 * eye(self.num_dim)

    @property
    def x_o(self):
        return zeros(self.num_dim)

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
            prior = MultivariateNormal(loc=self.prior_mean,
                                       covariance_matrix=self.prior_cov)
        elif self.prior_type == "uniform":
            prior = utils.BoxUniform(-2.0 * ones(self.num_dim),
                                     2.0 * ones(self.num_dim))
        else:
            raise NotImplementedError(f"Unsupported prior type: {self.prior_type}")
        return prior

    def get_training_data(self, num_simulations: int):
        theta = self.prior.sample((num_simulations,))
        x = linear_gaussian(theta, self.likelihood_shift, self.likelihood_cov)
        return theta, x

    def get_target_samples(self, num_samples):
        if self.prior_type in {"gaussian", None}:
            gt_posterior = true_posterior_linear_gaussian_mvn_prior(
                self.x_o, self.likelihood_shift, self.likelihood_cov, self.prior_mean,
                self.prior_cov
            )
            target_samples = gt_posterior.sample((num_samples,))
        else:  # prior_type == "uniform"
            target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
                self.x_o, self.likelihood_shift, self.likelihood_cov, prior=self.prior,
                num_samples=num_samples
            )
        return target_samples

class NpseTestCaseCollection:
    @pytest_cases.parametrize("sde_type, num_dim, prior_str",
                              [
                                  ("vp", 1, "gaussian"),
                                  ("vp", 3, "uniform"),
                                  ("vp", 3, "gaussian"),
                                  ("ve", 3, "uniform"),
                                  ("subvp", 3, "uniform"),
                              ],)
    def case_simple(self, sde_type, num_dim, prior_str) -> NpseTestCase:
        return NpseTestCase(num_dim, prior_str, sde_type)

    @pytest_cases.parametrize("sde_type, num_dim", [("vp", 1), ("vp", 3)])
    def case_gaussian(self, sde_type, num_dim) -> NpseTestCase:
        return NpseTestCase(num_dim, "gaussian", sde_type)


@pytest_cases.fixture
# @pytest_cases.parametrize("num_simulations", [5, 10])
# @pytest_cases.parametrize("training_batch_size", [200, None])
# @pytest_cases.parametrize("stop_after_epochs", [200, None])
@pytest_cases.parametrize_with_cases("test_case", cases=NpseTestCaseCollection)
def npse_trained_model(test_case: NpseTestCase):
    inference = NPSE(test_case.prior, sde_type=test_case.sde_type, show_progress_bars=True)
    num_simulations = 5
    theta, x = test_case.get_training_data(num_simulations)
    inference.append_simulations(theta, x)
    training_batch_size = x.shape[0]
    stop_after_epochs = 1

    """
    if stop_after_epochs is None and training_batch_size is None:
        score_estimator = inference.train()
    elif stop_after_epochs is not None and training_batch_size is None:
        score_estimator = inference.train(stop_after_epochs=stop_after_epochs)
    elif stop_after_epochs is None and training_batch_size is not None:
        score_estimator = inference.train(training_batch_size=training_batch_size)
    else:
    """
    score_estimator = inference.train(training_batch_size=training_batch_size,
                                          stop_after_epochs=stop_after_epochs)
    return inference, score_estimator


# @pytest_cases.fixture
# @pytest_cases.parametrize("sample_with", ["sde", "ode"])
# def npse_posterior(npse_trained_model, sample_with):
#     inference, score_estimator = npse_trained_model
#     return inference.build_posterior(score_estimator, sample_with=sample_with)
#
#
# @pytest_cases.fixture
# @pytest_cases.parametrize("iid_method", ["fnpe", "gauss", "auto_gauss", "jac_gauss"])
# def npse_samples(npse_posterior, iid_method):
#     num_samples = 10
#     return npse_posterior.sample((num_samples,), iid_method=iid_method)


@pytest.mark.slow
@pytest_cases.parametrize_with_cases("test_case",
                                     cases=NpseTestCaseCollection.case_gaussian,
                                     )
def test_c2st(test_case: NpseTestCase, npse_samples, current_cases):
    num_dim = test_case.num_dim
    sde_type = test_case.sde_type
    prior_str = test_case.prior_type
    check_c2st(
        npse_samples,
        test_case.get_target_samples(npse_samples.shape[0]),
        alg=f"npse-{sde_type or 'vp'}-{prior_str}-{num_dim}D-{test_case.sample_with}",
    )


# @pytest.mark.slow
# @pytest_cases.parametrize_with_cases("test_case",
#                                      cases=NpseTestCaseCollection.case_gaussian,)
# def test_kld_gaussian(test_case: NpseTestCase, npse_posterior):
#     # For the Gaussian prior, we compute the KLd between ground truth and
#     # posterior.
#     dkl = get_dkl_gaussian_prior(
#         npse_posterior,
#         test_case.x_o[0],
#         test_case.likelihood_shift,
#         test_case.likelihood_cov,
#         test_case.prior_mean,
#         test_case.prior_cov,
#     )
#
#     max_dkl = 0.15
#
#     assert dkl < max_dkl, (
#         f"D-KL={dkl} is more than 2 stds above the average performance."
#     )


def _train_and_sample(num_dim, num_samples, num_simulations, prior_type, sde_type,
                      stop_after_epochs, training_batch_size):
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
        stop_after_epochs=stop_after_epochs, training_batch_size=training_batch_size
    )
    posterior = inference.build_posterior(score_estimator)
    posterior.set_default_x(x_o)
    samples = posterior.sample((num_samples,), iid_method=iid_method)

    return inference, score_estimator, posterior, samples


@pytest.mark.parametrize(
    "sde_type, prior_type",
    [
        pytest.param("ve", None, id="ve-None"),
        pytest.param("vp", "gaussian", id="vp-gaussian"),
    ],
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
def test_npse_iid_inference_snapshot(sde_type, prior_type, iid_method, num_trial,
                                     snapshot):
    num_simulations = 10
    _, _, _, samples = _train_and_sample(1, 10, num_simulations,
                                         prior_type, sde_type, 1,
                                         training_batch_size=num_simulations)
    assert samples == snapshot

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
