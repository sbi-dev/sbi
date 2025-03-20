from dataclasses import dataclass
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


@dataclass
class NpseTestCase:
    num_dim: int
    prior_type: Optional[str]
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

    def get_target_samples(self, num_samples: int):
        if self.prior_type in {"gaussian", None}:
            gt_posterior = true_posterior_linear_gaussian_mvn_prior(
                self.x_o,
                self.likelihood_shift,
                self.likelihood_cov,
                self.prior_mean,
                self.prior_cov,
            )
            target_samples = gt_posterior.sample((num_samples,))
        else:  # prior_type == "uniform"
            target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
                self.x_o,
                self.likelihood_shift,
                self.likelihood_cov,
                prior=self.prior,
                num_samples=num_samples,
            )
        return target_samples


test_cases_gaussian = [
    pytest.param(NpseTestCase(1, "gaussian", "vp"), id="gaussian_prior-dim_1-vp"),
    pytest.param(NpseTestCase(1, "gaussian", "vp"), id="gaussian_prior-dim_1-vp"),
    pytest.param(NpseTestCase(2, "gaussian", "ve"), id="gaussian_prior-dim_2-ve"),
    pytest.param(NpseTestCase(2, "gaussian", "vp"), id="gaussian_prior-dim_2-vp"),
    pytest.param(NpseTestCase(2, "gaussian", "subvp"), id="gaussian_prior-dim_2-subvp"),
]

test_cases_uniform = [
    pytest.param(NpseTestCase(3, "uniform", "ve"), id="uniform_prior-dim_3-ve"),
    pytest.param(NpseTestCase(3, "uniform", "vp"), id="uniform_prior-dim_3-vp"),
    pytest.param(NpseTestCase(3, "uniform", "subvp"), id="uniform_prior-dim_3-subvp"),
]

test_cases_none = [
    pytest.param(NpseTestCase(2, None, "ve"), id="None_prior-dim_3-ve"),
    pytest.param(NpseTestCase(2, None, "vp"), id="None_prior-dim_3-vp"),
    pytest.param(NpseTestCase(2, None, "subvp"), id="None_prior-dim_3-subvp"),
]

test_cases_all = test_cases_gaussian + test_cases_uniform + test_cases_none
iid_methods = ["fnpe", "gauss", "auto_gauss", "jac_gauss"]
sample_with = ["sde", "ode"]


def _train_npse(
    test_case: NpseTestCase, num_simulations, stop_after_epochs, training_batch_size
):
    inference = NPSE(
        test_case.prior, show_progress_bars=True, sde_type=test_case.sde_type
    )
    theta, x = test_case.get_training_data(num_simulations)
    if stop_after_epochs is None and training_batch_size is None:
        score_estimator = inference.train()
    elif stop_after_epochs is not None and training_batch_size is None:
        score_estimator = inference.train(stop_after_epochs=stop_after_epochs)
    elif stop_after_epochs is None and training_batch_size is not None:
        score_estimator = inference.train(training_batch_size=training_batch_size)
    else:
        score_estimator = inference.append_simulations(theta, x).train(
            stop_after_epochs=stop_after_epochs, training_batch_size=training_batch_size
        )
    return inference, score_estimator


@pytest.fixture
def npse_trained_model(request):
    num_simulations = 5_000
    stop_after_epochs = 200
    training_batch_size = 100
    test_case: NpseTestCase = request.param
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
@pytest.mark.parametrize("sample_with", sample_with)
@pytest.mark.parametrize("iid_method", iid_methods)
@pytest.mark.parametrize("npse_trained_model", test_cases_all, indirect=True)
def test_c2st(npse_trained_model, iid_method, sample_with):
    num_samples = 1_000
    inference, score_estimator, test_case = npse_trained_model
    num_dim = test_case.num_dim
    sde_type = test_case.sde_type
    prior_str = test_case.prior_type

    posterior = inference.build_posterior(score_estimator, sample_with=sample_with)
    posterior.set_default_x(test_case.x_o)
    npse_samples = posterior.sample((num_samples,), iid_method=iid_method)
    check_c2st(
        npse_samples,
        test_case.get_target_samples(npse_samples.shape[0]),
        alg=f"npse-{sde_type or 'vp'}-{prior_str}-{num_dim}D-{test_case.sample_with}",
    )

    if prior_str == "gaussian":
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            test_case.x_o,
            test_case.likelihood_shift,
            test_case.likelihood_cov,
            test_case.prior_mean,
            test_case.prior_cov,
        )

        map_ = posterior.map(show_progress_bars=True, num_iter=5)
        assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5, "MAP is not close to GT."


@pytest.mark.slow
@pytest.mark.parametrize("npse_trained_model", test_cases_gaussian, indirect=True)
def test_kld_gaussian(npse_trained_model):
    # For the Gaussian prior, we compute the KLd between ground truth and
    # posterior.
    inference, score_estimator, test_case = npse_trained_model
    posterior = inference.build_posterior(score_estimator)
    posterior.set_default_x(test_case.x_o)
    dkl = get_dkl_gaussian_prior(
        posterior,
        test_case.x_o[0],
        test_case.likelihood_shift,
        test_case.likelihood_cov,
        test_case.prior_mean,
        test_case.prior_cov,
    )
    max_dkl = 0.15
    assert dkl < max_dkl, (
        f"D-KL={dkl} is more than 2 stds above the average performance."
    )


@pytest.mark.parametrize("test_case", test_cases_all)
@pytest.mark.parametrize("sample_with", sample_with)
@pytest.mark.parametrize("iid_method", iid_methods)
def test_npse_iid_inference_snapshot(
    test_case: NpseTestCase, sample_with, iid_method, snapshot
):
    num_simulations = 5
    num_samples = 2
    stop_after_epochs = 1
    training_batch_size = num_simulations
    inference, score_estimator = _train_npse(
        test_case, num_simulations, stop_after_epochs, training_batch_size
    )
    posterior = inference.build_posterior(score_estimator, sample_with=sample_with)
    posterior.set_default_x(test_case.x_o)
    samples = posterior.sample((num_samples,), iid_method=iid_method)
    assert samples == snapshot
