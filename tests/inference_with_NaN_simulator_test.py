# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import pytest
import torch
from torch import eye, ones, zeros

from sbi import utils as utils
from sbi.inference import SNL, SNPE_A, SNPE_C, SRE, prepare_for_sbi, simulate_for_sbi
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    samples_true_posterior_linear_gaussian_uniform_prior,
)
from sbi.utils import RestrictionEstimator
from sbi.utils.sbiutils import handle_invalid_x
from tests.test_utils import check_c2st


@pytest.mark.parametrize(
    "x_shape",
    (
        torch.Size((1, 1)),
        torch.Size((1, 10)),
        torch.Size((10, 1)),
        torch.Size((10, 10)),
    ),
)
def test_handle_invalid_x(x_shape, set_seed):

    x = torch.rand(x_shape)
    x[x < 0.1] = float("nan")
    x[x > 0.9] = float("inf")

    x_is_valid, *_ = handle_invalid_x(x, exclude_invalid_x=True)

    assert torch.isfinite(x[x_is_valid]).all()


@pytest.mark.parametrize("snpe_method", [SNPE_A, SNPE_C])
def test_z_scoring_warning(snpe_method: type):

    # Create data with large variance.
    num_dim = 2
    theta = torch.ones(100, num_dim)
    x = torch.rand(100, num_dim)
    x[:50] += 1e7

    # Make sure a warning is raised because z-scoring will map these data to duplicate
    # data points.
    with pytest.warns(UserWarning, match="Z-scoring these simulation outputs"):
        snpe_method(utils.BoxUniform(zeros(num_dim), ones(num_dim))).append_simulations(
            theta, x
        ).train(max_num_epochs=1)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("method", "exclude_invalid_x", "percent_nans"),
    ((SNPE_C, True, 0.05), (SNL, True, 0.05), (SRE, True, 0.05)),
)
def test_inference_with_nan_simulator(
    method: type, exclude_invalid_x: bool, percent_nans: float, set_seed
):

    # likelihood_mean will be likelihood_shift+theta
    num_dim = 3
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    x_o = zeros(1, num_dim)
    num_samples = 500
    num_simulations = 2000

    def linear_gaussian_nan(
        theta, likelihood_shift=likelihood_shift, likelihood_cov=likelihood_cov
    ):
        x = linear_gaussian(theta, likelihood_shift, likelihood_cov)
        # Set nan randomly.
        x[torch.rand(x.shape) < (percent_nans * 1.0 / x.shape[1])] = float("nan")

        return x

    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
    target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
        x_o,
        likelihood_shift=likelihood_shift,
        likelihood_cov=likelihood_cov,
        num_samples=num_samples,
        prior=prior,
    )

    simulator, prior = prepare_for_sbi(linear_gaussian_nan, prior)
    inference = method(prior)

    theta, x = simulate_for_sbi(simulator, prior, num_simulations)
    _ = inference.append_simulations(theta, x).train(
        exclude_invalid_x=exclude_invalid_x
    )

    posterior = inference.build_posterior().set_default_x(x_o)

    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"{method}")


@pytest.mark.slow
def test_inference_with_restriction_estimator(set_seed):

    # likelihood_mean will be likelihood_shift+theta
    num_dim = 3
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    x_o = zeros(1, num_dim)
    num_samples = 500

    def linear_gaussian_nan(
        theta, likelihood_shift=likelihood_shift, likelihood_cov=likelihood_cov
    ):
        condition = theta[:, 0] < 0.0
        x = linear_gaussian(theta, likelihood_shift, likelihood_cov)
        x[condition] = float("nan")

        return x

    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))
    target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
        x_o,
        likelihood_shift=likelihood_shift,
        likelihood_cov=likelihood_cov,
        num_samples=num_samples,
        prior=prior,
    )

    simulator, prior = prepare_for_sbi(linear_gaussian_nan, prior)
    rejection_estimator = RestrictionEstimator(prior=prior)
    proposals = [prior]
    num_rounds = 2

    for r in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposals[-1], 1000)
        rejection_estimator.append_simulations(theta, x)
        if r < num_rounds - 1:
            _ = rejection_estimator.train()
        proposals.append(rejection_estimator.restrict_prior())

    all_theta, all_x, _ = rejection_estimator.get_simulations()

    # Any method can be used in combination with the `RejectionEstimator`.
    inference = SNPE_C(prior=prior)
    _ = inference.append_simulations(all_theta, all_x).train()

    # Build posterior.
    posterior = inference.build_posterior().set_default_x(x_o)

    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"{SNPE_C}")
