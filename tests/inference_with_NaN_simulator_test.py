# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import (
    SNL,
    SNPE_A,
    SNPE_C,
    SRE,
    DirectPosterior,
    MCMCPosterior,
    prepare_for_sbi,
    simulate_for_sbi,
)
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
def test_handle_invalid_x(x_shape):

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
    method: type, exclude_invalid_x: bool, percent_nans: float
):

    # likelihood_mean will be likelihood_shift+theta
    num_dim = 3
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    x_o = zeros(1, num_dim)
    num_samples = 500
    num_simulations = 5000

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
    inference = method(prior=prior)

    theta, x = simulate_for_sbi(simulator, prior, num_simulations)
    _ = inference.append_simulations(theta, x).train(
        exclude_invalid_x=exclude_invalid_x
    )
    posterior = inference.build_posterior()

    samples = posterior.sample((num_samples,), x=x_o)

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"{method}")


@pytest.mark.slow
def test_inference_with_restriction_estimator():

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
    restriction_estimator = RestrictionEstimator(prior=prior)
    proposals = [prior]
    num_rounds = 2

    for r in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposals[-1], 1000)
        restriction_estimator.append_simulations(theta, x)
        if r < num_rounds - 1:
            _ = restriction_estimator.train()
        proposals.append(restriction_estimator.restrict_prior())

    all_theta, all_x, _ = restriction_estimator.get_simulations()

    # Any method can be used in combination with the `RejectionEstimator`.
    inference = SNPE_C(prior=prior)
    posterior_estimator = inference.append_simulations(all_theta, all_x).train()

    # Build posterior.
    posterior = DirectPosterior(
        prior=prior, posterior_estimator=posterior_estimator
    ).set_default_x(x_o)

    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"{SNPE_C}")


@pytest.mark.parametrize("prior", ("uniform", "gaussian"))
def test_restricted_prior_log_prob(prior):
    """Test whether the log-prob method of the restricted prior works appropriately."""

    def simulator(theta):
        perturbed_theta = theta + 0.5 * torch.randn(2)
        perturbed_theta[theta[:, 0] < 0.8] = torch.as_tensor(
            [float("nan"), float("nan")]
        )
        return perturbed_theta

    if prior == "uniform":
        prior = utils.BoxUniform(-2 * torch.ones(2), 2 * torch.ones(2))
    else:
        prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    theta, x = simulate_for_sbi(simulator, prior, 1000)

    restriction_estimator = RestrictionEstimator(prior=prior)
    restriction_estimator.append_simulations(theta, x)
    _ = restriction_estimator.train(max_num_epochs=40)
    restricted_prior = restriction_estimator.restrict_prior()

    def integrate_grid(distribution):
        resolution = 500
        range_ = 4
        x = torch.linspace(-range_, range_, resolution)
        y = torch.linspace(-range_, range_, resolution)
        X, Y = torch.meshgrid(x, y)
        xy = torch.stack([X, Y])
        xy = torch.reshape(xy, (2, resolution**2)).T
        dist_on_grid = torch.exp(distribution.log_prob(xy))
        integral = torch.sum(dist_on_grid) / resolution**2 * (2 * range_) ** 2
        return integral

    integal_restricted = integrate_grid(restricted_prior)
    error = torch.abs(integal_restricted - torch.as_tensor(1.0))
    assert error < 0.01, "The restricted prior does not integrate to one."

    theta = prior.sample((10_000,))
    restricted_prior_probs = torch.exp(restricted_prior.log_prob(theta))

    valid_thetas = restricted_prior.predict(theta).bool()
    assert torch.all(
        restricted_prior_probs[valid_thetas] > 0.0
    ), "Accepted theta have zero probability."
    assert torch.all(
        restricted_prior_probs[torch.logical_not(valid_thetas)] == 0.0
    ), "Rejected theta has non-zero probablity."
