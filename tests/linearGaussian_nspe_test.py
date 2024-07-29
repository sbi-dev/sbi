import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import simulate_for_sbi
from sbi.inference.nspe.nspe import NSPE
from sbi.simulators import linear_gaussian
from sbi.simulators.linear_gaussian import (
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.user_input_checks import prepare_for_sbi

from .test_utils import check_c2st


@pytest.mark.slow
@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize(
    "num_dim, prior_str",
    ((2, "gaussian"), (2, "uniform"), (1, "gaussian")),
)
def test_c2st_snpe_on_linearGaussian(sde_type, num_dim: int, prior_str: str):
    """Test whether SNPE infers well a simple example with available ground truth."""

    x_o = zeros(1, num_dim)
    num_samples = 1000
    num_simulations = 2500

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

    simulator, prior = prepare_for_sbi(
        lambda theta: linear_gaussian(theta, likelihood_shift, likelihood_cov),
        prior,
    )

    inference = NSPE(prior, sde_type=sde_type, show_progress_bars=False)

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=1000
    )
    score_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=100
    )
    posterior = inference.build_posterior(score_estimator)
    posterior.set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="nspe")

    # map_ = posterior.map(num_init_samples=1_000, show_progress_bars=False)
    # assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5

    # Checks for log_prob()
    # if prior_str == "gaussian":
    #     # For the Gaussian prior, we compute the KLd between ground truth and
    #     # posterior.
    #     dkl = get_dkl_gaussian_prior(
    #         posterior,
    #         x_o[0],
    #         likelihood_shift,
    #         likelihood_cov,
    #         prior_mean,
    #         prior_cov,
    #     )

    #     max_dkl = 0.15

    #     assert (
    #         dkl < max_dkl
    #     ), f"D-KL={dkl} is more than 2 stds above the average performance."
