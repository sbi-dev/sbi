from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal


from sbi import utils
from tests.sbiutils_test import conditional_of_mvn
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)

from tests.test_utils import check_c2st


def test_mdn_conditional_density(num_dim: int = 3, cond_dim: int = 1):
    """Test whether the conditional density infered from MDN parameters of a 
    `DirectPosterior` matches analytical results for MVN. This uses a n-D joint and
    conditions on the last m values to generate a conditional.

    Gaussian prior used for easier ground truthing of conditional posterior.

    Args:
        num_dim: Dimensionality of the MVM.
        cond_dim: Dimensionality of the condition.
    """

    assert (
        num_dim > cond_dim
    ), "The number of dimensions needs to be greater than that of the condition!"

    x_o = zeros(1, num_dim)
    num_samples = 1000
    num_simulations = 2500
    condition = 0.1 * ones(1, num_dim)

    dims = list(range(num_dim))
    dims2sample = dims[-cond_dim:]
    dims2condition = dims[:-cond_dim]

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    joint_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    joint_cov = joint_posterior.covariance_matrix
    joint_mean = joint_posterior.loc

    conditional_mean, conditional_cov = conditional_of_mvn(
        joint_mean, joint_cov, condition[0, dims2condition]
    )
    conditional_dist_gt = MultivariateNormal(conditional_mean, conditional_cov)

    conditional_samples_gt = conditional_dist_gt.sample((num_samples,))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE(prior, show_progress_bars=False, density_estimator="mdn")

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=1000
    )
    _ = inference.append_simulations(theta, x).train(training_batch_size=100)
    posterior = inference.build_posterior().set_default_x(x_o)

    conditional_samples_sbi = posterior.sample_conditional(
        (num_samples,), condition, dims2sample, x_o
    )
    check_c2st(
        conditional_samples_sbi,
        conditional_samples_gt,
        alg="analytic_mdn_conditioning_of_direct_posterior",
    )

