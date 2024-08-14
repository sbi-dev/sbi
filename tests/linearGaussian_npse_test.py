import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import NPSE
from sbi.simulators import linear_gaussian
from sbi.simulators.linear_gaussian import (
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)

from .test_utils import check_c2st, get_dkl_gaussian_prior


@pytest.mark.slow
@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize(
    "num_dim, prior_str",
    ((2, "gaussian"), (2, "uniform"), (1, "gaussian")),
)
def test_c2st_npse_on_linearGaussian(sde_type, num_dim: int, prior_str: str):
    """Test whether NPSE infers well a simple example with available ground truth."""

    x_o = zeros(1, num_dim)
    num_samples = 1000
    num_simulations = 3000

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

    inference = NPSE(prior, sde_type=sde_type, show_progress_bars=False)

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    score_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=100
    )
    posterior = inference.build_posterior(score_estimator)
    posterior.set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"npse-{sde_type}-{prior_str}-{num_dim}D")

    map_ = posterior.map(show_progress_bars=True)
    assert torch.allclose(map_, gt_posterior.mean, atol=0.2)

    # Checks for log_prob()
    if prior_str == "gaussian":
        # For the Gaussian prior, we compute the KLd between ground truth and
        # posterior.
        dkl = get_dkl_gaussian_prior(
            posterior,
            x_o[0],
            likelihood_shift,
            likelihood_cov,
            prior_mean,
            prior_cov,
        )

        max_dkl = 0.15

        assert (
            dkl < max_dkl
        ), f"D-KL={dkl} is more than 2 stds above the average performance."


def test_c2st_npse_on_linearGaussian_different_dims():
    """Test SNPE on linear Gaussian with different theta and x dimensionality."""

    theta_dim = 3
    x_dim = 2
    discard_dims = theta_dim - x_dim

    x_o = zeros(1, x_dim)
    num_samples = 1000
    num_simulations = 2000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(x_dim)
    likelihood_cov = 0.3 * eye(x_dim)

    prior_mean = zeros(theta_dim)
    prior_cov = eye(theta_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    target_samples = samples_true_posterior_linear_gaussian_mvn_prior_different_dims(
        x_o,
        likelihood_shift,
        likelihood_cov,
        prior_mean,
        prior_cov,
        num_discarded_dims=discard_dims,
        num_samples=num_samples,
    )

    def simulator(theta):
        return linear_gaussian(
            theta,
            likelihood_shift,
            likelihood_cov,
            num_discarded_dims=discard_dims,
        )

    # Test whether prior can be `None`.
    inference = NPSE(prior=None)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    # Test whether we can stop and resume.
    inference.append_simulations(theta, x).train(
        max_num_epochs=10, training_batch_size=100
    )
    inference.train(
        resume_training=True, force_first_round_loss=True, training_batch_size=100
    )
    posterior = inference.build_posterior().set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="npse_different_dims_and_resume_training")


# @pytest.mark.slow
# @pytest.mark.mcmc
# def test_sample_conditional(mcmc_params_accurate: dict):
#     """
#     Test whether sampling from the conditional gives the same results as
#     evaluating.

#     This compares samples that get smoothed with a Gaussian kde to evaluating
#     the conditional log-probability with `eval_conditional_density`.

#     `eval_conditional_density` is itself tested in `sbiutils_test.py`. Here, we
#     use a bimodal posterior to test the conditional.

#     NOTE: The comparison between conditional log_probs obtained from the MCMC
#     posterior and from analysis.eval_conditional_density can be gamed by
#     underfitting the posterior estimator, i.e., by using a small number of
#     simulations.
#     """

#     num_dim = 3
#     dim_to_sample_1 = 0
#     dim_to_sample_2 = 2
#     num_simulations = 5500
#     num_conditional_samples = 1000
#     num_conditions = 50

#     x_o = zeros(1, num_dim)

#     likelihood_shift = -1.0 * ones(num_dim)
#     likelihood_cov = 0.05 * eye(num_dim)

#     prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

#     # TODO: janfb does not see how this setup results in a bi-model posterior.
#     def simulator(theta):
#         batch_size, _ = theta.shape
#         # create -1 1 mask for bimodality
#         mask = ones(batch_size, 1)
#         # set mask to -1 randomly across the batch
#         mask = mask * 2 * (rand(batch_size, 1) > 0.5) - 1

#         # Sample bi-modally by applying a 1-(-1) mask to the likelihood shift.
#         return linear_gaussian(theta, mask * likelihood_shift, likelihood_cov)

# # Test whether SNPE works properly with structured z-scoring.
# net = posterior_nn("maf", z_score_x="structured", hidden_features=20)

# inference = SNPE_C(prior, density_estimator=net, show_progress_bars=True)

# # We need a pretty big dataset to properly model the bimodality.
# theta = prior.sample((num_simulations,))
# x = simulator(theta)
# posterior_estimator = inference.append_simulations(theta, x).train(
#     training_batch_size=1000, max_num_epochs=60
# )

# # generate conditions
# posterior = DirectPosterior(
#     prior=prior, posterior_estimator=posterior_estimator
# ).set_default_x(x_o)
# samples = posterior.sample((num_conditions,))

# # Evaluate the conditional density be drawing samples and smoothing with a Gaussian
# # kde.
# potential_fn, theta_transform = posterior_estimator_based_potential(
#     posterior_estimator, prior=prior, x_o=x_o
# )
# (
#     conditioned_potential_fn,
#     restricted_tf,
#     restricted_prior,
# ) = conditional_potential(
#     potential_fn=potential_fn,
#     theta_transform=theta_transform,
#     prior=prior,
#     condition=samples[0],
#     dims_to_sample=[dim_to_sample_1, dim_to_sample_2],
# )
# mcmc_posterior = MCMCPosterior(
#     potential_fn=conditioned_potential_fn,
#     theta_transform=restricted_tf,
#     proposal=restricted_prior,
#     method="slice_np_vectorized",
#     **mcmc_params_accurate,
# )
# cond_samples = mcmc_posterior.sample((num_conditional_samples,), x=x_o)

# limits = [[-2, 2], [-2, 2], [-2, 2]]

# # Fit a Gaussian KDE to the conditional samples and get log-probs.
# density = gaussian_kde(cond_samples.numpy().T, bw_method="scott")

# X, Y = np.meshgrid(
#     np.linspace(limits[0][0], limits[0][1], 50),
#     np.linspace(limits[1][0], limits[1][1], 50),
# )
# positions = np.vstack([X.ravel(), Y.ravel()])
# sample_kde_grid = np.reshape(density(positions).T, X.shape)

# # Get conditional log probs eval_conditional_density.
# eval_grid = analysis.eval_conditional_density(
#     posterior,
#     condition=samples[0],
#     dim1=dim_to_sample_1,
#     dim2=dim_to_sample_2,
#     limits=torch.tensor([[-2, 2], [-2, 2], [-2, 2]]),
# )

# # Compare the two densities.
# sample_kde_grid = sample_kde_grid / np.sum(sample_kde_grid)
# eval_grid = eval_grid / torch.sum(eval_grid)

# error = np.abs(sample_kde_grid - eval_grid.numpy())

# max_err = np.max(error)
# assert max_err < 0.0027
# print(f"Max error: {max_err}")
