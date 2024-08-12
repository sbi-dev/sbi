# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import gaussian_kde
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.analysis import conditional_potential
from sbi.inference import (
    DirectPosterior,
    MCMCPosterior,
    posterior_estimator_based_potential,
    simulate_for_sbi,
)

# TODO import this from inference
from sbi.inference.fmpe.base import FMPE
from sbi.neural_nets.factory import flowmatching_nn
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.user_input_checks import process_simulator

from .test_utils import (
    check_c2st,
    get_dkl_gaussian_prior,
    get_normalization_uniform_prior,
    get_prob_outside_uniform_prior,
)


@pytest.mark.parametrize(
    "num_dim, prior_str",
    ((2, "gaussian"), (2, "uniform"), (1, "gaussian")),
)
def test_c2st_fmpe_on_linearGaussian(num_dim: int, prior_str: str):
    """Test whether fmpe infers well a simple example with available ground truth."""

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

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = FMPE(prior, show_progress_bars=False)

    posterior_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=100
    )
    posterior = DirectPosterior(
        prior=prior, posterior_estimator=posterior_estimator
    ).set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="fmpe")

    map_ = posterior.map(num_init_samples=1_000, show_progress_bars=False)

    # Checks for log_prob()
    if prior_str == "gaussian":
        # For the Gaussian prior, we compute the KLd between ground truth and posterior.
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

        assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5

    elif prior_str == "uniform":
        # Check whether the returned probability outside of the support is zero.
        posterior_prob = get_prob_outside_uniform_prior(posterior, prior, num_dim)
        assert (
            posterior_prob == 0.0
        ), "The posterior probability outside of the prior support is not zero"

        # Check whether normalization (i.e. scaling up the density due
        # to leakage into regions without prior support) scales up the density by the
        # correct factor.
        (
            posterior_likelihood_unnorm,
            posterior_likelihood_norm,
            acceptance_prob,
        ) = get_normalization_uniform_prior(posterior, prior, x=x_o)
        # The acceptance probability should be *exactly* the ratio of the unnormalized
        # and the normalized likelihood. However, we allow for an error margin of 1%,
        # since the estimation of the acceptance probability is random (based on
        # rejection sampling).
        assert (
            acceptance_prob * 0.99
            < posterior_likelihood_unnorm / posterior_likelihood_norm
            < acceptance_prob * 1.01
        ), "Normalizing the posterior density using the acceptance probability failed."

        assert ((map_ - ones(num_dim)) ** 2).sum() < 0.5


# TODO: add more architectures, e.g., zuko mlp vs. resnet?
@pytest.mark.slow
@pytest.mark.parametrize("model", ["mlp"])
def test_fmpe_with_different_estimators(model):
    """Test fmpe with different vector field estimators on linear Gaussian."""

    theta_dim = 3
    x_dim = 3

    x_o = zeros(1, x_dim)
    num_samples = 1000
    num_simulations = 2000

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(x_dim)
    likelihood_cov = 0.3 * eye(x_dim)

    prior_mean = zeros(theta_dim)
    prior_cov = eye(theta_dim)

    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    estimator_build_fun = flowmatching_nn(model=model)

    inference = FMPE(prior, density_estimator=estimator_build_fun)

    posterior_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=100
    )
    posterior = DirectPosterior(
        prior=prior, posterior_estimator=posterior_estimator
    ).set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"fmpe_{model}")


def test_c2st_fmpe_on_linearGaussian_different_dims(density_estimator="maf"):
    """Test fmpe on linear Gaussian with different theta and x dimensionality."""

    theta_dim = 3
    x_dim = 2
    discarded_dims = theta_dim - x_dim

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
        num_discarded_dims=discarded_dims,
        num_samples=num_samples,
    )

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(
        theta, likelihood_shift, likelihood_cov, num_discarded_dims=discarded_dims
    )

    # Test whether prior can be `None`.
    inference = FMPE(
        prior=None,
        density_estimator=density_estimator,
        show_progress_bars=False,
    )

    inference = inference.append_simulations(theta, x)
    posterior_estimator = inference.train(
        max_num_epochs=10
    )  # Test whether we can stop and resume.
    posterior_estimator = inference.train(
        resume_training=True, force_first_round_loss=True
    )
    posterior = DirectPosterior(
        prior=prior, posterior_estimator=posterior_estimator
    ).set_default_x(x_o)
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg="fmpe_different_dims")


@pytest.mark.slow
def test_sample_conditional():
    """
    Test whether sampling from the conditional gives the same results as evaluating.

    This compares samples that get smoothed with a Gaussian kde to evaluating the
    conditional log-probability with `eval_conditional_density`.

    `eval_conditional_density` is itself tested in `sbiutils_test.py`. Here, we use
    a bimodal posterior to test the conditional.
    """

    num_dim = 3
    dim_to_sample_1 = 0
    dim_to_sample_2 = 2
    num_simulations = 6000
    num_conditional_samples = 500

    mcmc_parameters = dict(
        method="slice_np_vectorized", num_chains=20, warmup_steps=50, thin=5
    )

    x_o = zeros(1, num_dim)

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.1 * eye(num_dim)

    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    def simulator(theta):
        if torch.rand(1) > 0.5:
            return linear_gaussian(theta, likelihood_shift, likelihood_cov)
        else:
            return linear_gaussian(theta, -likelihood_shift, likelihood_cov)

    simulator = process_simulator(simulator, prior, False)

    # We need a pretty big dataset to properly model the bimodality.
    theta, x = simulate_for_sbi(
        simulator,
        prior,
        num_simulations,
        simulation_batch_size=10,  # choose small batch size to ensure bimoality.
    )

    # Test whether fmpe works properly with structured z-scoring.
    net = utils.posterior_nn("maf", z_score_x="structured", hidden_features=20)

    inference = FMPE(prior, density_estimator=net, show_progress_bars=False)
    posterior_estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=60
    )

    posterior = DirectPosterior(
        prior=prior, posterior_estimator=posterior_estimator
    ).set_default_x(x_o)
    samples = posterior.sample((50,))

    # Evaluate the conditional density be drawing samples and smoothing with a Gaussian
    # kde.
    potential_fn, theta_transform = posterior_estimator_based_potential(
        posterior_estimator, prior=prior, x_o=x_o
    )
    (
        conditioned_potential_fn,
        restricted_tf,
        restricted_prior,
    ) = conditional_potential(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        prior=prior,
        condition=samples[0],
        dims_to_sample=[dim_to_sample_1, dim_to_sample_2],
    )
    mcmc_posterior = MCMCPosterior(
        potential_fn=conditioned_potential_fn,
        theta_transform=restricted_tf,
        proposal=restricted_prior,
        **mcmc_parameters,
    )
    mcmc_posterior.set_default_x(x_o)  # TODO: This test has a bug? Needed to add this
    cond_samples = mcmc_posterior.sample((num_conditional_samples,))

    _ = analysis.pairplot(
        cond_samples,
        limits=[[-2, 2], [-2, 2], [-2, 2]],
        figsize=(2, 2),
        diag="kde",
        offdiag="kde",
    )

    limits = [[-2, 2], [-2, 2], [-2, 2]]

    density = gaussian_kde(cond_samples.numpy().T, bw_method="scott")

    X, Y = np.meshgrid(
        np.linspace(limits[0][0], limits[0][1], 50),
        np.linspace(limits[1][0], limits[1][1], 50),
    )
    positions = np.vstack([X.ravel(), Y.ravel()])
    sample_kde_grid = np.reshape(density(positions).T, X.shape)

    # Evaluate the conditional with eval_conditional_density.
    eval_grid = analysis.eval_conditional_density(
        posterior,
        condition=samples[0],
        dim1=dim_to_sample_1,
        dim2=dim_to_sample_2,
        limits=torch.tensor([[-2, 2], [-2, 2], [-2, 2]]),
    )

    # Compare the two densities.
    sample_kde_grid = sample_kde_grid / np.sum(sample_kde_grid)
    eval_grid = eval_grid / torch.sum(eval_grid)

    error = np.abs(sample_kde_grid - eval_grid.numpy())

    max_err = np.max(error)
    assert max_err < 0.0027
