# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import (
    AALR,
    SNRE_B,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
    prepare_for_sbi,
    ratio_estimator_based_potential,
    simulate_for_sbi,
)
from sbi.simulators.linear_gaussian import (
    diagonal_linear_gaussian,
    linear_gaussian,
    samples_true_posterior_linear_gaussian_mvn_prior_different_dims,
    samples_true_posterior_linear_gaussian_uniform_prior,
    true_posterior_linear_gaussian_mvn_prior,
)
from tests.test_utils import (
    check_c2st,
    get_dkl_gaussian_prior,
    get_prob_outside_uniform_prior,
)


@pytest.mark.parametrize("num_dim", (1, 3))
def test_api_sre_on_linearGaussian(num_dim: int):
    """Test inference API of SRE with linear Gaussian model.

    Avoids intense computation for fast testing of API etc.

    Args:
        num_dim: parameter dimension of the Gaussian model
    """

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))

    simulator, prior = prepare_for_sbi(diagonal_linear_gaussian, prior)
    inference = SNRE_B(
        classifier="resnet",
        show_progress_bars=False,
    )

    theta, x = simulate_for_sbi(simulator, prior, 1000, simulation_batch_size=50)
    ratio_estimator = inference.append_simulations(theta, x).train(max_num_epochs=5)

    for num_trials in [1, 2]:
        x_o = zeros(num_trials, num_dim)
        potential_fn, theta_transform = ratio_estimator_based_potential(
            ratio_estimator=ratio_estimator, prior=prior, x_o=x_o
        )
        posterior = MCMCPosterior(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            proposal=prior,
            num_chains=2,
        )
        posterior.sample(sample_shape=(10,))


def test_c2st_sre_on_linearGaussian():
    """Test whether SRE infers well a simple example with available ground truth.

    This example has different number of parameters theta than number of x. This test
    also acts as the only functional test for SRE not marked as slow.

    """

    theta_dim = 3
    x_dim = 2
    discard_dims = theta_dim - x_dim
    num_samples = 1000
    num_simulations = 2100

    likelihood_shift = -1.0 * ones(
        x_dim
    )  # likelihood_mean will be likelihood_shift+theta
    likelihood_cov = 0.3 * eye(x_dim)

    prior_mean = zeros(theta_dim)
    prior_cov = eye(theta_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    simulator, prior = prepare_for_sbi(
        lambda theta: linear_gaussian(
            theta, likelihood_shift, likelihood_cov, num_discarded_dims=discard_dims
        ),
        prior,
    )
    inference = SNRE_B(
        classifier="resnet",
        show_progress_bars=False,
    )

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=100
    )
    ratio_estimator = inference.append_simulations(theta, x).train()

    num_trials = 1
    x_o = zeros(num_trials, x_dim)
    target_samples = samples_true_posterior_linear_gaussian_mvn_prior_different_dims(
        x_o,
        likelihood_shift,
        likelihood_cov,
        prior_mean,
        prior_cov,
        num_discarded_dims=discard_dims,
        num_samples=num_samples,
    )
    potential_fn, theta_transform = ratio_estimator_based_potential(
        ratio_estimator=ratio_estimator, prior=prior, x_o=x_o
    )
    posterior = MCMCPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        proposal=prior,
        thin=5,
        num_chains=2,
    )
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"snre-{num_trials}trials")


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_dim, num_trials, prior_str, method_str",
    (
        (2, 5, "gaussian", "sre"),
        (1, 1, "gaussian", "sre"),
        (2, 1, "uniform", "sre"),
        (2, 5, "gaussian", "aalr"),
    ),
)
def test_c2st_sre_variants_on_linearGaussian(
    num_dim: int,
    num_trials: int,
    prior_str: str,
    method_str: str,
):
    """Test c2st accuracy of inference with SRE on linear Gaussian model.

    Args:
        num_dim: parameter dimension of the gaussian model
        prior_str: one of "gaussian" or "uniform"

    """

    x_o = zeros(num_trials, num_dim)
    num_samples = 500
    num_simulations = 2600 if num_trials == 1 else 40500

    # `likelihood_mean` will be `likelihood_shift + theta`.
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.8 * eye(num_dim)

    if prior_str == "gaussian":
        prior_mean = zeros(num_dim)
        prior_cov = eye(num_dim)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    else:
        prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    simulator, prior = prepare_for_sbi(simulator, prior)
    kwargs = dict(
        classifier="resnet",
        show_progress_bars=False,
    )

    inference = SNRE_B(**kwargs) if method_str == "sre" else AALR(**kwargs)

    # Should use default `num_atoms=10` for SRE; `num_atoms=2` for AALR
    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=50
    )
    ratio_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = ratio_estimator_based_potential(
        ratio_estimator=ratio_estimator, prior=prior, x_o=x_o
    )
    posterior = MCMCPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        proposal=prior,
        method="slice_np_vectorized",
        thin=5,
        num_chains=5,
    )
    samples = posterior.sample(sample_shape=(num_samples,))

    # Get posterior samples.
    if prior_str == "gaussian":
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
    else:
        target_samples = samples_true_posterior_linear_gaussian_uniform_prior(
            x_o, likelihood_shift, likelihood_cov, prior=prior, num_samples=num_samples
        )

    # Check performance based on c2st accuracy.
    check_c2st(
        samples, target_samples, alg=f"sre-{prior_str}-{method_str}-{num_trials}trials"
    )

    map_ = posterior.map(num_init_samples=1_000, init_method="proposal")

    # Checks for log_prob()
    if prior_str == "gaussian" and method_str == "aalr":
        # For the Gaussian prior, we compute the KLd between ground truth and
        # posterior. We can do this only if the classifier_loss was as described in
        # Hermans et al. 2020 ('aalr') since Durkan et al. 2020 version only allows
        # evaluation up to a constant.
        # For the Gaussian prior, we compute the KLd between ground truth and posterior
        dkl = get_dkl_gaussian_prior(
            posterior, x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )

        max_dkl = 0.15

        assert (
            dkl < max_dkl
        ), f"KLd={dkl} is more than 2 stds above the average performance."

        assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5

    if prior_str == "uniform":
        # Check whether the returned probability outside of the support is zero.
        posterior_prob = get_prob_outside_uniform_prior(posterior, prior, num_dim)
        assert (
            posterior_prob == 0.0
        ), "The posterior probability outside of the prior support is not zero"

        assert ((map_ - ones(num_dim)) ** 2).sum() < 0.5


@pytest.mark.slow
@pytest.mark.parametrize("num_trials", (1, 3))
def test_c2st_multi_round_snr_on_linearGaussian_vi(num_trials: int):
    """Test SNL on linear Gaussian, comparing to ground truth posterior via c2st."""

    num_dim = 2
    x_o = zeros((num_trials, num_dim))
    num_samples = 500
    num_simulations_per_round = 1000 * num_trials

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = gt_posterior.sample((num_samples,))

    simulator, prior = prepare_for_sbi(
        lambda theta: linear_gaussian(theta, likelihood_shift, likelihood_cov), prior
    )
    inference = SNRE_B(show_progress_bars=False)

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations_per_round, simulation_batch_size=50
    )
    ratio_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = ratio_estimator_based_potential(
        prior=prior, ratio_estimator=ratio_estimator, x_o=x_o
    )
    posterior1 = VIPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
    )
    posterior1.train()

    theta, x = simulate_for_sbi(
        simulator, posterior1, num_simulations_per_round, simulation_batch_size=50
    )
    ratio_estimator = inference.append_simulations(theta, x).train()
    potential_fn, theta_transform = ratio_estimator_based_potential(
        prior=prior, ratio_estimator=ratio_estimator, x_o=x_o
    )
    posterior = VIPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        q=posterior1,
    )
    posterior.train()

    samples = posterior.sample(sample_shape=(num_samples,))

    # Check performance based on c2st accuracy.
    check_c2st(samples, target_samples, alg="multi-round-snl")


@pytest.mark.slow
@pytest.mark.parametrize(
    "sampling_method, prior_str",
    (
        ("slice_np", "gaussian"),
        ("slice_np", "uniform"),
        ("slice_np_vectorized", "gaussian"),
        ("slice_np_vectorized", "uniform"),
        ("slice", "gaussian"),
        ("slice", "uniform"),
        ("nuts", "gaussian"),
        ("nuts", "uniform"),
        ("hmc", "gaussian"),
        ("rejection", "uniform"),
        ("rejection", "gaussian"),
        ("rKL", "uniform"),
        ("fKL", "uniform"),
        ("IW", "uniform"),
        ("alpha", "uniform"),
        ("rKL", "gaussian"),
        ("fKL", "gaussian"),
        ("IW", "gaussian"),
        ("alpha", "gaussian"),
    ),
)
def test_api_sre_sampling_methods(sampling_method: str, prior_str: str):
    """Test leakage correction both for MCMC and rejection sampling.

    Args:
        mcmc_method: which mcmc method to use for sampling
        prior_str: one of "gaussian" or "uniform"

    """
    num_dim = 2
    num_samples = 10
    num_trials = 2
    num_simulations = 2100
    x_o = zeros((num_trials, num_dim))
    # Test for multiple chains is cheap when vectorized.
    num_chains = 3 if sampling_method == "slice_np_vectorized" else 1
    if sampling_method == "rejection":
        sample_with = "rejection"
    elif (
        "slice" in sampling_method
        or "nuts" in sampling_method
        or "hmc" in sampling_method
    ):
        sample_with = "mcmc"
    else:
        sample_with = "vi"

    if prior_str == "gaussian":
        prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    else:
        prior = utils.BoxUniform(-1.0 * ones(num_dim), ones(num_dim))

    simulator, prior = prepare_for_sbi(diagonal_linear_gaussian, prior)
    inference = SNRE_B(classifier="resnet", show_progress_bars=False)

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=50
    )
    ratio_estimator = inference.append_simulations(theta, x).train(max_num_epochs=5)
    potential_fn, theta_transform = ratio_estimator_based_potential(
        ratio_estimator=ratio_estimator, prior=prior, x_o=x_o
    )
    if sample_with == "rejection":
        posterior = RejectionPosterior(potential_fn=potential_fn, proposal=prior)
    elif (
        "slice" in sampling_method
        or "nuts" in sampling_method
        or "hmc" in sampling_method
    ):
        posterior = MCMCPosterior(
            potential_fn,
            proposal=prior,
            theta_transform=theta_transform,
            method=sampling_method,
            thin=3,
            num_chains=num_chains,
        )
    else:
        posterior = VIPosterior(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            vi_method=sampling_method,
        )
        posterior.train(max_num_iters=10)

    posterior.sample(sample_shape=(num_samples,))
