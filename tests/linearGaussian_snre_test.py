# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils
from sbi.inference import (
    AALR,
    BNRE,
    SNRE_A,
    SNRE_B,
    SNRE_C,
    ImportanceSamplingPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
    ratio_estimator_based_potential,
    simulate_for_sbi,
)
from sbi.inference.snre.snre_base import RatioEstimator
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


@pytest.mark.mcmc
@pytest.mark.parametrize("num_dim", (1,))  # dim 3 is tested below.
@pytest.mark.parametrize("snre_method", (SNRE_B, SNRE_C))
def test_api_snre_multiple_trials_and_rounds_map(
    num_dim: int,
    snre_method: RatioEstimator,
    mcmc_params_fast: dict,
    num_rounds: int = 2,
    num_samples: int = 12,
    num_simulations: int = 100,
):
    """Test SNRE API with 2 rounds, different priors num trials and MAP."""
    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))

    simulator = diagonal_linear_gaussian
    inference = snre_method(prior=prior, classifier="mlp", show_progress_bars=False)

    proposals = [prior]
    for _ in range(num_rounds):
        theta, x = simulate_for_sbi(
            simulator,
            proposals[-1],
            num_simulations,
            simulation_batch_size=num_simulations,
        )
        inference.append_simulations(theta, x).train(
            training_batch_size=100, max_num_epochs=2
        )
        for num_trials in [1, 3]:
            x_o = zeros((num_trials, num_dim))
            posterior = inference.build_posterior(
                mcmc_method="slice_np_vectorized",
                mcmc_parameters=mcmc_params_fast,
            ).set_default_x(x_o)
            posterior.sample(sample_shape=(num_samples,))
        proposals.append(posterior)
        posterior.map(num_iter=1)


@pytest.mark.mcmc
@pytest.mark.parametrize("snre_method", (SNRE_B, SNRE_C))
def test_c2st_sre_on_linearGaussian(
    snre_method: RatioEstimator, mcmc_params_accurate: dict
):
    """Test whether SRE infers well a simple example with available ground truth.

    This example has different number of parameters theta than number of x. This test
    also acts as the only functional test for SRE not marked as slow.

    """

    theta_dim = 3
    x_dim = 2
    discard_dims = theta_dim - x_dim
    num_samples = 500
    num_simulations = 2100

    likelihood_shift = -1.0 * ones(
        x_dim
    )  # likelihood_mean will be likelihood_shift+theta
    likelihood_cov = 0.3 * eye(x_dim)

    prior_mean = zeros(theta_dim)
    prior_cov = eye(theta_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    def simulator(theta):
        return linear_gaussian(
            theta, likelihood_shift, likelihood_cov, num_discarded_dims=discard_dims
        )

    inference = snre_method(classifier="resnet", show_progress_bars=False)

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=100
    )
    ratio_estimator = inference.append_simulations(theta, x).train()

    x_o = zeros(1, x_dim)
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
        method="slice_np_vectorized",
        **mcmc_params_accurate,
    )
    samples = posterior.sample((num_samples,))

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(samples, target_samples, alg=f"{snre_method.__name__}")


@pytest.mark.mcmc
@pytest.mark.slow
@pytest.mark.parametrize("snre_method", (SNRE_A, SNRE_B, SNRE_C, BNRE))
@pytest.mark.parametrize("prior_str", ("gaussian", "uniform"))
@pytest.mark.parametrize("num_trials", (3,))  # num_trials=1 is tested above.
def test_c2st_snre_variants_on_linearGaussian_with_multiple_trials(
    snre_method: RatioEstimator,
    prior_str: str,
    num_trials: int,
    mcmc_params_accurate: dict,
):
    """Test C2ST and MAP accuracy of SNRE variants on linear gaussian.

    Args:
        num_dim: parameter dimension of the gaussian model
        prior_str: one of "gaussian" or "uniform"

    """

    num_dim = 2
    num_simulations = 1750
    num_samples = 500
    x_o = zeros(num_trials, num_dim)

    train_kwargs = {"training_batch_size": 100}
    if snre_method == BNRE:
        train_kwargs["regularization_strength"] = 20

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

    kwargs = dict(
        classifier="resnet",
        show_progress_bars=False,
    )

    inference = snre_method(**kwargs)

    # Should use default `num_atoms=10` for SRE; `num_atoms=2` for AALR
    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=num_simulations
    )
    ratio_estimator = inference.append_simulations(theta, x).train(**train_kwargs)
    potential_fn, theta_transform = ratio_estimator_based_potential(
        ratio_estimator=ratio_estimator, prior=prior, x_o=x_o
    )
    posterior = MCMCPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        proposal=prior,
        method="slice_np_vectorized",
        **mcmc_params_accurate,
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
        samples,
        target_samples,
        alg=f"snre-{prior_str}-{snre_method.__name__}-{num_trials}trials",
    )

    map_ = posterior.map(num_init_samples=1_000, init_method="proposal")

    # Checks for log_prob()
    if prior_str == "gaussian" and isinstance(snre_method, (AALR, BNRE)):
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
@pytest.mark.parametrize("snre_method", (SNRE_B, SNRE_C))
def test_c2st_multi_round_snr_on_linearGaussian_vi(
    num_trials: int, snre_method: RatioEstimator
):
    """Test C2ST accuracy of 2-round-SNRE with variational inference sampling."""

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

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = snre_method(show_progress_bars=False)

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
        pytest.param("slice_np", "gaussian", marks=pytest.mark.mcmc),
        pytest.param("slice_np", "uniform", marks=pytest.mark.mcmc),
        pytest.param("slice_np_vectorized", "gaussian", marks=pytest.mark.mcmc),
        pytest.param("slice_np_vectorized", "uniform", marks=pytest.mark.mcmc),
        pytest.param("slice", "gaussian", marks=pytest.mark.mcmc),
        pytest.param("slice", "uniform", marks=pytest.mark.mcmc),
        pytest.param("nuts", "gaussian", marks=pytest.mark.mcmc),
        pytest.param("nuts", "uniform", marks=pytest.mark.mcmc),
        pytest.param("hmc", "gaussian", marks=pytest.mark.mcmc),
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
        ("importance", "uniform"),
        ("importance", "gaussian"),
    ),
)
def test_api_sre_sampling_methods(
    sampling_method: str, prior_str: str, mcmc_params_fast: dict
):
    """Test leakage correction both for MCMC and rejection sampling.

    Args:
        mcmc_method: which mcmc method to use for sampling
        prior_str: one of "gaussian" or "uniform"

    """
    num_dim = 2
    num_samples = 10
    num_trials = 2
    num_simulations = 100
    x_o = zeros((num_trials, num_dim))
    # Test for multiple chains is cheap when vectorized.

    if sampling_method == "rejection":
        sample_with = "rejection"
    elif (
        "slice" in sampling_method
        or "nuts" in sampling_method
        or "hmc" in sampling_method
    ):
        sample_with = "mcmc"
    elif sampling_method == "importance":
        sample_with = "importance"
    else:
        sample_with = "vi"

    if prior_str == "gaussian":
        prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    else:
        prior = utils.BoxUniform(-ones(num_dim), ones(num_dim))

    simulator = diagonal_linear_gaussian

    inference = SNRE_B(classifier="resnet", show_progress_bars=False)

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=num_simulations
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
            **mcmc_params_fast,
        )
    elif sample_with == "importance":
        posterior = ImportanceSamplingPosterior(
            potential_fn,
            proposal=prior,
            theta_transform=theta_transform,
        )
    else:
        posterior = VIPosterior(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            vi_method=sampling_method,
        )
        posterior.train(max_num_iters=10)

    posterior.sample(sample_shape=(num_samples,))
