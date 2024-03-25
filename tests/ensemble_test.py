# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import SNLE_A, SNPE_C, SNRE_A
from sbi.inference.posteriors import EnsemblePosterior
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from tests.test_utils import check_c2st, get_dkl_gaussian_prior


def test_import_before_deprecation():
    with pytest.warns(DeprecationWarning):
        from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

        num_simulations = 100

        likelihood_shift = -1.0 * ones(2)
        likelihood_cov = 0.3 * eye(2)

        prior_mean = zeros(2)
        prior_cov = eye(2)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

        def simulator(theta):
            return linear_gaussian(theta, likelihood_shift, likelihood_cov)

        theta = prior.sample((num_simulations,))
        x = simulator(theta)
        inferer = SNPE_C(prior)
        inferer.append_simulations(theta, x).train(max_num_epochs=1)
        posterior = inferer.build_posterior()

        # create ensemble
        posterior = NeuralPosteriorEnsemble([posterior])


@pytest.mark.slow
@pytest.mark.parametrize(
    "inference_method, num_trials",
    (
        (SNPE_C, 1),
        pytest.param(SNPE_C, 5, marks=pytest.mark.xfail),
        pytest.param(SNLE_A, 1, marks=pytest.mark.mcmc),
        pytest.param(SNLE_A, 5, marks=pytest.mark.mcmc),
        pytest.param(SNRE_A, 1, marks=pytest.mark.mcmc),
        pytest.param(SNRE_A, 5, marks=pytest.mark.mcmc),
    ),
)
def test_c2st_posterior_ensemble_on_linearGaussian(
    inference_method, num_trials, mcmc_params_accurate: dict
):
    """Test whether EnsemblePosterior infers well a simple example with available
    ground truth.
    """

    num_dim = 2
    ensemble_size = 2
    x_o = zeros(num_trials, num_dim)
    num_samples = 500
    num_simulations = 2000

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

    # train ensemble components
    posteriors = []
    for _ in range(ensemble_size):
        theta = prior.sample((num_simulations,))
        x = simulator(theta)
        inferer = inference_method(prior)
        inferer.append_simulations(theta, x).train(
            training_batch_size=100,
            max_num_epochs=1 if inference_method == "SNPE" and num_trials > 1 else 100,
        )
        posteriors.append(inferer.build_posterior())

    # create ensemble
    posterior = EnsemblePosterior(posteriors)
    posterior.set_default_x(x_o)

    # test sampling and evaluation.
    if isinstance(inferer, (SNLE_A, SNRE_A)):
        samples = posterior.sample(
            (num_samples,),
            method="slice_np_vectorized",
            **mcmc_params_accurate,
        )
    else:
        samples = posterior.sample((num_samples,))
    _ = posterior.potential(samples)

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(
        samples,
        target_samples,
        alg="{} posterior ensemble".format(inference_method.__name__),
    )

    map_ = posterior.map(init_method=samples, show_progress_bars=False)
    assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5

    # Checks for log_prob()
    # For the Gaussian prior, we compute the KLd between ground truth and posterior.
    # This step is skipped for NLE since the probabilities are not normalised.
    if isinstance(inferer, (SNPE_C, SNRE_A)):
        dkl = get_dkl_gaussian_prior(
            posterior,
            x_o[0],
            likelihood_shift,
            likelihood_cov,
            prior_mean,
            prior_cov,
            num_samples=num_samples,
        )
        max_dkl = 0.15
        assert (
            dkl < max_dkl
        ), f"D-KL={dkl} is more than 2 stds above the average performance."

    # test individual log_prob and map
    posterior.log_prob(samples, individually=True)
