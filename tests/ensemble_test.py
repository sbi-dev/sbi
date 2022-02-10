# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import infer, prepare_for_sbi
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from tests.test_utils import check_c2st, get_dkl_gaussian_prior


@pytest.mark.slow
@pytest.mark.parametrize(
    "inference_method",
    [
        "SNLE_A",
        "SNRE_A",
        "SNPE_C",
    ],
)
def test_c2st_posterior_ensemble_on_linearGaussian(inference_method):
    """Test whether NeuralPosteriorEnsemble infers well a simple example with available
    ground truth.

    """

    num_trials = 1
    num_dim = 2
    x_o = zeros(num_trials, num_dim)
    num_samples = 1000
    num_simulations = 4000 if inference_method == "SNRE_A" else 2000

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

    # train ensemble components
    ensemble_size = 2
    posteriors = [
        infer(simulator, prior, inference_method, num_simulations)
        for i in range(ensemble_size)
    ]

    # create ensemble
    posterior = NeuralPosteriorEnsemble(posteriors)
    posterior.set_default_x(x_o)

    # test sampling and evaluation.
    if inference_method == "SNLE_A" or inference_method == "SNRE_A":
        samples = posterior.sample(
            (num_samples,), num_chains=20, method="slice_np_vectorized"
        )
    else:
        samples = posterior.sample((num_samples,))
    _ = posterior.potential(samples)

    # Compute the c2st and assert it is near chance level of 0.5.
    check_c2st(
        samples, target_samples, alg="{} posterior ensemble".format(inference_method)
    )

    map_ = posterior.map(init_method=samples, show_progress_bars=False)
    assert ((map_ - gt_posterior.mean) ** 2).sum() < 0.5

    # Checks for log_prob()
    # For the Gaussian prior, we compute the KLd between ground truth and posterior.
    # This step is skipped for NLE since the probabilities are not normalised.
    if "snpe" in inference_method.lower() or "snre" in inference_method.lower():
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
