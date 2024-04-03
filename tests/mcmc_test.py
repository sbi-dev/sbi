# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import Uniform

from sbi.inference import (
    SNLE,
    MCMCPosterior,
    likelihood_estimator_based_potential,
    simulate_for_sbi,
)
from sbi.neural_nets import likelihood_nn
from sbi.samplers.mcmc.slice_numpy import (
    SliceSampler,
    SliceSamplerSerial,
    SliceSamplerVectorized,
)
from sbi.simulators.linear_gaussian import (
    diagonal_linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.user_input_checks import (
    process_prior,
)
from tests.test_utils import check_c2st


@pytest.mark.mcmc
@pytest.mark.parametrize("num_dim", (1, 2))
def test_c2st_slice_np_on_Gaussian(num_dim: int):
    """Test MCMC on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model

    """
    warmup = 100
    num_samples = 500

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    def lp_f(x):
        return target_distribution.log_prob(torch.as_tensor(x, dtype=torch.float32))

    sampler = SliceSampler(
        lp_f=lp_f,
        x=np.zeros((num_dim,)).astype(np.float32),
        tuning=warmup,
    )
    warmup_samples = sampler.gen(warmup)
    assert warmup_samples.shape == (warmup, num_dim)

    samples = sampler.gen(num_samples)
    assert samples.shape == (num_samples, num_dim)

    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.mcmc
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("slice_sampler", (SliceSamplerVectorized, SliceSamplerSerial))
@pytest.mark.parametrize("num_workers", (1, 2))
def test_c2st_slice_np_vectorized_parallelized_on_Gaussian(
    num_dim: int, slice_sampler, num_workers: int, mcmc_params_accurate: dict
):
    """Test MCMC on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model

    """
    num_samples = 500
    warmup = mcmc_params_accurate["warmup_steps"]
    num_chains = (
        mcmc_params_accurate["num_chains"]
        if slice_sampler is SliceSamplerVectorized
        else 1
    )
    thin = mcmc_params_accurate["thin"]

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    def lp_f(x):
        return target_distribution.log_prob(torch.as_tensor(x, dtype=torch.float32))

    sampler = slice_sampler(
        log_prob_fn=lp_f,
        init_params=np.zeros((num_chains, num_dim)).astype(np.float32),
        tuning=warmup,
        thin=thin,
        num_chains=num_chains,
        num_workers=num_workers,
    )
    samples = sampler.run(thin * (warmup + int(num_samples / num_chains)))
    assert samples.shape == (
        num_chains,
        warmup + int(num_samples / num_chains),
        num_dim,
    )
    samples = samples[:, warmup:, :]
    samples = samples.reshape(-1, num_dim)

    samples = torch.as_tensor(samples, dtype=torch.float32)

    alg = {
        SliceSamplerVectorized: "slice_np_vectorized",
        SliceSamplerSerial: "slice_np",
    }[slice_sampler]

    check_c2st(samples, target_samples, alg=alg)


@pytest.mark.mcmc
@pytest.mark.parametrize(
    "method",
    (
        "nuts",
        "hmc",
        "slice",
        "slice_np",
        "slice_np_vectorized",
    ),
)
def test_getting_inference_diagnostics(method, mcmc_params_fast: dict):
    num_simulations = 100
    num_samples = 10
    num_dim = 2

    # Use composed prior to test MultipleIndependent case.
    prior = [
        Uniform(low=-ones(1), high=ones(1)),
        Uniform(low=-ones(1), high=ones(1)),
    ]

    prior, _, _ = process_prior(prior)
    simulator = diagonal_linear_gaussian
    density_estimator = likelihood_nn("maf", num_transforms=3)
    inference = SNLE(density_estimator=density_estimator, show_progress_bars=False)

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=num_simulations
    )
    likelihood_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=num_simulations, max_num_epochs=2
    )

    x_o = zeros((1, num_dim))
    potential_fn, theta_transform = likelihood_estimator_based_potential(
        prior=prior, likelihood_estimator=likelihood_estimator, x_o=x_o
    )
    posterior = MCMCPosterior(
        proposal=prior,
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        **mcmc_params_fast,
    )
    posterior.sample(
        sample_shape=(num_samples,),
        method=method,
    )
    idata = posterior.get_arviz_inference_data()

    az.plot_trace(idata)
