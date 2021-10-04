# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from pyro.infer.mcmc import MCMC
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import (
    SNL,
    MCMCPosterior,
    likelihood_estimator_based_potential,
    prepare_for_sbi,
    simulate_for_sbi,
)
from sbi.samplers.mcmc import SliceSamplerSerial, SliceSamplerVectorized
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian


@pytest.mark.parametrize(
    "sampling_method",
    (
        "slice_np",
        "slice_np_vectorized",
        "slice",
        "nuts",
        "hmc",
    ),
)
def test_api_posterior_sampler_set(sampling_method: str, set_seed):
    """Runs SNL and checks that posterior_sampler is correctly set.

    Args:
        mcmc_method: which mcmc method to use for sampling
        set_seed: fixture for manual seeding
    """

    num_dim = 2
    num_samples = 10
    num_trials = 2
    num_simulations = 10
    x_o = zeros((num_trials, num_dim))
    # Test for multiple chains is cheap when vectorized.
    num_chains = 3 if sampling_method in "slice_np_vectorized" else 1

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator, prior = prepare_for_sbi(diagonal_linear_gaussian, prior)
    inference = SNL(prior, show_progress_bars=False)

    theta, x = simulate_for_sbi(
        simulator, prior, num_simulations, simulation_batch_size=10
    )
    estimator = inference.append_simulations(theta, x).train(max_num_epochs=5)
    potential_fn, transform = likelihood_estimator_based_potential(
        estimator, prior, x_o
    )
    posterior = MCMCPosterior(
        potential_fn, theta_transform=transform, method=sampling_method, proposal=prior
    )

    assert posterior.posterior_sampler is None
    posterior.sample(
        sample_shape=(num_samples, num_chains),
        x=x_o,
        mcmc_parameters={
            "thin": 3,
            "num_chains": num_chains,
            "init_strategy": "prior",
        },
    )

    if sampling_method in ["slice", "hmc", "nuts"]:
        assert type(posterior.posterior_sampler) is MCMC
    elif sampling_method == "slice_np":
        assert type(posterior.posterior_sampler) is SliceSamplerSerial
    else:  # sampling_method == "slice_np_vectorized"
        assert type(posterior.posterior_sampler) is SliceSamplerVectorized
