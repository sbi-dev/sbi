# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from pyro.infer.mcmc import MCMC
from torch import eye, zeros, Tensor
from torch.distributions import MultivariateNormal

from sbi.inference import (
    SNL,
    MCMCPosterior,
    likelihood_estimator_based_potential,
    prepare_for_sbi,
    simulate_for_sbi,
)
from sbi.samplers.mcmc import PyMCSampler, SliceSamplerSerial, SliceSamplerVectorized
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian


@pytest.mark.parametrize(
    "sampling_method",
    (
        "slice_np",
        "slice_np_vectorized",
        "nuts_pyro",
        "hmc_pyro",
        "nuts_pymc",
        "hmc_pymc",
        "slice_pymc",
    ),
)
def test_api_posterior_sampler_set(
    sampling_method: str,
    set_seed,
    num_dim: int = 2,
    num_samples: int = 42,
    num_trials: int = 2,
    num_simulations: int = 10,
):
    """Runs SNL and checks that posterior_sampler is correctly set."""
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
    samples = posterior.sample(
        sample_shape=(num_samples, num_chains),
        x=x_o,
        mcmc_parameters={
            "thin": 2,
            "num_chains": num_chains,
            "init_strategy": "prior",
            "warmup_steps": 10,
        },
    )
    assert isinstance(samples, Tensor)
    assert samples.shape == (num_samples, 1, num_dim)  # TODO check the 2nd dim =? 1

    if "pyro" in sampling_method:
        assert type(posterior.posterior_sampler) is MCMC
    elif "pymc" in sampling_method:
        assert type(posterior.posterior_sampler) is PyMCSampler
    elif sampling_method == "slice_np":
        assert type(posterior.posterior_sampler) is SliceSamplerSerial
    else:  # sampling_method == "slice_np_vectorized"
        assert type(posterior.posterior_sampler) is SliceSamplerVectorized
