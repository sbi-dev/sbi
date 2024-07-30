# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
from pyro.infer.mcmc import MCMC
from torch import Tensor, eye, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import (
    SNL,
    MCMCPosterior,
    likelihood_estimator_based_potential,
)
from sbi.samplers.mcmc import PyMCSampler, SliceSamplerSerial, SliceSamplerVectorized
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian


@pytest.mark.mcmc
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
@pytest.mark.parametrize("num_chains", (1, pytest.param(3, marks=pytest.mark.slow)))
def test_api_posterior_sampler_set(
    sampling_method: str,
    num_chains: int,
    set_seed,
    mcmc_params_fast: dict,
    num_dim: int = 2,
    num_samples: int = 42,
    num_trials: int = 2,
    num_simulations: int = 10,
):
    """Runs SNL and checks that posterior_sampler is correctly set."""
    x_o = zeros((num_trials, num_dim))
    mcmc_params_fast["num_chains"] = num_chains

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = SNL(prior, show_progress_bars=False)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)
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
        mcmc_parameters={"init_strategy": "prior", **mcmc_params_fast},
    )
    assert isinstance(samples, Tensor)
    assert samples.shape == (num_samples, num_chains, num_dim)

    if "pyro" in sampling_method:
        assert type(posterior.posterior_sampler) is MCMC
    elif "pymc" in sampling_method:
        assert type(posterior.posterior_sampler) is PyMCSampler
    elif sampling_method == "slice_np":
        assert type(posterior.posterior_sampler) is SliceSamplerSerial
    else:  # sampling_method == "slice_np_vectorized"
        assert type(posterior.posterior_sampler) is SliceSamplerVectorized
