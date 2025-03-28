# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import sys

import numpy as np
import pymc
import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import Uniform

from sbi.inference import (
    NLE,
    MCMCPosterior,
    likelihood_estimator_based_potential,
)
from sbi.inference.posteriors.mcmc_posterior import build_from_potential
from sbi.neural_nets import likelihood_nn
from sbi.samplers.mcmc.pymc_wrapper import PyMCSampler
from sbi.samplers.mcmc.slice_numpy import (
    SliceSampler,
    SliceSamplerSerial,
    SliceSamplerVectorized,
)
from sbi.simulators.linear_gaussian import (
    diagonal_linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils import BoxUniform
from sbi.utils.metrics import check_c2st
from sbi.utils.user_input_checks import process_prior


@pytest.mark.mcmc
@pytest.mark.parametrize("num_dim", (1, 2))
def test_c2st_slice_np_on_Gaussian(
    num_dim: int, warmup: int = 100, num_samples: int = 500
):
    """Test MCMC on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
    """
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
    num_samples = 1000
    warmup = mcmc_params_accurate["warmup_steps"]
    num_chains = (
        mcmc_params_accurate["num_chains"]
        if slice_sampler is SliceSamplerVectorized
        else 1
    )
    thin = mcmc_params_accurate["thin"]

    likelihood_shift = -5.0 * ones(num_dim)
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
@pytest.mark.slow
@pytest.mark.parametrize("step", ("nuts", "hmc", "slice"))
@pytest.mark.parametrize("num_chains", (1, 3))
def test_c2st_pymc_sampler_on_Gaussian(
    step: str,
    num_chains: int,
    num_dim: int = 2,
    num_samples: int = 1100,  # Had to change from 1000 to 1100 in #1247.
    warmup: int = 100,
):
    """Test PyMC on Gaussian, comparing to ground truth target via c2st."""
    likelihood_shift = -5.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    def lp_f(x, track_gradients=True):
        with torch.set_grad_enabled(track_gradients):
            return target_distribution.log_prob(x)

    sampler = PyMCSampler(
        potential_fn=lp_f,
        initvals=np.zeros((num_chains, num_dim)).astype(np.float32),
        step=step,
        draws=(int(num_samples / num_chains)),  # PyMC does not use thinning
        tune=warmup,
        chains=num_chains,
    )
    samples = sampler.run()
    assert samples.shape == (
        num_chains,
        int(num_samples / num_chains),
        num_dim,
    )
    samples = samples.reshape(-1, num_dim)

    samples = torch.as_tensor(samples, dtype=torch.float32)
    alg = f"pymc_{step}"

    check_c2st(samples, target_samples, alg=alg)


@pytest.mark.mcmc
@pytest.mark.parametrize(
    "method",
    (
        "nuts_pyro",
        "hmc_pyro",
        pytest.param(
            "nuts_pymc",
            marks=pytest.mark.skipif(
                condition=sys.version_info >= (3, 10) and pymc.__version__ >= "5.20.1",
                reason="Inconsistent behaviour with pymc>=5.20.1 and python>=3.10",
            ),
        ),
        "hmc_pymc",
        "slice_pymc",
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

    simulator = diagonal_linear_gaussian
    density_estimator = likelihood_nn("maf", num_transforms=3)
    inference = NLE(density_estimator=density_estimator, show_progress_bars=False)
    prior, *_ = process_prior(prior)
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
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

    assert hasattr(idata, "posterior"), (
        f"`MCMCPosterior.get_arviz_inference_data()` for method {method} "
        f"returned invalid InferenceData. Must contain key 'posterior', "
        f"but found only {list(idata.keys())}"
    )
    samples = getattr(idata.posterior, posterior.param_name).data
    samples = samples.reshape(-1, samples.shape[-1])[:: mcmc_params_fast["thin"]][
        :num_samples
    ]
    assert samples.shape == (
        num_samples,
        num_dim,
    ), (
        f"MCMC samples for method {method} have incorrect shape (n_samples, n_dims). "
        f"Expected {(num_samples, num_dim)}, got {samples.shape}"
    )


@pytest.mark.mcmc
def test_direct_mcmc_unconditional():
    "Test MCMCPosterior from user defined potential (unconditional)"
    num_samples = 100
    theta_dim = 2

    prior = BoxUniform(low=-2 * torch.ones(theta_dim), high=2 * torch.ones(theta_dim))

    def potential_fn(theta: np.ndarray) -> np.ndarray:
        # Example: a 2D Gaussian with mean=[0,0], identity covariance
        return -0.5 * (theta**2).sum(axis=-1)

    mcmc_posterior = build_from_potential(potential_fn, prior)

    # test sampling
    samples = mcmc_posterior.sample(
        (num_samples,), num_chains=10, warmup_steps=50, thin=10
    )

    assert samples.shape == (num_samples, theta_dim), (
        f"MCMC samples have incorrect shape (n_samples, n_dims). "
        f"Expected {(num_samples, theta_dim)}, got {samples.shape}"
    )

    # test potential evaluation
    dist = torch.distributions.MultivariateNormal(
        torch.zeros(theta_dim), torch.eye(theta_dim)
    )
    samples = dist.sample((num_samples,))
    log_p = mcmc_posterior.potential(samples)

    assert log_p.shape == (num_samples,), (
        f"Potential evals have incorrect shape. "
        f"Expected ({num_samples}), got {log_p.shape}"
    )


@pytest.mark.mcmc
def test_direct_mcmc_conditional():
    "Test MCMCPosterior from user defined potential (conditional)"
    theta_dim = 2
    num_samples = 100
    num_batches = 5
    num_samples_batch = num_samples // num_batches

    prior = BoxUniform(low=-2 * torch.ones(theta_dim), high=2 * torch.ones(theta_dim))

    def potential_fn(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        # Example: a 2D Gaussian with mean=[0,0], variance conditioned on x
        return -x * (theta**2).sum(axis=-1)

    # test sampling
    x = torch.tensor([0.5])
    mcmc_posterior = build_from_potential(potential_fn, prior, x=x)
    samples = mcmc_posterior.sample(
        (num_samples,), num_chains=10, warmup_steps=50, thin=10
    )

    assert samples.shape == (num_samples, theta_dim), (
        f"MCMC samples have incorrect shape (n_chains, n_samples, n_dims). "
        f"Expected {(num_samples, theta_dim)}, got {samples.shape}"
    )

    # test batched sampling
    x_batch = torch.linspace(0.1, 0.9, num_batches).unsqueeze(1)
    samples_batched = mcmc_posterior.sample_batched(
        (num_samples_batch,), x=x_batch, num_chains=10, warmup_steps=50, thin=10
    )
    assert samples_batched.shape == (num_samples_batch, num_batches, theta_dim), (
        f"MCMC samples have incorrect shape (n_samples, n_batches, n_dims). "
        f"Expected {(num_samples, num_batches, theta_dim)}, got {samples.shape}"
    )

    # test potential evaluation
    dist = torch.distributions.MultivariateNormal(
        torch.zeros(theta_dim), torch.eye(theta_dim)
    )
    theta_samples = dist.sample((num_samples,))
    x_samples = torch.rand((num_samples,))
    log_p = mcmc_posterior.potential(theta_samples, x_samples)

    assert log_p.shape == (
        1,
        num_samples,
    ), (
        f"Potential evals have incorrect shape. "
        f"Expected (1, {num_samples}), got {log_p.shape}"
    )
