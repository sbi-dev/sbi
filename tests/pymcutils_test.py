# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Minimal test for SBI-PyMC bridge using NLE on linear Gaussian model."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import NLE
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.metrics import check_c2st
from sbi.utils.pymcutils import neural_likelihood_to_pymc


@pytest.fixture(scope="module")
def linear_gaussian_setup():
    """Setup for linear Gaussian model."""
    num_dim = 3
    prior_mean = zeros(num_dim)
    prior_cov = 2.0 * eye(num_dim)
    prior = MultivariateNormal(prior_mean, prior_cov)
    likelihood_shift = zeros(num_dim)
    likelihood_cov = 1.0 * eye(num_dim)

    # Simulator: x = theta + N(0, I)
    def simulator(theta):
        # linear_gaussian expects theta with batch dimension
        return linear_gaussian(
            theta,
            likelihood_shift=likelihood_shift,
            likelihood_cov=likelihood_cov,  # Fixed observation noise
        )

    return {
        "num_dim": num_dim,
        "prior": prior,
        "simulator": simulator,
        "prior_mean": prior_mean,
        "prior_cov": prior_cov,
        "likelihood_shift": likelihood_shift,
        "likelihood_cov": likelihood_cov,
    }


@pytest.fixture(scope="module")
def get_trained_nle(linear_gaussian_setup):
    """Fixture to get a trained NLE instance."""
    num_simulations = 10000
    prior = linear_gaussian_setup["prior"]
    simulator = linear_gaussian_setup["simulator"]

    theta_train = prior.sample((num_simulations,))
    x_train = simulator(theta_train)

    inference = NLE(prior=prior, density_estimator="maf")
    inference.append_simulations(theta_train, x_train)
    inference.train()
    return inference


@pytest.mark.parametrize("method", [pm.Slice, pm.NUTS])
def test_nle_pymc_bridge_minimal(method, get_trained_nle, linear_gaussian_setup):
    """Minimal test: NLE-PyMC bridge on 3D linear Gaussian model.

    Tests different MCMC methods:
    - Slice: gradient-free, good for testing basic functionality
    - NUTS: gradient-based, adaptive HMC
    """

    # Observed data with multiple iid trials.
    true_theta = torch.tensor([1.0, -0.5, 0.3])  # 3D parameters
    num_trials = 10
    x_o = linear_gaussian_setup["simulator"](
        true_theta.unsqueeze(0).repeat(num_trials, 1)
    )
    prior_mean = linear_gaussian_setup["prior_mean"]
    prior_cov = linear_gaussian_setup["prior_cov"]
    num_dim = linear_gaussian_setup["num_dim"]
    likelihood_nn = get_trained_nle._neural_net

    true_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o,
        linear_gaussian_setup["likelihood_shift"],
        linear_gaussian_setup["likelihood_cov"],
        prior_mean,
        prior_cov,
    )

    prior_mean_np = prior_mean.numpy()
    prior_std_np = np.sqrt(np.diag(prior_cov.numpy()))
    num_posterior_samples = 1000
    num_chains = 4
    draws = 250
    tune = 500

    with pm.Model():
        # Prior in PyMC
        theta_pymc = pm.Normal(
            "theta", mu=prior_mean_np, sigma=prior_std_np, shape=num_dim
        )

        # Neural likelihood wrapper
        neural_likelihood_to_pymc(
            likelihood_nn=likelihood_nn,
            theta=theta_pymc,
            observed=x_o.numpy(),
            name="x",
        )

        trace = pm.sample(
            draws=draws,
            tune=tune,
            step=method(),
            chains=num_chains,
            cores=1,
            progressbar=True,
            return_inferencedata=True,
        )

    true_samples = true_posterior.sample((num_posterior_samples,))
    inferred_samples = torch.from_numpy(
        trace.posterior["theta"].values.reshape(-1, num_dim)
    ).float()

    if method == pm.NUTS:
        r_hat = pm.rhat(trace)["theta"].values
        assert np.all(r_hat < 1.1), f"R-hat values too high: {r_hat}"

    check_c2st(
        true_samples[:num_posterior_samples],
        inferred_samples[:num_posterior_samples],
        alg=f"nle-pymc-{method.__name__}",
        tol=0.1,
    )


@pytest.fixture(scope="module")
def scalar_gaussian_setup():
    """Setup for 1D linear Gaussian model (scalar theta)."""
    num_dim = 1
    prior_mean = zeros(num_dim)
    prior_cov = 2.0 * eye(num_dim)
    prior = MultivariateNormal(prior_mean, prior_cov)
    likelihood_shift = zeros(num_dim)
    likelihood_cov = 0.5 * eye(num_dim)

    def simulator(theta):
        return linear_gaussian(
            theta,
            likelihood_shift=likelihood_shift,
            likelihood_cov=likelihood_cov,
        )

    return {
        "num_dim": num_dim,
        "prior": prior,
        "simulator": simulator,
        "prior_mean": prior_mean,
        "prior_cov": prior_cov,
        "likelihood_shift": likelihood_shift,
        "likelihood_cov": likelihood_cov,
    }


@pytest.fixture(scope="module")
def get_trained_nle_scalar(scalar_gaussian_setup):
    """Fixture to get a trained NLE instance for 1D model."""
    num_simulations = 5000
    prior = scalar_gaussian_setup["prior"]
    simulator = scalar_gaussian_setup["simulator"]

    theta_train = prior.sample((num_simulations,))
    x_train = simulator(theta_train)

    inference = NLE(prior=prior, density_estimator="maf")
    inference.append_simulations(theta_train, x_train)
    inference.train()
    return inference


@pytest.mark.parametrize("method", [pm.Slice, pm.NUTS])
def test_scalar_theta_pymc(method, get_trained_nle_scalar, scalar_gaussian_setup):
    """Test that scalar theta parameters (shape ()) work correctly.

    PyMC scalar parameters have shape () and should be automatically handled
    by the NeuralLikelihoodOp without any special configuration.
    """
    # Generate observed data
    true_theta = torch.tensor([0.5])
    num_trials = 10
    x_o = scalar_gaussian_setup["simulator"](
        true_theta.unsqueeze(0).repeat(num_trials, 1)
    )
    prior_mean = scalar_gaussian_setup["prior_mean"]
    prior_cov = scalar_gaussian_setup["prior_cov"]
    likelihood_nn = get_trained_nle_scalar._neural_net

    # Compute true posterior for comparison
    true_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o,
        scalar_gaussian_setup["likelihood_shift"],
        scalar_gaussian_setup["likelihood_cov"],
        prior_mean,
        prior_cov,
    )

    prior_mean_np = float(prior_mean.numpy())
    prior_std_np = float(np.sqrt(prior_cov.numpy()[0, 0]))

    with pm.Model():
        theta_pymc = pm.Normal("theta", mu=prior_mean_np, sigma=prior_std_np)

        neural_likelihood_to_pymc(
            likelihood_nn=likelihood_nn,
            theta=theta_pymc,
            observed=x_o.numpy(),
            name="x",
        )

        trace = pm.sample(
            draws=500,
            tune=500,
            step=method(),
            chains=2,
            cores=1,
            progressbar=True,
            return_inferencedata=True,
        )

    num_posterior_samples = 500
    true_samples = true_posterior.sample((num_posterior_samples,))
    inferred_samples = torch.from_numpy(
        trace.posterior["theta"].values.reshape(-1, 1)
    ).float()

    if method == pm.NUTS:
        r_hat = pm.rhat(trace)["theta"].values
        assert r_hat < 1.1, f"R-hat too high: {r_hat}"

    check_c2st(
        true_samples[:num_posterior_samples],
        inferred_samples[:num_posterior_samples],
        alg=f"scalar-nle-pymc-{method.__name__}",
        tol=0.1,
    )


@pytest.fixture(scope="module")
def hierarchical_nle_setup():
    """Setup for hierarchical model: train NLE on single-observation-per-subject data.

    The NLE learns p(x | theta) where both theta and x are 1-dimensional.
    During training, each simulation represents a single subject with one observation.
    At inference time, this same NLE is applied to multiple subjects hierarchically.
    """
    num_simulations = 5000
    sigma_x = 0.5
    theta_train = torch.randn(num_simulations, 1)
    x_train = theta_train + sigma_x * torch.randn(num_simulations, 1)

    prior = MultivariateNormal(zeros(1), eye(1))
    inference = NLE(prior=prior, density_estimator="maf")
    inference.append_simulations(theta_train, x_train)
    inference.train()

    return {
        "likelihood_nn": inference._neural_net,
        "sigma_x": sigma_x,
    }


def flatten_pymc_samples(trace, var_names: list[str]) -> torch.Tensor:
    """Flatten PyMC trace samples for c2st comparison.

    Args:
        trace: PyMC InferenceData trace
        var_names: List of variable names to include

    Returns:
        Tensor of shape (num_samples, total_params) with all samples concatenated
    """
    samples = []
    for var in sorted(var_names):
        # trace.posterior[var] has shape (chain, draw, *var_shape)
        var_samples = trace.posterior[var].values
        # Flatten to (chain * draw, -1)
        var_dim = int(np.prod(var_samples.shape[2:]))
        var_samples = var_samples.reshape(-1, var_dim)
        samples.append(var_samples)
    return torch.from_numpy(np.concatenate(samples, axis=1)).float()


@pytest.mark.slow
def test_hierarchical_pymc_c2st(hierarchical_nle_setup):
    """Test hierarchical model accuracy via c2st against true PyMC likelihood.

    This test:
    1. Runs MCMC with the TRUE Gaussian likelihood in PyMC
    2. Runs MCMC with the NLE via our bridge
    3. Compares posteriors via c2st

    Model structure:
    - mu ~ N(0, 1)           # hyperprior for population mean
    - tau ~ InvGamma(2, 1)   # hyperprior for population variance
    - theta[s] ~ N(mu, sqrt(tau))  # subject-level parameters
    - x[t,s] ~ N(theta[s], sigma_x)  # observations
    """
    num_trials = 5
    num_subjects = 3
    likelihood_nn = hierarchical_nle_setup["likelihood_nn"]
    sigma_x = hierarchical_nle_setup["sigma_x"]

    # Generate observed data from the true hierarchical model
    np.random.seed(42)  # For reproducibility
    true_mu = 0.5
    true_tau = 0.3
    true_theta = true_mu + np.sqrt(true_tau) * np.random.randn(num_subjects)

    # x[t, s] ~ N(theta[s], sigma_x^2)
    x_o = np.zeros((num_trials, num_subjects))
    for s in range(num_subjects):
        x_o[:, s] = true_theta[s] + sigma_x * np.random.randn(num_trials)

    draws = 1000
    tune = 500
    chains = 2

    with pm.Model():
        mu = pm.Normal("mu", mu=0, sigma=1)
        tau = pm.InverseGamma("tau", alpha=2, beta=1)
        theta = pm.Normal("theta", mu=mu, sigma=pm.math.sqrt(tau), shape=num_subjects)
        pm.Normal("x", mu=theta, sigma=sigma_x, observed=x_o)

        trace_true = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            progressbar=True,
            return_inferencedata=True,
        )

    x_o_with_event = x_o[..., np.newaxis]

    with pm.Model():
        mu = pm.Normal("mu", mu=0, sigma=1)
        tau = pm.InverseGamma("tau", alpha=2, beta=1)
        theta = pm.Normal(
            "theta",
            mu=mu,
            sigma=pm.math.sqrt(tau),
            shape=(num_subjects, 1),
        )
        neural_likelihood_to_pymc(
            likelihood_nn=likelihood_nn,
            theta=theta,
            observed=x_o_with_event,
            name="x",
            num_trials=num_trials,
            num_subjects=num_subjects,
        )

        trace_nle = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            progressbar=True,
            return_inferencedata=True,
        )

    true_samples = flatten_pymc_samples(trace_true, ["mu", "tau", "theta"])
    nle_mu = trace_nle.posterior["mu"].values.reshape(-1, 1)
    nle_tau = trace_nle.posterior["tau"].values.reshape(-1, 1)
    nle_theta = trace_nle.posterior["theta"].values.reshape(-1, num_subjects)
    nle_samples = torch.from_numpy(
        np.concatenate([nle_mu, nle_tau, nle_theta], axis=1)
    ).float()

    check_c2st(
        true_samples,
        nle_samples,
        alg="hierarchical-pymc-nle-vs-true",
        tol=0.1,
    )


def test_dims_keyword_support(linear_gaussian_setup, get_trained_nle):
    """Test that dims keyword is properly forwarded to PyMC CustomDist."""
    likelihood_nn = get_trained_nle._neural_net
    num_dim = linear_gaussian_setup["num_dim"]
    num_obs = 5
    true_theta = np.ones(num_dim) * 0.5
    x_o = true_theta + np.random.randn(num_obs, num_dim) * 0.1
    coords = {"observation": np.arange(num_obs), "param_dim": np.arange(num_dim)}

    with pm.Model(coords=coords) as model:
        theta = pm.Normal("theta", mu=0, sigma=1, shape=num_dim)
        neural_likelihood_to_pymc(
            likelihood_nn=likelihood_nn,
            theta=theta,
            observed=x_o,
            name="x",
            dims=("observation", "param_dim"),
        )
        assert "x" in model.named_vars
        assert hasattr(model, "named_vars_to_dims"), \
            "PyMC Model should have named_vars_to_dims attribute"
        assert model.named_vars_to_dims.get("x") == ("observation", "param_dim"), \
            "dims were not correctly forwarded to CustomDist"

    with model:
        trace = pm.sample(
            draws=10,
            tune=10,
            chains=1,
            cores=1,
            step=pm.Slice(),
            progressbar=False,
        )
        assert trace.posterior["theta"].shape == (1, 10, num_dim)
