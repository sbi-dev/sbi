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


@pytest.mark.parametrize("method", [pm.Slice, pm.NUTS, pm.HamiltonianMC])
def test_nle_pymc_bridge_minimal(method, get_trained_nle, linear_gaussian_setup):
    """Minimal test: NLE-PyMC bridge on 3D linear Gaussian model.

    Tests different MCMC methods:
    - Slice: gradient-free, good for testing basic functionality
    - NUTS: gradient-based, adaptive HMC
    - HamiltonianMC: gradient-based, standard HMC

    Args:
        mcmc_method: PyMC step method class to use for sampling
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

    print(f"True parameters: {true_theta.numpy()}")
    print(f"Observed data: {x_o.numpy()}")

    # === PyMC Integration ===
    prior_mean_np = prior_mean.numpy()
    prior_std_np = np.sqrt(np.diag(prior_cov.numpy()))
    num_posterior_samples = 1000

    # Adjust sampling parameters based on method
    if method == pm.HamiltonianMC:
        method_options = dict(
            target_accept=0.9,
            path_length=1.0,
            scaling=prior_cov.numpy(),
            is_cov=True,
        )
    else:
        method_options = {}
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
            step=method(**method_options),
            chains=num_chains,
            cores=1,
            progressbar=True,
            return_inferencedata=True,
        )

    # compare with c2st
    true_samples = true_posterior.sample((num_posterior_samples,))
    # Stack the PyMC samples correctly: shape should be (n_samples, n_params)
    # trace.posterior["theta"] has shape (chain, draw, dim)
    # We want (n_samples, dim) where n_samples = chain * draw
    inferred_samples = torch.from_numpy(
        trace.posterior["theta"].values.reshape(-1, num_dim)
    ).float()

    print(
        f"Posterior mean (PyMC {method.__name__}): "
        f"{inferred_samples.mean(dim=0).numpy()}"
    )
    print(f"True posterior mean: {true_samples.mean(dim=0).numpy()}")

    # Check convergence for gradient-based methods
    if method in [pm.NUTS, pm.HamiltonianMC]:
        r_hat = pm.rhat(trace)["theta"].values
        print(f"R-hat values: {r_hat}")
        assert np.all(r_hat < 1.1), f"R-hat values too high: {r_hat}"

    check_c2st(
        true_samples[:num_posterior_samples],
        inferred_samples[:num_posterior_samples],
        alg=f"nle-pymc-{method.__name__}",
        tol=0.1,
    )

    print(f"✓ Test passed: NLE-PyMC bridge with {method.__name__} is working!")


# === Scalar theta tests ===


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

    print(f"True theta: {true_theta.item()}")
    print(f"Observed data shape: {x_o.shape}")

    # PyMC model with scalar theta (no shape argument)
    prior_mean_np = float(prior_mean.numpy())
    prior_std_np = float(np.sqrt(prior_cov.numpy()[0, 0]))

    with pm.Model():
        # Scalar prior - this is what we're testing
        theta_pymc = pm.Normal("theta", mu=prior_mean_np, sigma=prior_std_np)

        # Neural likelihood - scalar theta (shape ()) is auto-detected
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

    # Compare posterior
    num_posterior_samples = 500
    true_samples = true_posterior.sample((num_posterior_samples,))

    # trace.posterior["theta"] has shape (chain, draw) for scalar
    inferred_samples = torch.from_numpy(
        trace.posterior["theta"].values.reshape(-1, 1)
    ).float()

    posterior_mean = inferred_samples.mean().item()
    true_mean = true_samples.mean().item()

    print(f"Posterior mean (PyMC {method.__name__}): {posterior_mean:.3f}")
    print(f"True posterior mean: {true_mean:.3f}")

    # Check convergence
    if method == pm.NUTS:
        r_hat = pm.rhat(trace)["theta"].values
        print(f"R-hat: {r_hat}")
        assert r_hat < 1.1, f"R-hat too high: {r_hat}"

    # Check posterior accuracy
    check_c2st(
        true_samples[:num_posterior_samples],
        inferred_samples[:num_posterior_samples],
        alg=f"scalar-nle-pymc-{method.__name__}",
        tol=0.1,
    )

    print(f"✓ Test passed: Scalar theta with {method.__name__} is working!")
