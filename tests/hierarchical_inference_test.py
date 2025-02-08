import jax.numpy as jnp
import numpy as np
import pyro
import pyro.distributions as dist
import pytest
import torch
from jax import random
from pyro.infer import MCMC, NUTS
from torch.distributions import HalfNormal, Normal

from sbi.inference import NLE
from sbi.simulators.linear_gaussian import true_posterior_linear_gaussian_mvn_prior
from sbi.utils.hierarchical_inference_utils import (
    HierarchicalPrior,
    TruncatedNormal,
)
from tests.test_utils import check_c2st

# Ground truth data and posterior
num_subjects, num_dim = 10, 3  # Subjects, trials, dimensions
mu_0 = torch.zeros(num_dim)
sigma_0 = torch.ones(num_dim) * 1.0
alpha_0 = torch.ones(num_dim) * 2.0
beta_0 = torch.ones(num_dim)
sigma_x = 0.5
num_samples = 1000
warmup_steps = 1000


# fixture for observed data
@pytest.fixture(scope="module")
def x_o_and_params():
    theta, mu, sigma2, x_o = hierarchical_gaussian_model()
    return x_o, theta, mu, sigma2


# fixture for reference posterior, using fixture above.
@pytest.fixture(scope="module")
def reference_posterior(x_o_and_params):
    x_o, *_ = x_o_and_params

    # Run HMC inference
    nuts_kernel = NUTS(hierarchical_gaussian_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=warmup_steps)
    mcmc.run(x_o=x_o)

    return mcmc


@pytest.mark.parametrize("sample_shape", [(1000,), (1000, 10)])
@pytest.mark.parametrize("num_variables", [1, 2])
def test_truncated_normal(sample_shape, num_variables):
    loc = torch.tensor([0.0] * num_variables)
    scale = torch.tensor([1.0] * num_variables)
    low = torch.tensor([-1.0] * num_variables)
    high = torch.tensor([1.0] * num_variables)
    dist = TruncatedNormal(loc, scale, low, high)

    samples = dist.sample(sample_shape)
    assert samples.shape == (sample_shape + (num_variables,)), "Sample shape incorrect."
    # test low and high
    assert (samples >= low).all(), "Sample below lower bound."
    assert (samples <= high).all(), "Sample above upper bound."

    log_prob = dist.log_prob(samples)
    # log_prob outputs batch shape
    assert log_prob.shape == sample_shape + (num_variables,), (
        "Log probability shape incorrect."
    )


@pytest.mark.parametrize("sample_shape", [(10,), (20, 10)])
def test_hierarchical_prior(sample_shape):
    num_subjects = 2
    lower_bounds = torch.tensor([0.1, 0.1, 0.1])
    upper_bounds = torch.tensor([3.0, 2.5, 1.0])
    num_parameters = lower_bounds.shape[0]
    mu_prior = Normal(torch.zeros(num_parameters), torch.ones(num_parameters) * 2)
    sigma_prior = HalfNormal(torch.ones(num_parameters) * 0.5)

    hierarchical_prior = HierarchicalPrior(
        mu_prior, sigma_prior, lower_bounds, upper_bounds, num_subjects=num_subjects
    )

    samples = hierarchical_prior.sample(sample_shape)
    assert samples.shape == sample_shape + (num_subjects + 2, num_parameters), (
        "Sample shape incorrect."
    )

    log_prob = hierarchical_prior.log_prob(samples)
    assert log_prob.shape == sample_shape, "Log probability shape incorrect."


def test_hierarchical_inference(x_o_and_params, reference_posterior):
    # train density estimator on single-trial data
    # Define prior
    num_simulations = 10000
    mus = torch.randn(num_simulations, num_dim) * torch.from_numpy(
        sigma_0
    ) + torch.from_numpy(mu_0)
    sigmas = torch.distributions.InverseGamma(
        torch.from_numpy(alpha_0), torch.from_numpy(beta_0)
    ).sample((num_simulations,))
    thetas = torch.randn(num_simulations, num_dim) * sigmas + mus
    x = torch.randn(num_simulations, num_dim) * sigma_x + thetas

    # Train NLE
    trainer = NLE().append_simulations(theta=thetas.float(), x=x.float())
    density_estimator = trainer.train(max_num_epochs=1000)

    # Wrap for pyro
    from sbi.utils.hierarchical_inference_utils import PyroConditionalDensityEstimator

    pyro_density_estimator = PyroConditionalDensityEstimator(density_estimator)

    def sbi_pyro_model(likelihood, x_o=None):
        mu = pyro.sample("mu", dist.Normal(mu_0, sigma_0))  # Shape: (D,)
        sigma = pyro.sample("sigma", dist.InverseGamma(alpha_0, beta_0))  # Shape: (D,)

        # Sample subject-level parameters (batch shape: M, event shape: D)
        with pyro.plate("subjects", num_subjects, dim=-2):
            theta = pyro.sample("theta", dist.Normal(mu, sigma).to_event(1))

            # Step 4: Sample observations x_mn given θ_m using the density estimator
            with pyro.plate("trials", num_trials, dim=-1):
                if x_o is not None:
                    pyro.sample("x_o", dist.Delta(x_o).to_event(1), obs=x_o)
                else:
                    x = likelihood.sample(
                        (num_trials,), numpy_to_torch(theta.squeeze())
                    ).transpose(1, 0, 2)
        if x_o is None:
            return theta, mu, sigma, x
        else:
            return x_o

    # test
    with pyro.handlers.seed(rng_seed=0):
        theta, mu, sigma, x = sbi_pyro_model(pyro_density_estimator)
        sbi_pyro_model(pyro_density_estimator, x_o=x)

    # Run inference
    x_o, *_ = x_o_and_params
    nuts_kernel = NUTS(sbi_pyro_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=warmup_steps)
    mcmc.run(x_o=x_o, rng_key=random.PRNGKey(2), likelihood=pyro_density_estimator)

    # Compare posterior samples
    posterior_samples = reference_posterior.get_samples()
    sbi_posterior_samples = mcmc.get_samples()

    import pdb

    pdb.set_trace()


def test_reference_gaussian_model():
    num_trials = 100
    num_samples = 1000
    theta_o, x_o = gaussian_model(num_trials=num_trials)

    gt_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o=x_o,
        likelihood_shift=torch.zeros(num_dim),
        likelihood_cov=sigma_x * torch.eye(num_dim),
        prior_mean=mu_0,
        prior_cov=torch.diag(sigma_0),
    )

    posterior_samples = gt_posterior.sample((num_samples,))

    nuts_kernel = NUTS(gaussian_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=1000)
    mcmc.run(x_o=x_o)

    pyro_samples = mcmc.get_samples()["theta"]

    # import matplotlib.pyplot as plt
    # from sbi.analysis import pairplot

    # pairplot([posterior_samples, pyro_samples, sbi_samples], samples_labels=["GT", "Pyro", "SBI"], legend=True)
    # plt.show()

    check_c2st(
        posterior_samples, pyro_samples, tol=0.1, alg="pyro MCMC vs GT posterior"
    )


def test_sbi_pyro_on_gaussian():
    num_trials = 1
    num_samples = 1000
    theta_o, x_o = gaussian_model(num_trials=num_trials)

    nuts_kernel = NUTS(gaussian_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=1000)
    mcmc.run(x_o=x_o)

    pyro_samples = torch.from_numpy(np.array(mcmc.get_samples()["theta"]))

    # Run sbi
    num_simulations = 5000
    prior = torch.distributions.MultivariateNormal(
        loc=mu_0, covariance_matrix=torch.diag(sigma_0)
    )
    theta = []
    x = []
    for i in range(num_simulations):
        ti, xi = gaussian_model(num_trials=1)
        theta.append(torch.from_numpy(np.array(ti)))
        x.append(torch.from_numpy(np.array(xi)))
    theta = torch.stack(theta, dim=0).float()
    x = torch.stack(x, dim=0).float().squeeze()

    trainer = NLE(prior=prior).append_simulations(theta=theta, x=x)
    density_estimator = trainer.train()

    # Wrap for pyro
    from torch.distributions import constraints

    class SBILikelihoodPyro(dist.TorchDistribution):
        """
        A Pyro-compatible wrapper for an `sbi` likelihood model.
        """

        support = constraints.real  # Assuming continuous values
        has_rsample = False  # Since `sbi` uses `.sample()`, not `.rsample()`

        def __init__(self, likelihood, theta):
            """
            Wrap an `sbi` likelihood for use in Pyro.

            Args:
                likelihood: An `sbi` likelihood model with a `.sample()` method.
                theta: The conditioning parameter for the likelihood.
            """
            self.likelihood = likelihood
            self.theta = theta
            batch_shape = theta.shape[:-1]  # Inherit batch shape from `theta`
            event_shape = theta.shape[-1:]  # Dimensionality of `x`
            super().__init__(batch_shape=batch_shape, event_shape=event_shape)

        def sample(self, sample_shape=torch.Size()):
            """Generate samples from the SBI likelihood."""
            return self.likelihood.sample(sample_shape, condition=self.theta)

        def log_prob(self, x, theta):
            """Compute log probability of x given theta."""
            return self.likelihood.log_prob(x, condition=theta)

    def sbi_pyro_model(likelihood, x_o=None):
        theta = pyro.sample(
            "theta", dist.MultivariateNormal(mu_0, torch.diag(sigma_0))
        )  # (D,)

        with pyro.plate("trials", num_trials):
            if x_o is not None:
                pyro.sample("x", dist.Delta(x_o).to_event(1), obs=x_o)
                return x_o
            else:
                x = pyro.sample(
                    "x", SBILikelihoodPyro(likelihood, theta.expand(num_trials, -1))
                )
                return theta, x

    # test
    theta, x = sbi_pyro_model(density_estimator)
    sbi_pyro_model(density_estimator, x_o=x)

    # Run inference
    nuts_kernel = NUTS(sbi_pyro_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=warmup_steps)
    mcmc.run(x_o=x_o, likelihood=density_estimator)

    # Compare posterior samples
    sbi_samples = mcmc.get_samples()["theta"]

    import matplotlib.pyplot as plt

    from sbi.analysis import pairplot

    pairplot([pyro_samples, sbi_samples], samples_labels=["Pyro", "SBI"], legend=True)
    plt.show()

    check_c2st(sbi_samples, pyro_samples, tol=0.1, alg="pyro MCMC vs SBI-pyro MCMC")


def test_reference_hierarchical_posterior(x_o_and_params, reference_posterior):
    # Generate synthetic data
    x_o, theta, mu, sigma = x_o_and_params
    posterior_samples = reference_posterior.get_samples()

    # Convert JAX arrays to PyTorch tensors
    theta_m_samples = posterior_samples["theta"]  # Shape: (num_samples, M, D)
    mu_samples = posterior_samples["mu"]  # Shape: (num_samples, D)
    sigma2_samples = posterior_samples["sigma"]  # Shape: (num_samples, D)

    # Check if the posterior samples are close to the ground truth
    assert jnp.allclose(theta_m_samples.mean(0), theta, atol=theta.std(0).max()), (
        "Subject-level parameter mean incorrect."
    )
    assert jnp.allclose(mu_samples.mean(0), mu, atol=mu_samples.std(0).max()), (
        "Group mean incorrect."
    )
    assert jnp.allclose(
        sigma2_samples.mean(0), sigma, atol=sigma2_samples.std(0).max()
    ), "Group variance incorrect."


def hierarchical_gaussian_model(x_o=None):
    """
    Pyro model for hierarchical Bayesian inference with Gaussian likelihoods.

    Args:
        x_o (torch.Tensor, optional): Observed data (M, N, D), if available.

    Returns:
        Pyro model for inference.
    """
    mu = numpyro.sample("mu", dist.Normal(mu_0, sigma_0))  # Shape: (D,)
    sigma = numpyro.sample("sigma", dist.InverseGamma(alpha_0, beta_0))  # Shape: (D,)

    # Sample subject-level parameters (batch shape: M, event shape: D)
    with numpyro.plate("subjects", num_subjects, dim=-2):
        theta = numpyro.sample("theta", dist.Normal(mu, sigma).to_event(1))

        # Sample observations (batch shape: M, N, D)
        with numpyro.plate("trials", num_trials, dim=-1):
            x = numpyro.sample("x", dist.Normal(theta, sigma_x).to_event(1), obs=x_o)

    if x_o is None:
        return theta, mu, sigma, x


def gaussian_model(x_o=None, num_trials=1, sigma_x=0.5):
    """
    Pyro model for Bayesian inference with Gaussian likelihoods.

    Args:
        x_o (torch.Tensor, optional): Observed data (N, D), if available.

    Returns:
        Pyro model for inference.
    """
    theta = pyro.sample(
        "theta", dist.MultivariateNormal(mu_0, torch.diag(sigma_0))
    )  # Shape: (D,)

    # Sample observations (batch shape: N, event shape: D)
    with pyro.plate("trials", num_trials):
        x = pyro.sample(
            "x", dist.MultivariateNormal(theta, sigma_x * torch.eye(num_dim)), obs=x_o
        )

    if x_o is None:
        return theta, x
    else:
        return x_o
