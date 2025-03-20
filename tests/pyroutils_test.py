# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer import MCMC, NUTS
from pyro.infer.predictive import Predictive

from sbi.inference import NLE, NPE, NRE
from sbi.utils.pyroutils import (
    ConditionalDensityEstimatorDistribution,
    RatioEstimatorDistribution,
    get_transforms,
)
from tests.test_utils import check_c2st


def single_level_prior_theta(num_dim=10):
    mu_0 = torch.zeros(num_dim)
    sigma_0 = torch.ones(num_dim) * 1.0
    return dist.MultivariateNormal(mu_0, torch.diag(sigma_0))


def pyro_gaussian_model(x_o=None, num_dim=None, num_trials=1, sigma_x=0.5):
    """
    Pyro model for Bayesian inference with Gaussian likelihoods.

    Args:
        x_o (torch.Tensor, optional): Observed data (N, D), if available.

    Returns:
        Pyro model for inference.
    """
    if num_dim is None:
        if x_o is not None:
            num_dim = x_o.shape[-1]
        else:
            raise ValueError("num_dim must be provided if x_o is not provided")

    theta = pyro.sample("theta", single_level_prior_theta(num_dim))  # Shape: (D,)

    # Sample observations (batch shape: N, event shape: D)
    with pyro.plate("trials", num_trials):
        x = pyro.sample(
            "x",
            dist.MultivariateNormal(theta, sigma_x**2 * torch.eye(num_dim)),
            obs=x_o,
        )

    if x_o is None:
        return theta, x
    else:
        return x_o


@pytest.fixture
def data_and_pyro_gaussian_mcmc_samples(num_dim, num_trials, num_samples, warmup_steps):
    """Generate observed data and return MCMC samples from the true model."""
    # Get data we will condition on
    _, x_o = pyro_gaussian_model(num_trials=num_trials, num_dim=num_dim)

    # Get MCMC samples from a model using the true likelihood
    nuts_kernel = NUTS(pyro_gaussian_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run(x_o=x_o)
    pyro_samples = mcmc.get_samples()["theta"]

    return x_o, pyro_samples


@pytest.fixture
def num_simulations(trainer_cls, num_dim, num_trials):
    num_simulations = num_dim * num_trials * 200
    if trainer_cls is NRE:
        num_simulations *= 5
    return num_simulations


@pytest.fixture
def gaussian_simulation_data(num_simulations, num_dim):
    samples = Predictive(
        pyro_gaussian_model, num_samples=num_simulations, parallel=True
    )(num_dim=num_dim, num_trials=1)
    theta = samples['theta'].squeeze(1)
    x = samples['x'].squeeze(1)
    return theta, x


def test_unbounded_transform():
    prior_dim = 10
    prior_params = {
        "low": -1.0 * torch.ones((prior_dim,)),
        "high": +1.0 * torch.ones((prior_dim,)),
    }
    prior_dist = pyro.distributions.Uniform(**prior_params).to_event(1)

    def prior(num_samples=1):
        return pyro.sample("theta", prior_dist.expand_by([num_samples]))

    transforms = get_transforms(prior)

    to_unbounded = transforms["theta"]
    to_bounded = transforms["theta"].inv

    assert to_unbounded(prior(1000)).max() > 1.0
    assert to_bounded(to_unbounded(prior(1000))).max() < 1.0


@pytest.mark.parametrize(
    "trainer_cls, distribution_cls",
    [
        (NLE, ConditionalDensityEstimatorDistribution),
        (NPE, ConditionalDensityEstimatorDistribution),
        (NRE, RatioEstimatorDistribution),
    ],
)
def test_estimator_distribution_basic_properties(
    trainer_cls,
    distribution_cls,
    num_simulations: int = 100,
    dim: int = 5,
):
    """Test basic properties of the estimator distribution."""
    prior = torch.distributions.MultivariateNormal(
        loc=torch.zeros(dim), covariance_matrix=torch.diag(torch.ones(dim))
    )
    theta = prior.sample(torch.Size([num_simulations]))
    x = torch.distributions.Normal(theta, 1.0).sample()
    trainer = trainer_cls(prior=prior).append_simulations(theta=theta, x=x)
    density_estimator = trainer.train()

    # Test checking of condition shape
    with pytest.raises(ValueError):
        distribution_cls(estimator=density_estimator, condition=theta[:, :1])

    # Test basic properties
    estimator_dist = distribution_cls(estimator=density_estimator, condition=theta)
    assert isinstance(estimator_dist, distribution_cls)
    assert estimator_dist.estimator == density_estimator
    assert estimator_dist.get_condition_and_event_shapes() == (
        torch.Size([dim]),
        torch.Size([dim]),
    )
    assert estimator_dist.batch_shape == torch.Size([num_simulations])
    assert estimator_dist.event_shape == torch.Size([dim])
    assert estimator_dist.support == torch.distributions.constraints.real

    # Test sample method
    if trainer_cls is not NRE:
        assert estimator_dist.sample().shape == torch.Size([num_simulations, dim])
        x_samples = estimator_dist.sample(torch.Size([3]))
        assert x_samples.shape == torch.Size([3, num_simulations, dim])
    else:
        with pytest.raises(NotImplementedError):
            estimator_dist.sample()
        with pytest.raises(NotImplementedError):
            estimator_dist.sample(torch.Size([3]))
        x_samples = torch.distributions.Normal(theta, 1.0).sample(torch.Size([3]))

    # Test log_prob method
    assert estimator_dist.log_prob(x).shape == torch.Size([num_simulations])
    assert estimator_dist.log_prob(x_samples).shape == torch.Size([3, num_simulations])

    # Test expand method
    estimator_dist_expanded = estimator_dist.expand(torch.Size([2, num_simulations]))
    assert isinstance(estimator_dist_expanded, distribution_cls)
    assert estimator_dist_expanded.batch_shape == torch.Size([2, num_simulations])
    assert estimator_dist_expanded.event_shape == estimator_dist.event_shape
    assert torch.equal(
        estimator_dist_expanded.condition,
        theta.expand(torch.Size([2, num_simulations, dim])),
    )
    assert estimator_dist_expanded.estimator == estimator_dist.estimator


@pytest.mark.parametrize("num_samples", [500])
@pytest.mark.parametrize("warmup_steps", [500])
@pytest.mark.parametrize("num_dim", [3, 5])
@pytest.mark.parametrize("num_trials", [1, 5])
@pytest.mark.parametrize(
    "trainer_cls, distribution_cls",
    [
        (NLE, ConditionalDensityEstimatorDistribution),
        (NRE, RatioEstimatorDistribution),
    ],
)
def test_pyro_gaussian_model(
    data_and_pyro_gaussian_mcmc_samples,
    trainer_cls,
    distribution_cls,
    num_dim,
    num_trials,
    gaussian_simulation_data,
    num_samples,
    warmup_steps,
):
    """Test consistency of MCMC samples between the true and estimated likelihood."""
    x_o, pyro_samples = data_and_pyro_gaussian_mcmc_samples

    theta, x = gaussian_simulation_data
    prior = single_level_prior_theta(num_dim=num_dim)
    trainer = trainer_cls(prior=prior).append_simulations(theta=theta, x=x)
    density_estimator = trainer.train()

    # Define a model that uses the estimated likelihood
    def sbi_pyro_model(likelihood, x_o=None):
        theta = pyro.sample("theta", single_level_prior_theta(num_dim))  # (D,)

        with pyro.plate("trials", num_trials):
            x = pyro.sample("x", distribution_cls(likelihood, theta), obs=x_o)

        if x_o is None:
            return theta, x
        else:
            return x_o

    if trainer_cls is not NRE:
        _, _ = sbi_pyro_model(density_estimator)
    sbi_pyro_model(density_estimator, x_o=x_o)

    # Get MCMC samples from a model that uses the estimated likelihood
    nuts_kernel = NUTS(sbi_pyro_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run(x_o=x_o, likelihood=density_estimator)
    sbi_samples = mcmc.get_samples()["theta"]

    # Compare posterior samples
    check_c2st(
        sbi_samples,
        pyro_samples,
        tol=0.1,
        alg=f"pyro MCMC vs SBI-pyro-{trainer_cls.__name__} MCMC",
    )
