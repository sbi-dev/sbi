# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from torch import Tensor, eye, zeros
from torch.distributions import MultivariateNormal, Normal

from sbi.inference import (
    NLE_A,
    NRE,
    MCMCPosterior,
    likelihood_estimator_based_potential,
)
from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters
from sbi.inference.potentials.ratio_based_potential import RatioBasedPotential
from sbi.samplers.mcmc import SliceSamplerSerial, SliceSamplerVectorized
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
    mcmc_params_fast: MCMCPosteriorParameters,
    num_dim: int = 2,
    num_samples: int = 42,
    num_trials: int = 2,
    num_simulations: int = 10,
):
    """Runs NLE and checks that posterior_sampler is correctly set."""
    x_o = zeros((num_trials, num_dim))
    mcmc_params_fast = mcmc_params_fast.with_param(
        num_chains=num_chains, method=sampling_method
    )

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = NLE_A(prior, show_progress_bars=False)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)
    estimator = inference.append_simulations(theta, x).train(max_num_epochs=5)
    potential_fn, transform = likelihood_estimator_based_potential(
        estimator, prior, x_o
    )
    posterior = MCMCPosterior(
        potential_fn,
        theta_transform=transform,
        proposal=prior,
        **asdict(mcmc_params_fast),
    )

    assert posterior.posterior_sampler is None
    samples = posterior.sample(
        sample_shape=(num_samples, num_chains),
        x=x_o,
    )
    assert isinstance(samples, Tensor)
    assert samples.shape == (num_samples, num_chains, num_dim)

    if "pyro" in sampling_method:
        from pyro.infer.mcmc import MCMC

        assert type(posterior.posterior_sampler) is MCMC
    elif "pymc" in sampling_method:
        from sbi.samplers.mcmc.pymc_wrapper import PyMCSampler

        assert type(posterior.posterior_sampler) is PyMCSampler
    elif sampling_method == "slice_np":
        assert type(posterior.posterior_sampler) is SliceSamplerSerial
    else:  # sampling_method == "slice_np_vectorized"
        assert type(posterior.posterior_sampler) is SliceSamplerVectorized


def test_direct_posterior_shape_validation():
    """Test malformed theta shapes raise a clean ValueError (issue #2136)."""
    import torch

    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=torch.tensor([-2.0]), high=torch.tensor([2.0]))

    def simulator(theta):
        return theta + 0.1 * torch.randn_like(theta)

    theta = prior.sample((20,))
    x = simulator(theta)

    inferer = NPE(prior=prior)
    inferer.append_simulations(theta, x)
    density_estimator = inferer.train(max_num_epochs=1, show_train_summary=False)
    posterior = inferer.build_posterior(density_estimator)

    # Passing a 1D tensor with 2 elements instead of the expected 1.
    bad_theta = torch.tensor([1.0, 2.0])

    with pytest.raises(ValueError, match="Expected theta to have event shape"):
        posterior.log_prob(bad_theta, x=x[0])

    with pytest.raises(ValueError, match="Expected theta to have event shape"):
        posterior.potential(bad_theta, x=x[0])


def test_nle_non_iid_single_x_multiple_theta():
    """Test NLE supports one x with multiple theta when x_is_iid=False."""
    prior = MultivariateNormal(loc=zeros(2), covariance_matrix=eye(2))

    theta = prior.sample((20,))
    x = diagonal_linear_gaussian(theta)

    inference = NLE_A(prior, show_progress_bars=False)
    estimator = inference.append_simulations(theta, x).train(max_num_epochs=5)

    potential_fn, _ = likelihood_estimator_based_potential(estimator, prior, x[:1])
    potential_fn = potential_fn.bind(x[:1], x_is_iid=False)

    theta_batch = prior.sample((4,))
    log_prob = potential_fn(theta_batch)

    assert log_prob.shape == (4,)
    assert torch.isfinite(log_prob).all()


def test_nle_non_iid_multiple_x_multiple_theta():
    """Test NLE supports matching batches of x and theta when x_is_iid=False."""
    prior = MultivariateNormal(loc=zeros(2), covariance_matrix=eye(2))

    theta = prior.sample((20,))
    x = diagonal_linear_gaussian(theta)

    inference = NLE_A(prior, show_progress_bars=False)
    estimator = inference.append_simulations(theta, x).train(max_num_epochs=5)

    potential_fn, _ = likelihood_estimator_based_potential(estimator, prior, x[:4])
    potential_fn = potential_fn.bind(x[:4], x_is_iid=False)

    theta_batch = prior.sample((4,))
    log_prob = potential_fn(theta_batch)

    assert log_prob.shape == (4,)
    assert torch.isfinite(log_prob).all()


def test_nle_non_iid_normal_prior_shape():
    prior = Normal(torch.zeros(1), torch.ones(1))
    theta = prior.sample((100,))
    x = theta + torch.randn_like(theta)

    inference = NLE_A(prior=prior)
    inference.append_simulations(theta, x).train(max_num_epochs=5)

    posterior = inference.build_posterior()

    potential_fn = posterior.potential_fn.bind(
        x_o=x[:1],
        x_is_iid=False,
    )

    theta_batch = prior.sample((4,))

    potential_values = potential_fn(
        theta_batch,
        track_gradients=False,
    )

    assert potential_values.shape == (4,)


def test_nre_non_iid_normal_prior_shape():
    prior = Normal(torch.zeros(1), torch.ones(1))
    theta = prior.sample((100,))
    x = theta + torch.randn_like(theta)

    inference = NRE(prior=prior)
    inference.append_simulations(theta, x).train(max_num_epochs=5)

    posterior = inference.build_posterior()

    potential_fn = posterior.potential_fn.bind(
        x_o=x[:1],
        x_is_iid=False,
    )

    theta_batch = prior.sample((4,))

    potential_values = potential_fn(
        theta_batch,
        track_gradients=False,
    )

    assert potential_values.shape == (4,)


def test_nre_non_iid_single_x_multiple_theta():
    """Test NRE supports one x with multiple theta when x_is_iid=False."""
    prior = MultivariateNormal(loc=zeros(2), covariance_matrix=eye(2))

    theta = prior.sample((20,))
    x = diagonal_linear_gaussian(theta)

    inference = NRE(prior=prior, show_progress_bars=False)
    estimator = inference.append_simulations(theta, x).train(max_num_epochs=5)

    potential_fn = RatioBasedPotential(estimator, prior, x[:1])
    potential_fn = potential_fn.bind(x[:1], x_is_iid=False)

    theta_batch = prior.sample((4,))
    log_prob = potential_fn(theta_batch)

    assert log_prob.shape == (4,)
    assert torch.isfinite(log_prob).all()


def test_nre_non_iid_multiple_x_multiple_theta():
    """Test NRE supports matching batches of x and theta when x_is_iid=False."""
    prior = MultivariateNormal(loc=zeros(2), covariance_matrix=eye(2))

    theta = prior.sample((20,))
    x = diagonal_linear_gaussian(theta)

    inference = NRE(prior=prior, show_progress_bars=False)
    estimator = inference.append_simulations(theta, x).train(max_num_epochs=5)

    potential_fn = RatioBasedPotential(estimator, prior, x[:4])
    potential_fn = potential_fn.bind(x[:4], x_is_iid=False)

    theta_batch = prior.sample((4,))
    log_prob = potential_fn(theta_batch)

    assert log_prob.shape == (4,)
    assert torch.isfinite(log_prob).all()
