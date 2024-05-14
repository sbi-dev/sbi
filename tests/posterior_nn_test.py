# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import (
    SNLE_A,
    SNPE_A,
    SNPE_C,
    DirectPosterior,
    simulate_for_sbi,
)
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian


@pytest.mark.parametrize("snpe_method", [SNPE_A, SNPE_C])
@pytest.mark.parametrize(
    "x_o_batch_dim",
    (
        0,
        1,
        pytest.param(2, marks=pytest.mark.xfail(raises=ValueError)),
    ),
)
def test_log_prob_with_different_x(snpe_method: type, x_o_batch_dim: bool):
    num_dim = 2

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = snpe_method(prior=prior)
    theta, x = simulate_for_sbi(simulator, prior, 1000)
    posterior_estimator = inference.append_simulations(theta, x).train(max_num_epochs=3)

    if x_o_batch_dim == 0:
        x_o = ones(num_dim)
    elif x_o_batch_dim == 1:
        x_o = ones(1, num_dim)
    elif x_o_batch_dim == 2:
        x_o = ones(2, num_dim)
    else:
        raise NotImplementedError

    posterior = DirectPosterior(
        posterior_estimator=posterior_estimator, prior=prior
    ).set_default_x(x_o)
    samples = posterior.sample((10,))
    _ = posterior.log_prob(samples)


@pytest.mark.parametrize("snpe_method", [SNPE_A, SNPE_C])
@pytest.mark.parametrize(
    "x_o_batch_dim",
    (
        0,
        1,
        2,
    ),
)
def test_batched_sample_log_prob_with_different_x(
    snpe_method: type, x_o_batch_dim: bool
):
    num_dim = 2

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = snpe_method(prior=prior)
    theta, x = simulate_for_sbi(simulator, prior, 1000)
    posterior_estimator = inference.append_simulations(theta, x).train(max_num_epochs=3)

    x_o = ones(num_dim) if x_o_batch_dim == 0 else ones(x_o_batch_dim, num_dim)

    posterior = DirectPosterior(posterior_estimator=posterior_estimator, prior=prior)

    samples = posterior.sample_batched((10,), x_o)
    batched_log_probs = posterior.log_prob_batched(samples, x_o)

    assert (
        samples.shape == (10, x_o_batch_dim, num_dim)
        if x_o_batch_dim > 0
        else (10, num_dim)
    ), "Sample shape wrong"
    assert batched_log_probs.shape == (10, max(x_o_batch_dim, 1)), "logprob shape wrong"


@pytest.mark.mcmc
@pytest.mark.parametrize("snlre_method", [SNLE_A])
@pytest.mark.parametrize(
    "x_o_batch_dim",
    (
        0,
        1,
        2,
    ),
)
def test_batched_mcmc_sample_log_prob_with_different_x(
    snlre_method: type, x_o_batch_dim: bool, mcmc_params_fast: dict
):
    num_dim = 2

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = snlre_method(prior=prior)
    theta, x = simulate_for_sbi(simulator, prior, 1000)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=3)

    x_o = ones(num_dim) if x_o_batch_dim == 0 else ones(x_o_batch_dim, num_dim)

    posterior = inference.build_posterior(
        mcmc_method="slice_np_vectorized", mcmc_parameters=mcmc_params_fast
    )

    samples = posterior.sample_batched((10,), x_o)

    assert (
        samples.shape == (10, x_o_batch_dim, num_dim)
        if x_o_batch_dim > 0
        else (10, num_dim)
    ), "Sampel shape wrong"
