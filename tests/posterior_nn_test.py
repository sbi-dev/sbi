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
    SNRE_A,
    SNRE_B,
    SNRE_C,
    DirectPosterior,
    simulate_for_sbi,
)
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian
from sbi.utils.user_input_checks import process_prior, process_simulator


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
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
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


@pytest.mark.parametrize(
    "snplre_method", [SNPE_A, SNPE_C, SNLE_A, SNRE_A, SNRE_B, SNRE_C]
)
def test_importance_posterior_sample_log_prob(snplre_method: type):
    num_dim = 2

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = snplre_method(prior=prior)
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
    theta, x = simulate_for_sbi(simulator, prior, 1000)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=3)

    posterior = inference.build_posterior(sample_with="importance")

    x_o = ones(num_dim)
    samples = posterior.sample((10,), x=x_o)
    samples2, weights = posterior.sample((10,), x=x_o, method="importance")
    assert samples.shape == (10, num_dim), "Sample shape of sample is wrong"
    assert samples2.shape == (10, num_dim), "Sample of sample_with_weights shape wrong"
    assert weights.shape == (10,), "Weights shape wrong"

    log_prob = posterior.log_prob(samples, x=x_o)

    assert log_prob.shape == (10,), "logprob shape wrong"


@pytest.mark.parametrize("snpe_method", [SNPE_A, SNPE_C])
@pytest.mark.parametrize("x_o_batch_dim", (0, 1, 2))
def test_batched_sample_log_prob_with_different_x(
    snpe_method: type, x_o_batch_dim: bool
):
    num_dim = 2

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = snpe_method(prior=prior)
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
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
@pytest.mark.parametrize(
    "snlre_method",
    [
        pytest.param(SNLE_A, marks=pytest.mark.xfail(raises=NotImplementedError)),
        pytest.param(SNRE_A, marks=pytest.mark.xfail(raises=NotImplementedError)),
        pytest.param(SNRE_B, marks=pytest.mark.xfail(raises=NotImplementedError)),
        pytest.param(SNRE_C, marks=pytest.mark.xfail(raises=NotImplementedError)),
    ],
)
@pytest.mark.parametrize("x_o_batch_dim", (0, 1, 2))
def test_batched_mcmc_sample_log_prob_with_different_x(
    snlre_method: type, x_o_batch_dim: bool, mcmc_params_fast: dict
):
    num_dim = 2

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = snlre_method(prior=prior)
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
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
    ), "Sample shape wrong"
