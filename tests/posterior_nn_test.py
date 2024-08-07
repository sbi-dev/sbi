# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
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
)
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian


@pytest.mark.parametrize("snpe_method", [SNPE_A, SNPE_C])
@pytest.mark.parametrize(
    "x_o_batch_dim",
    (
        0,
        1,
        pytest.param(2, marks=pytest.mark.xfail(raises=AssertionError)),
    ),
)
def test_log_prob_with_different_x(snpe_method: type, x_o_batch_dim: bool):
    num_dim = 2
    num_simulations = 1000

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = snpe_method(prior=prior)
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
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
    num_simulations = 1000

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = snplre_method(prior=prior)
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
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
    num_simulations = 1000

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = snpe_method(prior=prior)
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
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
@pytest.mark.parametrize("snlre_method", [SNLE_A, SNRE_A, SNRE_B, SNRE_C, SNPE_C])
@pytest.mark.parametrize("x_o_batch_dim", (0, 1, 2))
@pytest.mark.parametrize("init_strategy", ["proposal", "resample"])
@pytest.mark.parametrize(
    "sample_shape",
    (
        (5,),  # less than num_chains
        (4, 2),  # 2D batch
        (15,),  # not divisible by num_chains
    ),
)
def test_batched_mcmc_sample_log_prob_with_different_x(
    snlre_method: type,
    x_o_batch_dim: bool,
    mcmc_params_fast: dict,
    init_strategy: str,
    sample_shape: torch.Size,
):
    num_dim = 2
    num_simulations = 100
    num_chains = 10

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = snlre_method(prior=prior)
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
    inference.append_simulations(theta, x).train(max_num_epochs=2)

    x_o = ones(num_dim) if x_o_batch_dim == 0 else ones(x_o_batch_dim, num_dim)

    posterior = inference.build_posterior(
        sample_with="mcmc",
        mcmc_method="slice_np_vectorized",
        mcmc_parameters=mcmc_params_fast,
    )

    samples = posterior.sample_batched(
        sample_shape,
        x_o,
        init_strategy=init_strategy,
        num_chains=num_chains,
    )

    assert (
        samples.shape == (*sample_shape, x_o_batch_dim, num_dim)
        if x_o_batch_dim > 0
        else (*sample_shape, num_dim)
    ), "Sample shape wrong"

    # test only for 1 sample_shape case to avoid repeating this test.
    if x_o_batch_dim > 1 and sample_shape == (5,):
        assert samples.shape[1] == x_o_batch_dim, "Batch dimension wrong"
        inference = snlre_method(prior=prior)
        _ = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(
            sample_with="mcmc",
            mcmc_method="slice_np_vectorized",
            mcmc_parameters=mcmc_params_fast,
        )

        x_o = torch.stack([0.5 * ones(num_dim), -0.5 * ones(num_dim)], dim=0)
        # test with multiple chains to test whether correct chains are
        # concatenated.
        sample_shape = (1000,)  # use enough samples for accuracy comparison
        samples = posterior.sample_batched(
            sample_shape, x_o, num_chains=num_chains, warmup_steps=500
        )

        samples_separate1 = posterior.sample(
            sample_shape, x_o[0], num_chains=num_chains, warmup_steps=500
        )
        samples_separate2 = posterior.sample(
            sample_shape, x_o[1], num_chains=num_chains, warmup_steps=500
        )

        # Check if means are approx. same
        samples_m = torch.mean(samples, dim=0, dtype=torch.float32)
        samples_separate1_m = torch.mean(samples_separate1, dim=0, dtype=torch.float32)
        samples_separate2_m = torch.mean(samples_separate2, dim=0, dtype=torch.float32)
        samples_sep_m = torch.stack([samples_separate1_m, samples_separate2_m], dim=0)

        assert torch.allclose(
            samples_m, samples_sep_m, atol=0.2, rtol=0.2
        ), "Batched sampling is not consistent with separate sampling."
