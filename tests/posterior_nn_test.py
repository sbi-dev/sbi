# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import (
    NLE_A,
    NPE_A,
    NPE_C,
    NRE_A,
    NRE_B,
    NRE_C,
    DirectPosterior,
)
from sbi.simulators.linear_gaussian import (
    diagonal_linear_gaussian,
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.diagnostics_utils import get_posterior_samples_on_batch
from tests.test_utils import check_c2st


@pytest.mark.parametrize("snpe_method", [NPE_A, NPE_C])
@pytest.mark.parametrize(
    "x_o_batch_dim",
    (
        0,
        1,
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                raises=ValueError,
                reason=".log_prob() supports only batch size 1 for x_o.",
            ),
        ),
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


@pytest.mark.parametrize("snplre_method", [NPE_A, NPE_C, NLE_A, NRE_A, NRE_B, NRE_C])
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


@pytest.mark.parametrize("snpe_method", [NPE_A, NPE_C])
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
@pytest.mark.parametrize("snlre_method", [NLE_A, NRE_A, NRE_B, NRE_C, NPE_C])
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


@pytest.mark.slow
@pytest.mark.parametrize("density_estimator", ["mdn", "maf", "zuko_nsf"])
def test_batched_sampling_and_logprob_accuracy(density_estimator: str):
    """Test with two different observations and compare to sequential methods."""
    num_dim = 2
    num_simulations = 2000
    num_samples = 1000
    sample_shape = (num_samples,)
    xos = torch.stack((-1.0 * ones(num_dim), 1.0 * ones(num_dim)))
    num_xos = xos.shape[0]

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    inference = NPE_C(
        prior=prior, show_progress_bars=False, density_estimator=density_estimator
    )
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
    posterior_estimator = inference.append_simulations(theta, x).train()

    posterior = DirectPosterior(posterior_estimator=posterior_estimator, prior=prior)

    samples_batched = get_posterior_samples_on_batch(
        xos, posterior, sample_shape, use_batched_sampling=True
    )
    log_probs_batched = posterior.log_prob_batched(samples_batched, xos)

    # check c2st for each xos
    for idx in range(0, num_xos):
        gt_posterior = true_posterior_linear_gaussian_mvn_prior(
            xos[idx], likelihood_shift, likelihood_cov, prior_mean, prior_cov
        )
        target_samples = gt_posterior.sample((num_samples,))
        check_c2st(
            target_samples,
            samples_batched[:, idx],
            alg=f"c2st-batch-vs-non-batch-{density_estimator}-x-idx{idx}",
        )

        target_log_probs = gt_posterior.log_prob(samples_batched[:, idx])
        log_probs = posterior.log_prob(samples_batched[:, idx], xos[idx])
        assert torch.allclose(log_probs, log_probs_batched[:, idx])
        assert torch.allclose(
            target_log_probs.exp(), log_probs.exp(), atol=0.4, rtol=0.4
        ), "Batched log probs are not consistent with non-batched log probs."
