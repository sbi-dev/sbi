# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.diagnostics.sbc import _run_sbc, check_sbc
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
from tests.test_utils import check_c2st


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


@pytest.mark.parametrize(
    "snplre_method", [SNPE_A, SNPE_C, SNLE_A, SNRE_A, SNRE_B, SNRE_C]
)
def test_importance_posterior_sample_log_prob(snplre_method: type):
    num_dim = 2

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = snplre_method(prior=prior)
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "sbi_method", [SNPE_C]
)  # TODO: add test for SNLE with slice_np_vectorized
@pytest.mark.parametrize("density_estimator", ["maf", "mdn"])
@pytest.mark.parametrize("calib_method", ("sbc", "coverage"))
def test_batched_vs_sequential_sampling_with_coverage(
    sbi_method, density_estimator: str, calib_method: str
):
    """Compare batched vs sequential sampling on a batch of idential
    observations.

    checks whether samples indistinguishable in terms of c2st.

    checks whether sbc or coverage calibration passes.
    """
    num_dim = 2
    num_simulations = 2000

    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))
    simulator = diagonal_linear_gaussian

    inference = sbi_method(prior=prior, density_estimator=density_estimator)
    theta, x = simulate_for_sbi(simulator, prior, num_simulations)
    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(mcmc_method="slice_np_vectorized")

    # Batch of observations.
    xs = torch.zeros(10, num_dim)

    samples_batched = posterior.sample_batched((100,), xs).reshape(1000, num_dim)
    samples_sequential = torch.stack([
        posterior.sample((100,), x=xi, show_progress_bars=False) for xi in xs
    ]).reshape(1000, num_dim)

    assert samples_batched.shape == samples_sequential.shape
    check_c2st(
        samples_batched, samples_sequential, alg="batched vs sequential sampling"
    )

    # Check posterior calibration
    num_sbc_samples = 100
    num_posterior_samples = 1000
    thetas = prior.sample((num_sbc_samples,))
    xs = simulator(thetas)

    posterior_samples_batched = posterior.sample_batched(
        (num_posterior_samples,), xs, show_progress_bars=False
    ).transpose(0, 1)  # (num_sbc_samples, num_posterior_samples, num_dim)
    posterior_samples_sequential = torch.stack([
        posterior.sample((num_posterior_samples,), x=xi, show_progress_bars=False)
        for xi in xs
    ])

    labels = ["batched", "sequential"]
    for idx, samples in enumerate([
        posterior_samples_batched,
        posterior_samples_sequential,
    ]):
        daps = samples[:, 0, :]
        ranks = _run_sbc(
            thetas,
            xs,
            samples,
            posterior,
            reduce_fns="marginals" if calib_method == "sbc" else posterior.log_prob,
        )

        checks = check_sbc(
            ranks,
            prior.sample((num_sbc_samples,)),
            daps,
            num_posterior_samples,
        )

        assert (
            checks["ks_pvals"] > 0.05
        ).all(), f"{calib_method} ks_pvals failed for {labels[idx]} sampling: {checks}"
        assert (
            checks["c2st_ranks"] < 0.6
        ).all(), (
            f"{calib_method} c2st_ranks failed for {labels[idx]} sampling: {checks}"
        )
        assert (
            checks["c2st_dap"] < 0.6
        ).all(), f"{calib_method} c2st_dap failed for {labels[idx]} sampling: {checks}"
