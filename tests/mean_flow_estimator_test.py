# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch import Tensor, eye, ones, zeros
from torch.distributions import MultivariateNormal, Normal

from sbi.inference import NLE
from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters
from sbi.neural_nets import MeanFlowConfig
from sbi.neural_nets.estimators import MeanFlowEstimator, mean_flow_estimator
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.metrics import check_c2st


def _build(input_dim: int = 2, condition_dim: int = 3, num_simulations: int = 200):
    condition = torch.randn(num_simulations, condition_dim)
    input = 2.0 * torch.randn(num_simulations, input_dim) + 1.0
    estimator = MeanFlowConfig(hidden_features=32).build(input, condition)
    return estimator, input, condition


def _randomize_outputs(estimator: MeanFlowEstimator) -> None:
    for output in estimator.net.outputs.values():
        torch.nn.init.normal_(output.weight, std=0.2)


def test_import_in_fresh_process():
    """The builder sits in an import cycle with the vector-field configs."""
    code = "import sbi.neural_nets, sbi.inference; sbi.neural_nets.MeanFlowConfig"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_config_builds_mean_flow_estimator():
    estimator, _, _ = _build()
    assert isinstance(estimator, MeanFlowEstimator)
    assert estimator.num_steps == 2


@pytest.mark.parametrize("input_dim", (1, 3))
@pytest.mark.parametrize("condition_dim", (1, 2))
@pytest.mark.parametrize("sample_dim", (1, 4))
@pytest.mark.parametrize("batch_dim", (1, 5))
def test_shapes(input_dim: int, condition_dim: int, sample_dim: int, batch_dim: int):
    estimator, input, condition = _build(input_dim, condition_dim)
    _randomize_outputs(estimator)

    log_prob = estimator.log_prob(
        input[:batch_dim].expand(sample_dim, batch_dim, input_dim),
        condition[:batch_dim],
    )
    assert log_prob.shape == (sample_dim, batch_dim)
    assert torch.isfinite(log_prob).all()

    samples = estimator.sample((sample_dim,), condition[:batch_dim])
    assert samples.shape == (sample_dim, batch_dim, input_dim)

    loss = estimator.loss(input[:batch_dim], condition[:batch_dim])
    assert loss.shape == (batch_dim,)


@pytest.mark.parametrize("num_steps", (1, 3))
def test_untrained_flow_is_the_standardized_normal(num_steps: int):
    """Zero output layers make every flow map the identity."""
    estimator, input, condition = _build()

    log_prob = estimator.log_prob(input[:7], condition[:7], num_steps=num_steps)

    standardized = (input[:7] - estimator._mean_input) / estimator._std_input
    expected = (
        Normal(0.0, 1.0).log_prob(standardized).sum(-1)
        - torch.log(estimator._std_input).sum()
    )
    assert torch.allclose(log_prob[0], expected, atol=1e-5)


@pytest.mark.parametrize("num_steps", (1, 3))
def test_log_prob_matches_brute_force_jacobians(num_steps: int):
    estimator, input, condition = _build()
    _randomize_outputs(estimator)

    log_prob = estimator.log_prob(input[:4], condition[:4], num_steps=num_steps)[0]

    condition_emb = estimator.embedding_net(condition[:4])
    times = estimator._time_schedule(num_steps, input)
    expected = []
    for i in range(4):
        z = ((input[i] - estimator._mean_input) / estimator._std_input)[None]
        log_abs_det = 0.0
        for s, t in zip(times[:-1], times[1:], strict=True):

            def step(z_, emb=condition_emb[i : i + 1], s=s, t=t):
                return estimator._flow_map(z_, emb, s, t)

            jacobian = torch.autograd.functional.jacobian(step, z)[0, :, 0, :]
            log_abs_det = log_abs_det + torch.linalg.slogdet(jacobian).logabsdet
            z = step(z)
        expected.append(
            Normal(0.0, 1.0).log_prob(z).sum()
            + log_abs_det
            - torch.log(estimator._std_input).sum()
        )
    assert torch.allclose(log_prob, torch.stack(expected), atol=1e-4)


def test_log_prob_is_differentiable_in_condition():
    """Gradient-based MCMC on an NLE posterior differentiates in the parameters."""
    estimator, input, condition = _build()
    _randomize_outputs(estimator)
    condition = condition[:4].clone().requires_grad_(True)

    estimator.log_prob(input[:4], condition).sum().backward()

    assert torch.isfinite(condition.grad).all()
    assert condition.grad.abs().sum() > 0


def test_chunked_log_prob_matches_single_pass(monkeypatch):
    """`log_prob` splits rows into chunks to bound the memory of the Jacobians."""
    estimator, input, condition = _build(num_simulations=50)
    _randomize_outputs(estimator)
    single_pass = estimator.log_prob(input, condition)

    monkeypatch.setattr(mean_flow_estimator, "LOG_PROB_CHUNK_SIZE", 7)

    assert torch.allclose(estimator.log_prob(input, condition), single_pass)


def test_float64_statistics_keep_their_precision():
    mean = torch.tensor([100000001.0], dtype=torch.float64)
    estimator = MeanFlowConfig(hidden_features=8).build(
        torch.randn(10, 1, dtype=torch.float64), torch.randn(10, 1, dtype=torch.float64)
    )
    estimator = MeanFlowEstimator(
        estimator.net,
        torch.Size([1]),
        torch.Size([1]),
        mean_input=mean,
        std_input=torch.ones(1, dtype=torch.float64),
    ).double()

    log_prob = estimator.log_prob(mean[None], torch.zeros(1, 1, dtype=torch.float64))

    assert torch.allclose(
        log_prob, Normal(0.0, 1.0).log_prob(torch.tensor(0.0)).double()
    )


def test_sample_and_log_prob_keeps_sample_dims():
    estimator, _, condition = _build()
    _randomize_outputs(estimator)

    samples, log_probs = estimator.sample_and_log_prob(
        torch.Size((2, 3)), condition[:4]
    )

    assert samples.shape == (2, 3, 4, 2)
    assert log_probs.shape == (2, 3, 4)
    assert torch.allclose(
        log_probs,
        estimator.log_prob(samples.reshape(6, 4, 2), condition[:4]).reshape(2, 3, 4),
        atol=1e-5,
    )


def test_loss_trains_every_head():
    estimator, input, condition = _build()
    _randomize_outputs(estimator)

    estimator.loss(input[:32], condition[:32]).mean().backward()

    for head in ("mean", "velocity"):
        gradient = estimator.net.outputs[head].weight.grad
        assert gradient is not None and gradient.abs().sum() > 0


def test_cycle_loss_is_active_when_all_mean_flow_pairs_have_equal_times(monkeypatch):
    """The cycle draws its own time pairs, so it trains the maps even when every
    mean-flow pair sits on t = s, where a round trip would be trivially exact."""
    estimator, input, condition = _build()
    _randomize_outputs(estimator)
    monkeypatch.setattr(mean_flow_estimator, "EQUAL_TIMES_FRACTION", 1.0)

    def loss_with_cycle_weight(weight: float) -> Tensor:
        monkeypatch.setattr(mean_flow_estimator, "CYCLE_WEIGHT", weight)
        torch.manual_seed(0)
        return estimator.loss(input[:32], condition[:32])

    cycle_part = loss_with_cycle_weight(1.0) - loss_with_cycle_weight(0.0)

    assert cycle_part.sum() > 1e-3


def test_paper_variant_has_the_parameter_count_of_the_paper_network():
    """Arruda et al.'s g-and-k mean flow (2-D observation, 3 parameters) has
    1,755,940 trainable weights, as their training log reports."""
    estimator = MeanFlowConfig(variant="paper").build(
        torch.randn(100, 2), torch.randn(100, 3)
    )

    assert sum(p.numel() for p in estimator.net.parameters()) == 1_755_940


def test_paper_variant_trains_and_evaluates():
    condition, input = torch.randn(64, 3), torch.randn(64, 2)
    estimator = MeanFlowConfig(hidden_features=32, variant="paper").build(
        input, condition
    )

    estimator.loss(input, condition).mean().backward()
    log_prob = estimator.log_prob(input[:4], condition[:4], num_steps=2)

    for head in ("mean", "velocity"):
        assert estimator.net.outputs[head].weight.grad.abs().sum() > 0
    assert estimator.net.frequencies.grad.abs().sum() > 0
    assert torch.isfinite(log_prob).all()


@pytest.mark.parametrize("num_steps", (0, 1.5, True))
def test_invalid_num_steps_raises(num_steps):
    estimator, _, _ = _build()
    with pytest.raises((TypeError, ValueError)):
        estimator.num_steps = num_steps


@pytest.mark.slow
@pytest.mark.mcmc
def test_c2st_nle_mean_flow_on_linear_gaussian(
    mcmc_params_accurate: MCMCPosteriorParameters,
):
    num_dim = 2
    num_simulations = 2000
    x_o = zeros(1, num_dim)
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))

    theta = prior.sample((num_simulations,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)
    inference = NLE(prior, density_estimator=MeanFlowConfig(), show_progress_bars=False)
    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(posterior_parameters=mcmc_params_accurate)
    samples = posterior.sample((1000,), x=x_o)

    target = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, zeros(num_dim), eye(num_dim)
    ).sample((1000,))
    check_c2st(samples, target, alg="nle-mean_flow")


def test_paper_variant_density_gradient_is_finite_for_large_activations():
    """NUTS differentiates the density, which holds Jacobians of the network, so the
    network's second derivatives must stay finite where activations are large."""
    condition, input = torch.randn(64, 3), torch.randn(64, 2)
    estimator = MeanFlowConfig(hidden_features=32, variant="paper").build(
        input, condition
    )
    with torch.no_grad():
        for parameter in estimator.net.parameters():
            parameter.mul_(40.0)
    theta = condition[:1].clone().requires_grad_(True)

    log_prob = estimator.log_prob(input[:5, None], theta, num_steps=1).sum()
    (gradient,) = torch.autograd.grad(log_prob, theta)

    assert torch.isfinite(gradient).all()
