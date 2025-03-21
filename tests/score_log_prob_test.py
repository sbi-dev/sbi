# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from sbi.inference import NPSE
from sbi.simulators import linear_gaussian, true_posterior_linear_gaussian_mvn_prior


@pytest.mark.parametrize("num_dims", [1, 2])
@pytest.mark.parametrize("iid_batch_size", [1, 2])
def test_score_fn_log_prob(num_dims, iid_batch_size):
    '''
    Tests the log-probability computation of the score-based posterior.

    This test evaluates the ability of the score-based posterior to recover the
    true posterior with log probabilities as importance weights. It compares whether
    the effective sample size (ESS) is sufficiently high in comparision to the number
    of samples drawn from the proposal posterior.

    Args:
        num_dims (int): The number of dimensions for the Gaussian prior and simulator.
        iid_batch_size (int): The number of independent and identically distributed(IID)
        observations in the batch. iid_batch_size=1 corresponds to a single observation.

    Steps:
        1. Gaussian prior and simulator are assumed to construct true posterior.
        2. Train a score-based posterior estimator (NPSE) using the simulations.
        3. Build a proposal posterior and sample from it.
        4. Calculate the ESS with importance weights based on log probabilities.

    Raises:
        AssertionError: If the ESS is less than half the number of posterior samples,
        indicating poor recovery of the true posterior.
    '''
    num_sims = 10000
    num_posterior_samples = 1000
    # Prior Gaussian
    prior_mean = torch.zeros(num_dims)
    prior_diag = torch.ones(num_dims)
    prior_std = torch.diag(prior_diag)
    prior = torch.distributions.Independent(
        torch.distributions.Normal(prior_mean, prior_diag), 1
    )
    # Simulator Gaussian
    sim_mean = torch.ones(num_dims)
    sim_diag = torch.linspace(2.0, 0.5, num_dims)
    sim_std = torch.diag(sim_diag)
    simulator = lambda theta: linear_gaussian(
        theta, likelihood_shift=sim_mean, likelihood_cov=sim_std
    )

    # Produce simulations
    theta = prior.sample((num_sims,))
    x = simulator(theta)

    # Ground truth theta
    theta_o = torch.zeros(num_dims).unsqueeze(0)
    x_o = simulator(theta_o.repeat(iid_batch_size, 1))
    true_posterior = true_posterior_linear_gaussian_mvn_prior(
        x_o, sim_mean, sim_std, prior_mean, prior_std
    )

    # NPSE training
    inference = NPSE(prior=prior, sde_type="vp")
    inference.append_simulations(theta, x)
    score_estimator = inference.train()

    proposal_posterior = inference.build_posterior(
        score_estimator=score_estimator, prior=prior, sample_with="sde"
    )

    proposal_samples = proposal_posterior.sample((num_posterior_samples,), x=x_o)
    recovered_prob = proposal_posterior.log_prob(proposal_samples, x=x_o)
    true_prob = true_posterior.log_prob(proposal_samples)

    ess = _compute_ess(recovered_prob, true_prob)
    assert ess > num_posterior_samples / 2, (
        f"Effective sample size : {ess} too low \
            for number of samples {num_posterior_samples}"
    )


def _compute_ess(proposal_log_weights: Tensor, true_log_weights: Tensor):
    '''
    Computes the Effective Sample Size (ESS) based on importance weights.

    The ESS is a measure of how well the proposal distribution approximates
    the true distribution. It is calculated using the normalized importance
    weights derived from the log-probabilities of the proposal and true distributions.
    For more details refer :
    https://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html

    Args:
        proposal_log_weights: Log-probs of samples from proposal w.r.t. the proposal.
        true_log_weights: Log-probs of the same samples w.r.t. true distribution.

    Returns:
        float: The Effective Sample Size (ESS), where a higher value indicates
        better approximation of the true distribution by the proposal distribution.
    '''
    importance_weights = true_log_weights - proposal_log_weights
    norm_weights = torch.softmax(importance_weights, dim=0)
    return 1 / torch.sum(torch.square(norm_weights))
