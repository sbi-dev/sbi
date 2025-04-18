# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
import torch.distributions as dist

from sbi.diagnostics.misspecification import (
    calc_misspecification_logprob,
    calc_misspecification_mmd,
)
from sbi.inference import NPE
from sbi.inference.trainers.marginal import MarginalTrainer
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.utils.sbiutils import seed_all_backends

seed = 2025


# Use seed automatically for every test function.
@pytest.fixture(autouse=True)
def set_seed():
    seed_all_backends(seed)


@pytest.mark.parametrize("D, N", ((2, 1000),))
def test_mmd_x_space(D: int, N: int):
    """MMD in x-space on Gaussian data.
    Args:
        D: observation and parameter dimension
        N: number of samples
    """

    # true prior -- the observation comes from here
    mean_true = torch.zeros(D)
    cov_true = torch.eye(D)
    prior_true = dist.MultivariateNormal(loc=mean_true, covariance_matrix=cov_true)

    # misspecified prior
    offset = 4
    prior_mis = dist.MultivariateNormal(
        loc=mean_true + offset, covariance_matrix=cov_true
    )

    def simulator(theta):
        return theta + torch.randn_like(theta)

    # validation set to compute MMD distribution in the
    # well-specified case
    # this could just be a subset of the training data
    num_validations_mmd = 1000
    theta_val = prior_true.sample((num_validations_mmd,))
    x_val = simulator(theta_val)

    # generate observations from the well and the misspecified model
    # do inference given observed data
    num_observations = 1
    theta_o = prior_true.sample((num_observations,))
    x_o = simulator(theta_o)
    theta_o_mis = prior_mis.sample((num_observations,))
    x_o_mis = simulator(theta_o_mis)

    # perform two tests for misspecification
    # 1. well specified model
    p_val_well, _ = calc_misspecification_mmd(
        inference=None,
        x_obs=x_o,
        x=x_val,
        mode="x_space",
    )

    # 2. misspecified model
    p_val_mis, _ = calc_misspecification_mmd(
        inference=None,
        x_obs=x_o_mis,
        x=x_val,
        mode="x_space",
    )
    # check p_vals
    assert p_val_well > 0.05, (
        f"Expected large p_val for well-specified data, obtained {p_val_well}"
    )
    assert p_val_mis < 0.05, (
        f"Expected small p_val for misspecified data, obtained {p_val_mis}"
    )


@pytest.mark.parametrize("D, N", ((2, 1000),))
def test_mmd_x_emedding(D: int, N: int):
    """MMD in x-space on Gaussian data.
    Args:
        D: observation and parameter dimension
        N: number of samples
    """

    # true prior -- the observation comes from here
    mean_true = torch.zeros(D)
    cov_true = torch.eye(D)
    prior_true = dist.MultivariateNormal(loc=mean_true, covariance_matrix=cov_true)

    # misspecified prior
    offset = 4
    prior_mis = dist.MultivariateNormal(
        loc=mean_true + offset, covariance_matrix=cov_true
    )

    def simulator(theta):
        return theta + torch.randn_like(theta)

    # generate training data for well-specified model:
    theta_train = prior_true.sample((N,))
    x_train = simulator(theta_train)

    # validation set to compute MMD distribution in the
    # well-specified case
    # this could just be a subset of the training data
    num_validations_mmd = 1000
    theta_val = prior_true.sample((num_validations_mmd,))
    x_val = simulator(theta_val)

    # generate observations from the well and the misspecified model
    # do inference given observed data
    num_observations = 1
    theta_o = prior_true.sample((num_observations,))
    x_o = simulator(theta_o)
    theta_o_mis = prior_mis.sample((num_observations,))
    x_o_mis = simulator(theta_o_mis)

    # train NPE networks
    emb_net = FCEmbedding(
        input_dim=D, output_dim=D, num_layers=3, num_hiddens=20
    )  # minimal embedding network
    neural_posterior = posterior_nn(model="maf", embedding_net=emb_net)
    inference = NPE(prior=prior_true, density_estimator=neural_posterior)
    inference = inference.append_simulations(theta_train, x_train)
    _ = inference.train()

    # perform two tests for misspecification
    # 1. well specified model
    p_val_well, _ = calc_misspecification_mmd(
        inference=inference,
        x_obs=x_o,
        x=x_val,
        mode="embedding",
    )

    # 2. misspecified model
    p_val_mis, _ = calc_misspecification_mmd(
        inference=inference,
        x_obs=x_o_mis,
        x=x_val,
        mode="embedding",
    )
    # check p_vals
    assert p_val_well > 0.05, (
        f"Expected large p_val for well-specified data, obtained {p_val_well}"
    )
    assert p_val_mis < 0.05, (
        f"Expected small p_val for misspecified data, obtained {p_val_mis}"
    )


@pytest.mark.parametrize("D, N", ((2, 1000),))
# @pytest.mark.slow
def test_marginal_log_prob_x_space(D: int, N: int):
    """log prob in x-space on Gaussian data.
    Args:
        D: observation and parameter dimension
        N: number of samples
    """

    # true prior -- the observation comes from here
    mean_true = torch.zeros(D)
    cov_true = torch.eye(D)
    prior_true = dist.MultivariateNormal(loc=mean_true, covariance_matrix=cov_true)

    # misspecified prior
    offset = 4
    prior_mis = dist.MultivariateNormal(
        loc=mean_true + offset, covariance_matrix=cov_true
    )

    def simulator(theta):
        return theta + torch.randn_like(theta)

    # validation set to compute MMD distribution in the
    # well-specified case
    # this could just be a subset of the training data
    num_validations_mmd = 1000
    theta_val = prior_true.sample((num_validations_mmd,))
    x_val = simulator(theta_val)

    # generate observations from the well and the misspecified model
    # do inference given observed data
    num_observations = 1
    theta_o = prior_true.sample((num_observations,))
    x_o = simulator(theta_o)
    theta_o_mis = prior_mis.sample((num_observations,))
    x_o_mis = simulator(theta_o_mis)

    # fit the marginal
    num_training_marginal = 1000
    theta_train = prior_true.sample((num_training_marginal,))
    x_train = simulator(theta_train)

    # Instantiate a trainer for the marginal pdf and train it
    trainer = MarginalTrainer(density_estimator='NSF')
    print(f"Training marginal q(x) on {x_train.shape[0]} samples...")
    trainer.append_samples(x_train)
    est = trainer.train(max_num_epochs=3000)

    # perform tests for misspecification
    # 1. well specified model
    p_val_well, reject_H0_well = calc_misspecification_logprob(x_val, x_o, est)

    # 2. misspecified model
    p_val_mis, reject_H0_mis = calc_misspecification_logprob(x_val, x_o_mis, est)

    # check p_vals
    assert p_val_well > 0.05, (
        f"Expected large p_val for well-specified data, obtained {p_val_well}"
    )
    assert p_val_mis < 0.05, (
        f"Expected small p_val for misspecified data, obtained {p_val_mis}"
    )
