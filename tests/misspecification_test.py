# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributions as dist
from torch import nn

from sbi.diagnostics.misspecification import (
    calc_misspecification_logprob,
    calc_misspecification_mmd,
)
from sbi.inference import NPE
from sbi.inference.trainers.marginal import MarginalTrainer
from sbi.neural_nets import MarginalNSFConfig, posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.utils.sbiutils import seed_all_backends
from sbi.utils.torchutils import process_device

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
    trainer = MarginalTrainer(density_estimator=MarginalNSFConfig())
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


def test_mmd_embedding_mode_requires_trained_inference():
    """An untrained trainer must raise a `ValueError`, not crash."""
    x = torch.randn(10, 2)

    with pytest.raises(ValueError, match="No neural net found"):
        calc_misspecification_mmd(x_obs=x[:1], x=x, inference=NPE(), mode="embedding")


def test_mmd_many_observations_gives_finite_baseline():
    """The MMD baseline must stay finite when ``n_obs >= max_samples``."""
    x_obs = torch.randn(1000, 2) + 5.0
    x = torch.randn(3000, 2)

    p_val, _ = calc_misspecification_mmd(
        x_obs=x_obs, x=x, n_shuffle=20, max_samples=1000
    )

    # All 20 baseline MMDs are below the observed one: the smallest p-value.
    assert p_val == pytest.approx(1 / 21)


def test_mmd_does_not_depend_on_the_order_of_x():
    """Sorted synthetic samples must not make a correct model look misspecified."""
    x = torch.sort(torch.randn(2000, 1), dim=0).values
    x_obs = torch.randn(50, 1)

    _, (mmds_baseline, mmd) = calc_misspecification_mmd(
        x_obs=x_obs, x=x, n_shuffle=20, max_samples=200
    )

    # Over 200 seeds the ratio stayed below 2.9 with random rows and above 6.6 when
    # the observed MMD took the first rows of x.
    assert mmd < 4 * mmds_baseline.max()


def test_mmd_raises_for_non_finite_mmd():
    """Many identical samples give a zero bandwidth: raise, never p = 1.0."""
    x = torch.poisson(torch.full((1000, 1), 0.1))
    x_obs = torch.poisson(torch.full((50, 1), 1.0))

    with pytest.raises(ValueError, match="MMD is not finite"):
        calc_misspecification_mmd(x_obs=x_obs, x=x, n_shuffle=20)


def test_mmd_accepts_nan_padded_trials_in_embedding_mode():
    """Whole NaN trials are padding for a permutation-invariant embedding."""
    embedding_net = PermutationInvariantEmbedding(
        trial_net=nn.Linear(2, 4), trial_net_output_dim=4, output_dim=3
    )
    inference = SimpleNamespace(
        _neural_net=SimpleNamespace(embedding_net=embedding_net)
    )
    x = torch.randn(200, 5, 2)
    x[::2, -2:] = float("nan")
    x_obs = torch.randn(2, 5, 2)
    x_obs[0, -1] = float("nan")

    p_val, _ = calc_misspecification_mmd(
        x_obs=x_obs, x=x, inference=inference, mode="embedding", n_shuffle=20
    )

    assert 0 < p_val <= 1


@pytest.mark.parametrize(
    "device, dtype",
    (
        ("cpu", torch.float64),
        pytest.param("gpu", torch.float32, marks=pytest.mark.gpu),
    ),
)
def test_mmd_keeps_device_and_dtype(device: str, dtype: torch.dtype):
    dev = process_device(device)
    x = torch.randn(300, 2, device=dev, dtype=dtype)

    _, (mmds_baseline, mmd) = calc_misspecification_mmd(
        x_obs=x[:5], x=x[5:], n_shuffle=20
    )

    assert mmds_baseline.device == mmd.device == x.device
    assert mmds_baseline.dtype == mmd.dtype == dtype
