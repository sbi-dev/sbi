# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
import torch.distributions as dist

from sbi.diagnostics.misspecification import calc_misspecification_mmd

seed = 2025
torch.manual_seed(seed)


@pytest.mark.parametrize("D, N", ((2, 1000)))
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

    # generate training data for well-specified model:
    # not needed for this test
    # theta_train = prior_true.sample((N,))
    # x_train = simulator(theta_train)

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
    print(p_val_well, p_val_mis)
    # check p_vals
    assert p_val_well > 0.05, f"Expected large p_val , obtained {p_val_well}"
    assert p_val_mis < 0.05, f"Expected small p_val , obtained {p_val_mis}"
