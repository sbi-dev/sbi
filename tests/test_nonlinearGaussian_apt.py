import os
import sys

import numpy as np
import pytest
import torch
from torch import distributions

import sbi.utils as utils
from sbi.inference.snpe.snpe_c import APT
from sbi.simulators.nonlinear_gaussian import (
    get_ground_truth_posterior_samples_nonlinear_gaussian,
    non_linear_gaussian,
)
from sbi.simulators.simutils import set_simulator_attributes

# use cpu by default
torch.set_default_tensor_type("torch.FloatTensor")

# seed the simulations
torch.manual_seed(0)


@pytest.mark.slow
def test_nonlinearGaussian_based_on_mmd():
    simulator = non_linear_gaussian

    # ground truth parameters as specified in 'Sequential Neural Likelihood' paper.
    ground_truth_parameters = torch.tensor([-0.7, -2.9, -1.0, -0.9, 0.6])
    # ground truth observation using same seed as 'Sequential Neural Likelihood' paper.
    ground_truth_observation = torch.tensor(
        [
            -0.97071232,
            -2.94612244,
            -0.44947218,
            -3.42318484,
            -0.13285634,
            -3.36401699,
            -0.85367595,
            -2.42716377,
        ]
    )

    # assume batch dims
    parameter_dim = ground_truth_parameters.shape[0]
    observation_dim = ground_truth_observation.shape[0]

    prior = distributions.Uniform(
        low=-3 * torch.ones(parameter_dim), high=3 * torch.ones(parameter_dim),
    )

    apt = APT(
        simulator=simulator,
        true_observation=ground_truth_observation[None,],
        prior=prior,
        num_atoms=-1,
        z_score_obs=True,
        use_combined_loss=False,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
    )

    # run inference
    num_rounds, num_simulations_per_round = 2, 1000
    apt.run_inference(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    # draw samples from posterior
    samples = apt._neural_posterior.sample(1000)

    # define target distribution (analytically tractable) and sample from it
    target_samples = get_ground_truth_posterior_samples_nonlinear_gaussian(
        num_samples=1000
    )

    # compute the mmd
    mmd = utils.unbiased_mmd_squared(target_samples.float(), samples.float())

    # check if mmd is larger than expected
    max_mmd = 0.16  # mean mmd plus 2 stds.
    assert mmd < max_mmd, f"MMD={mmd} larger than mean plus 2 stds."
