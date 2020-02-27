import os
import sys

import numpy as np
import pytest
import torch
from torch import distributions

import sbi.simulators as simulators
import sbi.utils as utils
from sbi.inference.snpe.snpe_c import APT

# use cpu by default
torch.set_default_tensor_type("torch.FloatTensor")

# seed the simulations
torch.manual_seed(0)


@pytest.mark.slow
def test_nonlinearGaussian_based_on_mmd():
    task = "nonlinear-gaussian"
    (
        simulator,
        prior,
        ground_truth_parameters,
        ground_truth_observation,
    ) = simulators.get_simulator_prior_and_groundtruth(task)

    # assume batch dims
    parameter_dim = ground_truth_parameters.shape[0]
    observation_dim = ground_truth_observation.shape[0]

    print("here", ground_truth_observation)

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
    target_samples = simulator.get_ground_truth_posterior_samples(num_samples=1000)

    # compute the mmd
    mmd = utils.unbiased_mmd_squared(target_samples, samples)

    # check if mmd is larger than expected
    max_mmd = 0.16  # mean mmd plus 2 stds.
    assert mmd < max_mmd, f"MMD={mmd} larger than mean plus 2 stds."
