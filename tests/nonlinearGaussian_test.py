import os
import sys

import numpy as np
import sbi.simulators as simulators
import sbi.utils as utils
import torch
from sbi.inference.apt import APT
from torch import distributions

# use cpu by default
torch.set_default_tensor_type("torch.FloatTensor")

# seed the simulations
torch.manual_seed(0)


def test_nonlinearGaussian_based_on_mmd():
    task = "nonlinear-gaussian"
    simulator, prior = simulators.get_simulator_and_prior(task)

    parameter_dim, observation_dim = (
        simulator.parameter_dim,
        simulator.observation_dim,
    )
    true_observation = simulator.get_ground_truth_observation()

    # define nn for inference
    neural_posterior = utils.get_neural_posterior(
        "maf", parameter_dim, observation_dim, simulator
    )
    apt = APT(
        simulator=simulator,
        true_observation=true_observation,
        prior=prior,
        neural_posterior=neural_posterior,
        num_atoms=-1,
        use_combined_loss=False,
        train_with_mcmc=False,
        mcmc_method="slice-np",
        summary_net=None,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
    )

    # run inference
    num_rounds, num_simulations_per_round = 2, 1000
    apt.run_inference(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    # draw samples from posterior
    samples = apt.sample_posterior(1000)

    # define target distribution (analytically tractable) and sample from it
    target_samples = simulator.get_ground_truth_posterior_samples(num_samples=1000)

    # compute the mmd
    mmd = utils.unbiased_mmd_squared(target_samples, samples)

    # check if mmd is larger than expected
    max_mmd = 0.0
    assert mmd < max_mmd, f"MMD={mmd} no mmd thresold calculated yet."
