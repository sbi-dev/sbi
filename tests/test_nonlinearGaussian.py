import sys
import os

import torch
import sbi.simulators as simulators
import sbi.utils as utils
from torch import distributions
from sbi.inference.apt import APT

# use cpu by default 
torch.set_default_tensor_type("torch.FloatTensor")

# seed the simulations
torch.manual_seed(0)


task = "nonlinear-gaussian"
simulator, prior = simulators.get_simulator_and_prior(task)

parameter_dim, observation_dim = (simulator.parameter_dim, simulator.observation_dim)
true_observation = simulator.get_ground_truth_observation()

# define nn for inference
neural_posterior = utils.get_neural_posterior("maf", parameter_dim, observation_dim, simulator)
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
samples = utils.tensor2numpy(samples)
samples = torch.from_numpy(samples)

import numpy as np
np.save('target_data/nonlinearGaussianSamples_20000sims', samples)

# define target distribution (analytically tractable) and sample from it
target_dist = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.5]))
target_samples = target_dist.sample([1000])

# compute the mmd
mmd = utils.unbiased_mmd_squared(target_samples, samples)
mmd = utils.tensor2numpy(mmd)

# check if mmd is larger than expected
max_mmd = 0.07
#assert mmd < max_mmd, "MMD was larger than expected."