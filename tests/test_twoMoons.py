import torch
import sbi.simulators as simulators
import sbi.utils as utils

from torch import distributions
from sbi.inference.apt import APT
import numpy as np

# use cpu by default
torch.set_default_tensor_type("torch.FloatTensor")

# seed the simulations
torch.manual_seed(0)


dim, std = 3, 0.5
simulator = simulators.TwoMoonsSimulator()
a = 1
prior = distributions.Uniform(low=-a * torch.ones(simulator.parameter_dim), high=a * torch.ones(simulator.parameter_dim))


parameter_dim, observation_dim = (simulator.parameter_dim, simulator.observation_dim)

true_observation = simulator.get_ground_truth_observation()
neural_posterior = utils.get_neural_posterior("maf", parameter_dim, observation_dim, simulator)

apt = APT(
    simulator=simulator,
    true_observation=true_observation,
    prior=prior,
    neural_posterior=neural_posterior,
    num_atoms=10,
    use_combined_loss=False,
    train_with_mcmc=False,
    mcmc_method="slice-np",
    summary_net=None,
    retrain_from_scratch_each_round=False,
    discard_prior_samples=False,
)


num_rounds, num_simulations_per_round = 2, 500
apt.run_inference(
    num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round, batch_size=20)

samples = apt.sample_posterior(10000)
samples = utils.tensor2numpy(samples)
target_samples = np.load('target_samples_twoMoons/samples_gt.npy')

num_samples = 1000
t1 = torch.tensor([target_samples], dtype=torch.float32)[0, :num_samples]
t2 = torch.tensor([samples], dtype=torch.float32)[0, :num_samples]

# compute the mmd
mmd = utils.unbiased_mmd_squared(t1, t2)
mmd = utils.tensor2numpy(mmd)

# check if mmd is larger than expected
max_mmd = 0.02
assert mmd < max_mmd, "MMD was larger than expected."