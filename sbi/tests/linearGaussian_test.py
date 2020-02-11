import sys
sys.path.append('../inference/')
sys.path.append('../../')

import os
import torch
import sbi.simulators as simulators
import sbi.utils as utils
from torch import distributions
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

from apt import APT
os.environ['LFI_PROJECT_DIR'] = '/Users/deismic/Documents/PhD/sbi/sbi/project_dir'


dim, std = 3, 0.5
simulator = simulators.LinearGaussianSimulator(dim=dim, std=std)
prior = distributions.MultivariateNormal(loc=torch.zeros(dim), covariance_matrix=torch.eye(dim))

parameter_dim, observation_dim = (simulator.parameter_dim, simulator.observation_dim)

true_observation = simulator.get_ground_truth_observation()
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

num_rounds, num_simulations_per_round = 1, 500
apt.run_inference(
    num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
)

samples = apt.sample_posterior(2500)
samples = utils.tensor2numpy(samples)
figure = utils.plot_hist_marginals(
    data=samples,
    ground_truth=utils.tensor2numpy(
        simulator.get_ground_truth_parameters()
    ).reshape(-1),
    lims=simulator.parameter_plotting_limits,
)
