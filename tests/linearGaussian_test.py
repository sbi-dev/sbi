import sys
import os

sys.path.append('../inference/')
sys.path.append('../../')
os.environ['LFI_PROJECT_DIR'] = os.getcwd()

import torch
import sbi.simulators as simulators
import sbi.utils as utils
from torch import distributions
if torch.cuda.is_available(): torch.set_default_tensor_type("torch.cuda.FloatTensor")
else: torch.set_default_tensor_type("torch.FloatTensor")
from apt import APT

# seed the simulations
torch.manual_seed(0)

# will be called by pytest. Then runs test_compute(num_dim) for 1D and 3D
def pytest_generate_tests(metafunc):
    metafunc.parametrize("num_dim", [1, 3])


def test_compute(num_dim):

    dim, std = num_dim, 1.0
    simulator = simulators.LinearGaussianSimulator(dim=dim, std=std)
    prior = distributions.MultivariateNormal(loc=torch.zeros(dim), covariance_matrix=torch.eye(dim))

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
    num_rounds, num_simulations_per_round = 1, 1000
    apt.run_inference(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    # draw samples from posterior
    samples = apt.sample_posterior(1000)
    samples = utils.tensor2numpy(samples)
    samples = torch.from_numpy(samples)

    # define target distribution (analytically tractable) and sample from it
    if num_dim == 1:    target_dist = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.5]))
    elif num_dim == 3:  target_dist = torch.distributions.MultivariateNormal(torch.tensor([0.0, 0.0, 0.0]), 0.5*torch.eye(3))
    target_samples = target_dist.sample([1000])

    # compute the mmd
    mmd = utils.unbiased_mmd_squared(target_samples, samples)
    mmd = utils.tensor2numpy(mmd)

    # check if mmd is larger than expected
    if num_dim == 1:   max_mmd = 0.07
    elif num_dim == 3: max_mmd = 0.06
    assert mmd < max_mmd, "MMD was larger than expected."