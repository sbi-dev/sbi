import os
import pickle

import sbi.simulators as simulators
import sbi.utils as utils
import torch
from pyknos.nflows import distributions as distributions_
from torch import distributions


# TODO: we do not want this function
def simulation_wrapper(simulator, parameter_sample_fn, num_samples):

    if isinstance(simulator, simulators.LotkaVolterraSimulator):

        if not simulator._has_been_used:
            parameters, observations = simulator._get_prior_parameters_observations()
            return (
                torch.Tensor(parameters)[:num_samples],
                torch.Tensor(observations)[:num_samples],
            )

        else:
            num_remaining_samples = num_samples
            parameters, observations = [], []

            while num_remaining_samples > 0:

                proposed_parameters = parameter_sample_fn(num_remaining_samples)
                proposed_observations = simulator(proposed_parameters)

                for parameter, observation in zip(
                    proposed_parameters, proposed_observations
                ):
                    if observation is not None:
                        parameters.append(parameter.reshape(1, -1))
                        observations.append(torch.Tensor(observation.reshape(1, -1)))

                num_remaining_samples = num_samples - len(parameters)

            return torch.cat(parameters), torch.cat(observations)

    else:
        parameters = parameter_sample_fn(num_samples)
        observations = simulator(parameters)
        return torch.Tensor(parameters), torch.Tensor(observations)


def get_simulator_prior_and_groundtruth(task):

    if task == "nonlinear-gaussian":
        simulator = simulators.NonlinearGaussianSimulator()
        prior = distributions.Uniform(
            low=-3 * torch.ones(simulator.parameter_dim),
            high=3 * torch.ones(simulator.parameter_dim),
        )
        # ground truth parameters as specified in 'Sequential Neural Likelihood' paper.
        ground_truth_parameters = torch.Tensor([-0.7, -2.9, -1.0, -0.9, 0.6])
        # ground truth observation using same seed as 'Sequential Neural Likelihood' paper.
        ground_truth_observation = torch.Tensor(
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

    elif task == "nonlinear-gaussian-gaussian":
        simulator = simulators.NonlinearGaussianSimulator()
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(5), covariance_matrix=torch.eye(5)
        )
        # ground truth parameters as specified in 'Sequential Neural Likelihood' paper.
        ground_truth_parameters = torch.Tensor([-0.7, -2.9, -1.0, -0.9, 0.6])
        # ground truth observation using same seed as 'Sequential Neural Likelihood' paper.
        ground_truth_observation = torch.Tensor(
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

    elif task == "two-moons":
        simulator = simulators.TwoMoonsSimulator()
        a = 2
        prior = distributions.Uniform(
            low=-a * torch.ones(simulator.parameter_dim),
            high=a * torch.ones(simulator.parameter_dim),
        )
        # dummy ground truth parameters as none are specified in 'Automatic Posterior Transformation.'
        ground_truth_parameters = torch.Tensor([0, 0])
        ground_truth_observation = torch.Tensor([0, 0])

    elif task == "linear-gaussian":
        dim, std = 20, 0.5
        simulator = simulators.LinearGaussianSimulator(dim=dim, std=std)
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(dim), covariance_matrix=torch.eye(dim)
        )
        ground_truth_parameters = torch.zeros(dim)
        ground_truth_observation = simulator(ground_truth_parameters)

    elif task == "lotka-volterra":
        simulator = simulators.LotkaVolterraSimulator(
            summarize_observations=True, gaussian_prior=False
        )
        prior = distributions.Uniform(
            low=-5 * torch.ones(simulator.parameter_dim),
            high=2 * torch.ones(simulator.parameter_dim),
        )
        ground_truth_parameters = torch.log(torch.Tensor([0.01, 0.5, 1.0, 0.01]))
        path = os.path.join(utils.get_data_root(), "lotka-volterra", "obs_stats.pkl")
        with open(path, "rb") as file:
            true_observation = pickle.load(file, encoding="bytes")
        ground_truth_observation = torch.Tensor(true_observation)

    elif task == "lotka-volterra-gaussian":
        simulator = simulators.LotkaVolterraSimulator(
            summarize_observations=True, gaussian_prior=True
        )
        prior = distributions.MultivariateNormal(
            loc=torch.zeros(4), covariance_matrix=2 * torch.eye(4)
        )
        ground_truth_parameters = torch.log(torch.Tensor([0.01, 0.5, 1.0, 0.01]))
        path = os.path.join(utils.get_data_root(), "lotka-volterra", "obs_stats.pkl")
        with open(path, "rb") as file:
            true_observation = pickle.load(file, encoding="bytes")
        ground_truth_observation = torch.Tensor(true_observation)

    elif task == "mg1":
        simulator = simulators.MG1Simulator()
        prior = distributions_.MG1Uniform(
            low=torch.zeros(3), high=torch.Tensor([10.0, 10.0, 1.0 / 3.0])
        )
        ground_truth_parameters = torch.Tensor([1.0, 5.0, 0.2])
        path = os.path.join(utils.get_data_root(), "mg1", "observed_data.pkl")
        with open(path, "rb") as file:
            _, true_observation = pickle.load(file, encoding="bytes")
        ground_truth_observation = torch.Tensor(true_observation)

    else:
        raise ValueError(f"'{task}' simulator choice not understood.")

    return simulator, prior, ground_truth_parameters, ground_truth_observation
