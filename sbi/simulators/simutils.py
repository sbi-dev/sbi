import os
import pickle
from typing import Callable, Tuple

import torch
from pyknos.nflows import distributions as distributions_
from torch import distributions

import sbi.simulators as simulators
import sbi.utils as utils
from sbi.utils.torchutils import BoxUniform


def set_simulator_attributes(
    simulator_fun: Callable, prior: torch.distributions.Distribution, name=None
) -> Callable:
    """Add name and input and output dimension as attributes to the simulator function.
    
    Arguments:
        simulator_fun {Callable} -- simulator function taking parameters as input
        prior {torch.distributions.Distribution} -- prior as pytorch distributions object
    
    Keyword Arguments:
        name {[type]} -- name of the simulator, if None take __name__ (default: {None})
    
    Returns:
        Callable -- simualtor function with attributes name, parameter_dim, observation_dim.
    """

    parameter_dim, observation_dim = get_simulator_dimensions(simulator_fun, prior)
    if name is None:
        name = simulator_fun.__name__

    setattr(simulator_fun, "name", name)
    setattr(simulator_fun, "parameter_dim", parameter_dim)
    setattr(simulator_fun, "observation_dim", observation_dim)

    return simulator_fun


def get_simulator_dimensions(
    simulator_fun, prior: torch.distributions.Distribution
) -> Tuple[int, int]:
    """Infer simulator input output dimension from prior and simulating once. 
    
    Arguments:
        simulator_fun {function} -- simulator function taking parameter batch as only argument, return data. 
        parameter_sample_fun {function} -- prior function with kwarg 'num_samples'.
    
    Returns:
        dim_input [int] -- input dimension of simulator, i.e., parameter vector dimension.
        dim_output [int] -- output dimension of simualtor, i.e., dimension of data or summary stats.
    """
    # sample from prior to get parameter dimension
    param = prior.sample()
    dim_input = param.shape[0]

    # simulate once to get simulator output dimension
    data = simulator_fun(param)
    dim_output = data.shape[0]

    return dim_input, dim_output


def get_simulator_name(simulator_fun, name=None) -> str:
    """Get or set name of the simulator. 
    
    Returns function name if no name is given. 
    
    Arguments:
        simulator_fun {function} -- simulator function taking parameter batch as only argument, return data. 
    
    Keyword Arguments:
        name {string} -- name of the simualtor, e.g., 'linearGaussian' (default: {None})
    
    Returns:
        string -- Name of the simulator. 
    """
    if name is None:
        return simulator_fun.__name__
    else:
        simulator_fun.__name__ = name
        return name


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
        prior = BoxUniform(
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
        prior = BoxUniform(
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
        prior = BoxUniform(
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
