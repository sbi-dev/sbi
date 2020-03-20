import os
import pickle
from sbi.simulators.linear_gaussian import linear_gaussian
from typing import Callable, Tuple

import torch
from pyknos.nflows import distributions as distributions_
from torch.distributions import Distribution, MultivariateNormal, Uniform

import sbi.simulators as simulators
import sbi.utils as utils
from sbi.utils.torchutils import BoxUniform
import warnings
import numpy as np


def set_simulator_attributes(
    simulator_fun: Callable, prior: Distribution, observed_data: torch.Tensor, name=None
) -> Callable:
    """Add name and input and output dimension as attributes to the simulator function.
    
    Arguments:
        simulator_fun {Callable} -- simulator function taking parameters as input
        prior {torch.distributions.Distribution} -- prior as pytorch distributions object
        observed_data {torch.Tensor} -- Observed data points, x0
    
    Keyword Arguments:
        name {Optional(str)} -- name of the simulator, if None take __name__ (default: {None})
    
    Returns:
        Callable -- simualtor function with attributes name, parameter_dim, observation_dim.
    """

    parameter_dim, observation_dim = get_simulator_dimensions(prior, observed_data)
    if name is None:
        name = simulator_fun.__name__

    setattr(simulator_fun, "name", name)
    setattr(simulator_fun, "parameter_dim", parameter_dim)
    setattr(simulator_fun, "observation_dim", observation_dim)

    return simulator_fun


def check_prior_and_data_dimensions(prior: Distribution, observed_data: torch.Tensor):
    """Check prior event shape and data dimensionality and warn. 
    
    Arguments:
        prior {Distribution} -- [description]
        observed_data {torch.Tensor} -- [description]

    Raises: 
        warning if prior is Uniform and dim input > 1. 
        warning if observed data is multidimensional.
    """

    # infer parameter dim by simulating once
    dim_input = prior.sample().numel()

    if isinstance(prior, Uniform) and dim_input > 1:
        warnings.warn(
            f"The paramerer dimension (`event_shape`) of the simualtor inferred from the "
            "prior is D={dim_input}>1 and the prior PyTorch Uniform. Therefore, beware "
            "that you are using a `batch_shape` of {dim_input} implicitly and "
            "`event_shape` 1, because Pytorch does not support multivariate Uniform. "
            "Consider using a BoxUniform prior instead."
        )

    if observed_data.squeeze().ndim > 1:
        warnings.warn(
            "The `true_observation` Tensor has more than one dimension, i.e., it is a matrix "
            "of observed data or batch of observed data points. " 
            "SBI supports only single observed data points."
        )


def get_simulator_dimensions(
    prior: Distribution, observed_data: torch.Tensor
) -> Tuple[int, int]:
    """Return simulator input output dimension from prior and observed data. 
    
    Arguments:
        prior {Distribution} -- pytorch prior distribution with event and batch shapes
        observed_data {torch.Tensor} -- Observed data point, x0
    
    Returns:
        dim_input [int] -- input dimension of simulator, i.e., parameter vector dimension, event shape.
        dim_output [int] -- output dimension of simualtor, i.e., dimension of data or summary stats.
    """
    # infer parameter dim by sampling once
    return prior.sample().numel(), observed_data.numel()


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


# TODO: deprecated, can be removed
def simulation_wrapper(
    simulator: Callable, parameter_sample_fn: Callable, num_samples: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return parameters and simulated data given a simulator and parameters.

    Arguments:
        simulator {Callable} -- simulator function taking parameters and returning data,
                                both as torch.Tensor
        parameter_sample_fn {Callable} -- prior function, wrapped such that it takes
                                          int argument num_samples

    Keyword Arguments:
        num_samples {int} -- number of samples (default: {1})

    Returns:
        Tuple[torch.Tensor, torch.Tensor] -- sampled parameters, simulated data
    """
    parameters = parameter_sample_fn(num_samples)
    observations = simulator(parameters)
    return parameters, observations


def simulation_wrapper_batch(
        simulator: Callable,
        parameter_sample_fn: Callable,
        num_samples: int,
        simulation_batch_size: int,
        x_dim: torch.Size
):
    """
    Runs simulations in a loop. Each loop simulates simulation_batch_size parameter sets until
    a total of num_samples is reached.
    Features: - allows to simulate in batches of arbitrary size
              - if simulation_batch_size==-1, all simulations are run at the same time
              - simulator output can be list, np.array, or torch.Tensor

    Args:
        simulator: Simulator function.
            If simulation_batch_size == 1: takes in thetas of shape (num_dim_theta)
                and outputs xs of shape (num_dim_x)
            If simulation_batch_size > 1: takes in thetas of shape (simulation_batch_size, num_dim_theta)
                and outputs xs of shape (simulation_batch_size, num_dim_x)
        parameter_sample_fn: Function to call for generating theta, e.g. prior sampling
        num_samples: Number of simulations to run
        simulation_batch_size: Number of simulations that are run within a single batch
            If simulation_batch_size == -1, we run a batch with all simulations required,
            i.e. simulation_batch_size = num_samples
        x_dim: dimensionality of a single simulator output

    Returns: torch.Tensor parameters theta, torch.Tensor simulated data x
    """

    # generate theta by sampling from prior (round 1) or proposal (round > 1)
    parameters = parameter_sample_fn(num_samples)

    # if simulation_batch_size == -1, we run all simulations in a single batch
    if simulation_batch_size == -1:
        simulation_batch_size = num_samples

    # initialize an empty array that stores the simulation outputs
    # I opted to use numpy here since torch.Tensors are automatically converted to numpy,
    # but np.arrays are not automatically converted to torch.Tensors
    all_x_shape = np.concatenate((np.asarray([0], dtype=int), np.asarray(x_dim, dtype=int)))
    all_x = np.empty(all_x_shape, dtype=np.float32)

    # count total number of simulations
    n_accepted = 0

    # loop over simulations # Todo: should we use jit here for speed-up?
    while n_accepted < num_samples:

        # make sure batch size doesn't exceed number of total simulations
        n_batch = min(simulation_batch_size, num_samples - n_accepted, )

        # run simulator
        # if case is needed due to different behavior of simulation_batch_size == 1 and
        # simulation_batch_size > 1. See docstring.
        if simulation_batch_size > 1:
            x = simulator(parameters[n_accepted:n_accepted + n_batch])
        else:
            x = simulator(parameters[n_accepted])[None, ]

        # add simulations to database of simulations. If simulator output x is torch.Tensor,
        # it automatically gets converted to a np.array here.
        all_x = np.concatenate((all_x, x), axis=0)

        n_accepted += simulation_batch_size

    # if simulation_batch_size >= num_samples, all_x might be a tensor already.
    # In all other cases, it is a np.array, so we convert it to a tensor here.
    if not isinstance(all_x, torch.Tensor):
        all_x = torch.from_numpy(all_x)

    return torch.tensor(parameters), all_x


def get_simulator_prior_and_groundtruth(task):

    if task == "nonlinear-gaussian":
        simulator = simulators.NonlinearGaussianSimulator()
        prior = BoxUniform(
            low=-3 * torch.ones(simulator.parameter_dim),
            high=3 * torch.ones(simulator.parameter_dim),
        )
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

    elif task == "nonlinear-gaussian-gaussian":
        simulator = simulators.NonlinearGaussianSimulator()
        prior = MultivariateNormal(loc=torch.zeros(5), covariance_matrix=torch.eye(5))
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

    elif task == "two-moons":
        simulator = simulators.TwoMoonsSimulator()
        a = 2
        prior = BoxUniform(
            low=-a * torch.ones(simulator.parameter_dim),
            high=a * torch.ones(simulator.parameter_dim),
        )
        # dummy ground truth parameters as none are specified in 'Automatic Posterior Transformation.'
        ground_truth_parameters = torch.zeros((1, 2))
        ground_truth_observation = torch.zeros((1, 2))

    elif task == "linear-gaussian":
        dim, std = 20, 0.5
        simulator = lambda theta: linear_gaussian(theta, std=std)
        prior = MultivariateNormal(
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
        ground_truth_parameters = torch.log(torch.tensor([0.01, 0.5, 1.0, 0.01]))
        path = os.path.join(utils.get_data_root(), "lotka-volterra", "obs_stats.pkl")
        with open(path, "rb") as file:
            true_observation = pickle.load(file, encoding="bytes")
        ground_truth_observation = torch.tensor(true_observation)

    elif task == "lotka-volterra-gaussian":
        simulator = simulators.LotkaVolterraSimulator(
            summarize_observations=True, gaussian_prior=True
        )
        prior = MultivariateNormal(
            loc=torch.zeros(4), covariance_matrix=2 * torch.eye(4)
        )
        ground_truth_parameters = torch.log(torch.tensor([0.01, 0.5, 1.0, 0.01]))
        path = os.path.join(utils.get_data_root(), "lotka-volterra", "obs_stats.pkl")
        with open(path, "rb") as file:
            true_observation = pickle.load(file, encoding="bytes")
        ground_truth_observation = torch.tensor(true_observation)

    elif task == "mg1":
        simulator = simulators.MG1Simulator()
        prior = distributions_.MG1Uniform(
            low=torch.zeros(3), high=torch.tensor([10.0, 10.0, 1.0 / 3.0])
        )
        ground_truth_parameters = torch.tensor([1.0, 5.0, 0.2])
        path = os.path.join(utils.get_data_root(), "mg1", "observed_data.pkl")
        with open(path, "rb") as file:
            _, true_observation = pickle.load(file, encoding="bytes")
        ground_truth_observation = torch.tensor(true_observation)

    else:
        raise ValueError(f"'{task}' simulator choice not understood.")

    return simulator, prior, ground_truth_parameters, ground_truth_observation
