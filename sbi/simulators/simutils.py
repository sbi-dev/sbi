import math
import os
import pickle
import warnings
from typing import Callable, Tuple, Union
import numpy as np
from scipy.stats._multivariate import multi_rv_frozen
from scipy.stats._distn_infrastructure import rv_frozen

import torch
from pyknos.nflows import distributions as distributions_
from torch.distributions import Distribution, MultivariateNormal, Uniform

import sbi.simulators as simulators
import sbi.utils as utils
from sbi.utils.torchutils import BoxUniform


def check_prior(prior: Callable) -> Tuple[Distribution, int, bool]:
    """Check prior object provided by the user. Return pytorch distribution like prior
    object.

    Arguments:
        prior -- prior object provided by the user.
    
    Raises:
        AttributeError: If prior objects lacks .sample or .log_prob
    
    Returns:
        prior -- Pytorch distribution prior. 
        parameter_dim -- event shape of the prior, number of parameters.
        prior_returns_numpy -- whether the original prior return type is numpy array.
    """

    if isinstance(prior, Distribution):
        return check_pytorch_distribution_prior(prior)

    # XXX: check for numpy random distribution object explicitly? How?

    # check for scipy stats distribution objects
    if isinstance(prior, (rv_frozen, multi_rv_frozen)):
        # wrap to pytorch distribution and check
        prior = ScipyToPytorchWrapper(prior)
        return check_pytorch_distribution_prior(prior)

    # check for sample and log_prob methods
    n_batch = 2
    try:
        theta = prior.sample((n_batch,))
        log_probs = prior.log_prob(theta)

    except AttributeError:
        raise AttributeError(
            "Prior needs methods .sample() and .log_prob(). Consider using a pytorch Distribution"
        )

    # check for batch shape
    assert (
        theta.shape[0] == n_batch
    ), "prior.sample((batch_size, )) does not return batch_size parameters."

    assert (
        log_probs.shape[0] == n_batch
    ), "prior.log_prob doesnt return the input batch size."

    # check numpy or pytorch
    assert isinstance(theta, torch.Tensor) or isinstance(
        theta, np.ndarray
    ), f"prior must return torch.Tensor or np.ndarray, but is {type(theta)}"

    # wrap prior into torch distribution if it returns numpy
    if isinstance(theta, np.ndarray):
        prior = CustomToPytorchWrapper(
            prior_numpy=prior, event_shape=torch.Size([prior.sample().size])
        )
        prior_returns_numpy = True
    else:
        prior_returns_numpy = False

    # infer parameter dimension
    parameter_dim = prior.sample().numel()

    return prior, parameter_dim, prior_returns_numpy


def check_pytorch_distribution_prior(
    prior: Distribution,
) -> Tuple[Distribution, int, bool]:
    """Check validity of pytorch prior, return corrected prior.
    
    Arguments:
        prior  -- pytorch distribution prior provided by the user.
    
    Raises:
        ValueError: If prior is defined over a scalar variable.
    
    Returns:
        prior -- Pytorch distribution prior. 
        parameter_dim -- event shape of the prior, number of parameters.
        prior_returns_numpy -- False.
    """

    # batch shape must not be multidimensional
    assert (
        torch.as_tensor(prior.batch_shape).ndim == 1
    ), "prior batch_shape has to one dimensional"

    # reject scalar priors
    if prior.sample().ndim == 0:
        raise ValueError(
            "Detected scalar prior. Please make sure to pass a pytorch prior with batch_shape=torch.Size([1])"
        )

    # assert batch shape.
    # NOTE: batch shape can be empty, as long as event shape is not.
    assert prior.batch_shape == torch.Size([1]) or prior.batch_shape == torch.Size(
        []
    ), "The prior has batch_shape>1. Please define it with batch_shape=1."

    # infer parameter dimension
    parameter_dim = prior.sample().numel()

    # final prior batch behavior check
    assert prior.sample((2,)).shape == torch.Size(
        [2, parameter_dim]
    ), "pytorch prior batch behavior is wrong."

    # check batch shape vs event shape confusion in pytorch Uniform
    # XXX: use Alvaros more in depth check function here
    if isinstance(prior, Uniform) and prior.sample().numel() > 1:
        warnings.warn(
            f"The paramerer dimension (`event_shape`) of the simualtor inferred from the "
            "prior is D={dim_input}>1 and the prior PyTorch Uniform. Therefore, beware "
            "that you are using a `batch_shape` of {dim_input} implicitly and "
            "`event_shape` 1, because Pytorch does not support multivariate Uniform. "
            "Consider using a BoxUniform prior instead."
        )
    return prior, parameter_dim, False


class CustomToPytorchWrapper(Distribution):
    """
    Wrap custom prior object with .sample and .log_prob methods to pytorch 
    Distribution object.
    """

    def __init__(
        self,
        prior_numpy,
        batch_shape=torch.Size(),
        event_shape=torch.Size(),
        validate_args=None,
    ):
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

        self.prior_numpy = prior_numpy

    def log_prob(self, value):
        return torch.as_tensor(self.prior_numpy.log_prob(value), dtype=torch.float32)

    def sample(self, sample_shape=torch.Size()):
        return torch.as_tensor(
            self.prior_numpy.sample(sample_shape), dtype=torch.float32
        )


class ScipyToPytorchWrapper(Distribution):
    """
    Wrap scipy.stats prior to pytorch Distribution object.
    """

    def __init__(
        self,
        prior_scipy: Union[rv_frozen, multi_rv_frozen],
        batch_shape=torch.Size(),
        event_shape=torch.Size(),
        validate_args=None,
    ):
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

        self.prior_scipy = prior_scipy

    def log_prob(self, value):
        return torch.as_tensor(self.prior_scipy.logpdf(x=value), dtype=torch.float32)

    def sample(self, sample_shape=torch.Size()):
        return torch.as_tensor(
            self.prior_scipy.rvs(size=sample_shape), dtype=torch.float32
        )


def check_simulator(user_simulator: Callable, prior, numpy_input: bool) -> Callable:

    assert isinstance(user_simulator, Callable), "simulator needs to be a function"

    # if simulator takes numpy input wrap to torch
    if numpy_input:
        # first assert that it input is consistent with output
        theta = prior.sample().numpy()
        data = user_simulator(theta)
        assert isinstance(
            data, np.ndarray
        ), f"Simulator output type {type(data)} should match simulator input type "
        f"{type(theta)}"

        # then define wrapped simulator
        def pytorch_simulator(theta: torch.Tensor):
            return torch.as_tensor(user_simulator(theta.numpy()), dtype=torch.float32)

        warnings.warn("Simulator cant handle torch input. It will be wrapped.")
    else:
        pytorch_simulator = user_simulator

    # check batch shape handling
    can_handle_batch = True
    try:
        n_batch = 2
        data = pytorch_simulator(prior.sample((n_batch,)))
        assert data.shape[0] == n_batch
    except:
        can_handle_batch = False
        warnings.warn(
            "Simulator can't handle batches of parameters. It will be wrapped using 'map', "
            "which can be inefficient."
        )

    if not can_handle_batch:

        def batch_simulator(thetas: torch.Tensor):
            # use map to get data for every theta in batch
            # use stack to collect list of tensors in tensor
            assert (
                thetas.ndim > 1
            ), f"batch simulator needs batch dimension. shape: {thetas.shape}"
            return torch.stack(list(map(pytorch_simulator, thetas)))

    else:
        batch_simulator = pytorch_simulator

    return batch_simulator


def check_observed_data(observed_data, simulator, prior):

    # cast to tensor
    observed_data = torch.as_tensor(observed_data, dtype=torch.float32)

    assert observed_data.ndim > 1, (
        f"observed data needs at least 2 dimensions, batch dim and data dim, e.g., "
        f"(1, data_dim). has only {observed_data.ndim}"
    )

    # sample from original prior and simulate, remove batch dimension
    # cast to tensor for comparison
    simulated_data = torch.as_tensor(
        simulator(prior.sample()), dtype=torch.float32
    ).squeeze(0)

    # match simulator output and observed data
    assert observed_data.shape[1:] == simulated_data.shape, (
        f"Observed data shape ({observed_data.shape[1:]}) must match "
        f"simulator output shape ({simulated_data.shape})."
    )

    if observed_data.shape[0] > 1:
        raise NotImplementedError(
            "It seems you passed a batch of observed data. SBI does not support "
            " multiple observations yet."
        )

    # check dimensionality
    if observed_data.ndim > 2:
        raise NotImplementedError(
            "The observed data has more than one dimension, e.g., it is a matrix. "
            "This is not implemented in SBI yet"
        )

    observation_dim = observed_data[0, :].numel()

    return observed_data, observation_dim


def check_user_input(
    user_simulator: Callable, user_prior, user_observed_data
) -> Tuple[Callable, Callable, torch.Tensor]:

    # check prior, return pytorch prior
    prior, parameter_dim, prior_returns_numpy = check_prior(user_prior)

    # check data, returns batched 1D data
    observed_data, observation_dim = check_observed_data(
        user_observed_data, user_simulator, user_prior
    )

    # check simulator, returns pytorch batch simulator
    simulator = check_simulator(user_simulator, prior, prior_returns_numpy)

    # set function attributes for logging
    simulator = set_simulator_attributes(simulator, prior, observed_data)

    return simulator, prior, observed_data


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

    setattr(simulator_fun, "parameter_dim", parameter_dim)
    setattr(simulator_fun, "observation_dim", observation_dim)

    return simulator_fun


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


def simulate_in_batches(
    simulator: Callable,
    parameter_sample_fn: Callable,
    num_samples: int,
    simulation_batch_size: int,
    x_dim: torch.Size,
) -> (torch.Tensor, torch.Tensor):
    """
    Draw `num_samples` parameter sets and simulate them in batches of size `simulation_batch_size`.

    Features: - allows to simulate in batches of arbitrary size
              - if `simulation_batch_size==-1`, all simulations are run at the same time
              - simulator output can be np.array or torch.Tensor

    Args:
        simulator: simulator function.
            If `simulation_batch_size == 1`: takes in parameters of shape (1, num_dim_parameters)
                and outputs xs of shape (1, num_dim_x)
            If `simulation_batch_size > 1`: takes in thetas of shape (simulation_batch_size, num_dim_parameters)
                and outputs xs of shape (simulation_batch_size, num_dim_x)
        parameter_sample_fn: Function to call for generating theta, e.g. prior sampling
        num_samples: Number of simulations to run
        simulation_batch_size: Number of simulations that are run within a single batch
            If `simulation_batch_size == -1`, we run a batch with all simulations required,
            i.e. `simulation_batch_size = num_samples`
        x_dim: dimensionality of a single simulator output

    Returns: torch.Tensor simulation input parameters of shape (num_samples, num_dim_parameters),
             torch.Tensor simulator outputs x of shape (num_samples, num_dim_x)
    """

    # generate parameters (simulation inputs) by sampling from prior (round 1) or proposal (round > 1)
    parameters = parameter_sample_fn(num_samples)

    if simulation_batch_size == -1:
        # run all simulations in a single batch
        simulation_batch_size = num_samples

    # split parameter set into batches of size (simulation_batch_size, num_dim_parameters)
    n_chunks = math.ceil(num_samples / simulation_batch_size)
    parameter_batches = torch.chunk(parameters, chunks=n_chunks)

    all_x = []
    for batch in parameter_batches:
        with torch.no_grad():
            # XXX: if we assert the that simultor return Tensor with batch dim we can avoid the following 2 checks
            x = simulator(batch)
            if not isinstance(x, torch.Tensor):
                # convert simulator output to torch in case it was numpy array
                x = torch.from_numpy(x)
            if simulation_batch_size == 1:
                # squeeze in case simulator provides an additional dimension for single parameters,
                # e.g. in linearGaussian example. Then prepend a dimension to be able to concatenate.
                # XXX: here we are squeezing and unsqueezing on the same dim, no?
                x = torch.squeeze(x, dim=0).unsqueeze(0)
        # collect batches in list
        all_x.append(x)

    return torch.tensor(parameters), torch.cat(all_x)
