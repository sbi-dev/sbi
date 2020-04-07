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
from torch import Tensor

import sbi.simulators as simulators
import sbi.utils as utils
from sbi.utils.torchutils import BoxUniform


def process_prior(prior: Callable) -> Tuple[Distribution, int, bool]:
    """Check prior object provided by the user. Return pytorch distribution like prior
    object.

    Args:
        prior: prior object provided by the user.
    
    Raises:
        AttributeError: If prior objects lacks .sample or .log_prob
    
    Returns:
        prior: Pytorch distribution prior. 
        parameter_dim: event shape of the prior, number of parameters.
        prior_returns_numpy: whether the original prior return type is numpy array.
    """

    if isinstance(prior, Distribution):
        return process_pytorch_prior(prior)

    # If prior is given as scipy.stats object, wrap to pytorch.
    elif isinstance(prior, (rv_frozen, multi_rv_frozen)):
        prior = ScipyToPytorchWrapper(prior)
        return process_pytorch_prior(prior)

    # Otherwise it is a custom prior - check for .sample and .log_prob methods.
    else:
        return process_custom_prior(prior)


def process_custom_prior(prior) -> Tuple[Distribution, int, bool]:
    """Check and correct prior object defined by the user."""

    check_prior_methods(prior)

    check_prior_batch_behavior(prior)

    prior, prior_returns_numpy = maybe_wrap_prior_to_pytorch(prior)

    parameter_dim = prior.sample().numel()

    return prior, parameter_dim, prior_returns_numpy


def maybe_wrap_prior_to_pytorch(prior) -> Tuple[Distribution, bool]:
    """Check prior return type and maybe wrap to pytorch.
    
    Args:
        prior: prior object with .sample and .log_prob methods.
    
    Raises:
        TypeError: if prior return type is pytorch or numpy.
    
    Returns:
        prior: prior returning only pytorch tensor. 
        prior_returns_numpy: whether the prior returned numpy before wrapping.
    """

    # Get samples, log probs and event dimension.
    num_samples = 2
    theta = prior.sample((num_samples,))
    log_probs = prior.log_prob(theta)
    _, prior_event_dim = theta.shape

    # Check return types
    if isinstance(theta, Tensor) and isinstance(log_probs, Tensor):
        prior_returns_numpy = False
    elif isinstance(theta, np.ndarray) and isinstance(log_probs, np.ndarray):
        # wrap prior into torch distribution if it returns numpy
        prior = CustomToPytorchWrapper(
            prior_numpy=prior, event_shape=torch.Size([prior_event_dim])
        )
        prior_returns_numpy = True
    else:
        raise TypeError(
            "Prior must return torch.Tensor or np.ndarray, but is {type(theta)}"
        )

    return prior, prior_returns_numpy


def process_pytorch_prior(prior: Distribution,) -> Tuple[Distribution, int, bool]:
    """Check validity of pytorch prior, return corrected prior.
    
    Args:
        prior: pytorch distribution prior provided by the user.
    
    Raises:
        ValueError: If prior is defined over a scalar variable.
    
    Returns:
        prior: Pytorch distribution prior. 
        parameter_dim: event shape of the prior, number of parameters.
        prior_returns_numpy: False.
    """

    # reject scalar priors
    if prior.sample().ndim == 0:
        raise ValueError(
            "Detected scalar prior. Please make sure to pass a pytorch prior with "
            "batch_shape=torch.Size([1]), or event_shape=torch.Size([1])"
        )

    assert prior.batch_shape in (
        torch.Size([1]),
        torch.Size([]),
    ), f"The prior must have batch shape torch.Size([]) or torch.Size([1]), but has {prior.batch_shape}."

    check_prior_batch_behavior(prior)

    check_for_batch_reinterpretation_extra_d_uniform(prior)

    parameter_dim = prior.sample().numel()

    return prior, parameter_dim, False


def check_for_batch_reinterpretation_extra_d_uniform(prior):
    """Warn when prior is a batch of scalar Uniforms.

    Most likely the user wants to specify a prior on a multi-dimensional parameter
    rather than several 1D priors at once.
    """

    # Note .batch_shape will always work because and is short-circuiting.
    if isinstance(prior, Uniform) and prior.batch_shape.numel() > 1:
        raise ValueError(
            f"""The specified Uniform prior is a prior on *several scalar parameters*
            (i.e. a batch), not a prior on a multi-dimensional parameter.

            Please use utils.torchutils.BoxUniform if you'd rather put a prior on a 
            multi-dimensional parameter.
            """
        )


def check_for_possibly_batched_observations(true_observation: Tensor):
    """Raise ValueError if dimensionality of data doesn't match requirements.
    
    The current requirements are that is 2D with a leading batch dimension of size 1.
    That is, neither multiple, batched observations are supported, 
    nor a multidimensional observation e.g., an image.
    """

    if true_observation.squeeze().ndim > 1:
        raise ValueError(
            """`true_observation` has D>1 dimensions. SBI interprets the leading 
                dimension as a batch dimension, but it *currently* only processes 
                a single observation, i.e. the first element of the batch.
                
                For example:
                
                > true_observation = [ [1,2,3], [4,5,6] ] 
                
                is interpreted as two vector observations, only the first of which 
                is currently used to condition inference.
                
                Use rather:
                
                > true_observation = [ [[1,2,3], [4,5,6]] ]
                > true_observation = [ [1], [2], [3]]
                
                if your single observation is matrix-shaped or scalar-shaped.
                
                Finally:
                
                > true_observation = [1]
                > true_observation = [1, 2, 3]
                
                will be interpreted as one scalar observation or one vector observation
                and don't require wrapping (unsqueezing).
                """
        )


def check_prior_methods(prior):
    """Check whether the prior has methods .sample and .log_prob. 
    
    Raises: 
        AttributionError: if either of the two methods doesnt works as expected. 
    """

    # Sample a batch of two parameters to check batch behaviour > 1.
    num_samples = 2
    # Check .sample and log_prob, keep for later use.
    try:
        theta = prior.sample((num_samples,))
    except AttributeError:
        raise AttributeError(
            "Prior needs method .sample((num_samples, )). Consider using a "
            "pytorch Distribution"
        )
    try:
        log_probs = prior.log_prob(theta)
    except AttributeError:
        raise AttributeError(
            "Prior needs method .log_prob(values). Consider using a pytorch Distribution"
        )


def check_prior_batch_behavior(prior):
    """Assert that it is possible to sample and evaluate batches of parameters."""

    # Check for correct batch size in .sample and .log_prob
    num_samples = 2
    theta = prior.sample((num_samples,))
    log_probs = prior.log_prob(theta)

    assert (
        len(theta.shape) == 2
    ), f"""A parameter batch sampled from the prior must be 2D, 
    (num_samples, parameter_dim), but is {len(theta.shape)}"""

    num_sampled, parameter_dim = theta.shape
    num_log_probs = log_probs.shape[0]

    assert (
        num_sampled == num_samples
    ), "prior.sample((batch_size, )) does not return batch_size parameters."

    assert (
        num_log_probs == num_samples
    ), "prior.log_prob doesnt return the input batch size."


class CustomToPytorchWrapper(Distribution):
    """Wrap custom prior object with .sample and .log_prob methods to pytorch 
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
    """Wrap scipy.stats prior to pytorch Distribution object."""

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


def process_simulator(
    user_simulator: Callable, prior, is_numpy_simulator: bool
) -> Callable:
    """Check requirements and return a pytorch batch simulator.
    
    That is, a simulator that received as input and returns a Tensor, 
    and that can simulate batches of parameters.
    """

    assert isinstance(user_simulator, Callable), "Simulator must be a function."

    pytorch_simulator = wrap_as_pytorch_simulator(
        user_simulator, prior, is_numpy_simulator
    )

    batch_simulator = wrap_as_batch_simulator(pytorch_simulator, prior)

    return batch_simulator


def wrap_as_pytorch_simulator(
    simulator: Callable, prior, is_numpy_simulator
) -> Callable:
    """Return a simulator that receives as input and returns a Tensor."""

    if is_numpy_simulator:
        # Get data to check input type is consistent with data
        theta = prior.sample().numpy()  # cast to numpy because prior is fixed already
        data = simulator(theta)
        assert isinstance(
            data, np.ndarray
        ), f"Simulator output type {type(data)} should match simulator input type "
        f"{type(theta)}"

        # Define a wrapper function to pytorch
        def pytorch_simulator(theta: Tensor):
            return torch.as_tensor(simulator(theta.numpy()), dtype=torch.float32)

        warnings.warn("Simulator cant handle torch input. It will be wrapped.")
    else:
        pytorch_simulator = simulator

    return pytorch_simulator


def wrap_as_batch_simulator(simualtor: Callable, prior) -> Callable:
    """Return a batch simulator. 
    
    A batch simulator can handle a batch of parameters and return the 
    corresponding batch of simulated data.
    """

    is_batch_simulator = True
    try:
        n_batch = 2
        data = simualtor(prior.sample((n_batch,)))
        assert data.shape[0] == n_batch
    except:
        warnings.warn(
            "Simulator can't handle batches of parameters. It will be wrapped using 'map', "
            "which can be inefficient."
        )
        is_batch_simulator = False

        def batch_simulator(thetas: Tensor):
            # use map to get data for every theta in batch
            # use stack to collect list of tensors in tensor
            assert (
                thetas.ndim > 1
            ), f"batch simulator needs batch dimension. shape: {thetas.shape}"
            return torch.stack(list(map(simualtor, thetas)))

    return simualtor if is_batch_simulator else batch_simulator


def process_observed_data(
    observed_data: Union[Tensor, np.ndarray], simulator: Callable, prior
) -> Tuple[Tensor, int]:
    """Check and correct for requirements on the observed data object.
    
    Args:
        observed_data: observed data as provided by the user.
        simulator: simulator function. 
        prior: prior object.
    
    Returns:
        observed data: corrected observed data.
        observation_dim: number of elements in a single data point.
    """

    # Cast to Tensor because data could be in numpy.
    observed_data = torch.as_tensor(observed_data, dtype=torch.float32)

    check_for_possibly_batched_observations(observed_data)

    # Get unbatched simulated data by sampling from prior and simualtor.
    # cast to tensor for comparison
    simulated_data = torch.as_tensor(
        simulator(prior.sample()), dtype=torch.float32
    ).squeeze(0)

    # Get data shape by ommitting the batch dimension.
    observed_data_shape = observed_data.shape[1:]

    assert observed_data_shape == simulated_data.shape, (
        f"Observed data shape ({observed_data.shape[1:]}) must match "
        f"simulator output shape ({simulated_data.shape})."
    )

    observation_dim = observed_data[0, :].numel()

    return observed_data, observation_dim


def prepare_sbi_problem(
    user_simulator: Callable, user_prior, user_observed_data: Union[Tensor, np.ndarray]
) -> Tuple[Callable, Callable, Tensor]:
    """Prepare simulator, prior and observed data for usage in sbi. 

    The following requirements are tried to be met by wrapping or reshaping and casting
    to pytorch: 
        - the simulator function receives as input and returns a Tensor.
        - the simulator can simulate batches of parameters and return batches of data.
        - the prior behaves like a pytorch.distributions.Distribution.
        - the observed data is a Tensor and has a leading batch dimension of one. 
    
    If this is not possible assertion erros or corresponding erros are raised.

    Args:
        user_simulator: simulator as provided by the user
        user_prior: prior as provided by the user
        user_observed_data: observed data as provided by the user

    Returns:
        simulator: corrected simulator ready to be used in sbi. 
        prior: corrected prior. 
        observed_data: corrected observed data.
    """

    # check prior, return pytorch prior
    prior, parameter_dim, prior_returns_numpy = process_prior(user_prior)

    # check data, returns batched 1D data
    observed_data, observation_dim = process_observed_data(
        user_observed_data, user_simulator, user_prior
    )

    # check simulator, returns pytorch batch simulator
    simulator = process_simulator(user_simulator, prior, prior_returns_numpy)

    # final check
    check_sbi_problem(simulator, prior, observed_data)

    return simulator, prior, observed_data


def check_sbi_problem(simulator: Callable, prior, observed_data: Tensor):
    """Assert requirements for simulator, prior and x0 for usage in sbi. 
    
    Args:
        simulator: simulator function
        prior: prior (Distribution like)
        observed_data: observed data
    """
    num_samples = 2
    theta = prior.sample((num_samples,))
    data = simulator(theta)
    assert isinstance(theta, Tensor), "theta must be a Tensor."
    assert isinstance(data, Tensor), "simulator output must be a Tensor."
    assert isinstance(observed_data, Tensor), "observed data must be a Tensor."
    assert (
        theta.shape[0] == num_samples
    ), f"Theta batch shape {theta.shape[0]} must match num_samples={num_samples}."
    assert (
        data.shape[0] == num_samples
    ), f"Data batch shape {data.shape[0]} must match num_samples={num_samples}."
    assert (
        observed_data.shape[1:] == data[0, :].shape
    ), f"Observed data shape must match simulated data shape."


def set_simulator_attributes(
    simulator_fun: Callable, prior: Distribution, observed_data: Tensor, name=None
) -> Callable:
    """Add name and input and output dimension as attributes to the simulator function.
    
    Arguments:
        simulator_fun {Callable} -- simulator function taking parameters as input
        prior {torch.distributions.Distribution} -- prior as pytorch distributions object
        observed_data {Tensor} -- Observed data points, x0
    
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
    prior: Distribution, observed_data: Tensor
) -> Tuple[int, int]:
    """Return simulator input output dimension from prior and observed data. 
    
    Arguments:
        prior {Distribution} -- pytorch prior distribution with event and batch shapes
        observed_data {Tensor} -- Observed data point, x0
    
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
) -> (Tensor, Tensor):
    """
    Draw `num_samples` parameter sets and simulate them in batches 
    of size `simulation_batch_size`.

    Features: 
        Allows to simulate in batches of arbitrary size.
        If `simulation_batch_size==-1`, all simulations are run at the same time.

    Args:
        simulator: simulator function.
        parameter_sample_fn: Function to call for generating theta, e.g. prior sampling
        num_samples: Number of simulations to run
        simulation_batch_size: Number of simulations that are run within a single batch
            If `simulation_batch_size == -1`, we run a batch with all simulations required,
            i.e. `simulation_batch_size = num_samples`
        x_dim: dimensionality of a single simulator output

    Returns: Tensor simulation input parameters of shape (num_samples, num_dim_parameters),
             Tensor simulator outputs x of shape (num_samples, num_dim_x)
    """

    # generate parameters (simulation inputs) by sampling from prior
    # (round 1) or proposal (round > 1)
    parameters = parameter_sample_fn(num_samples)

    if simulation_batch_size == -1:
        # run all simulations in a single batch
        simulation_batch_size = num_samples

    # split parameter set into batches of size (simulation_batch_size, num_dim_parameters)
    n_chunks = math.ceil(num_samples / simulation_batch_size)
    parameter_batches = torch.chunk(parameters, chunks=n_chunks)

    xs = []
    for batch in parameter_batches:
        with torch.no_grad():
            xs.append(simulator(batch))

    return torch.tensor(parameters), torch.cat(xs)
