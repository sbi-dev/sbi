from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple, Union

from numpy import ndarray
import torch
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen
from torch import Tensor, float32
from torch.distributions import Distribution, Uniform, Beta, Independent, Gamma
from sbi.simulators.user_input_checks_utils import (
    CustomPytorchWrapper,
    ScipyPytorchWrapper,
    PytorchReturnTypeWrapper,
    IndependentJoint,
)

from sbi.utils.torchutils import atleast_2d, ensure_theta_batched


def process_prior(prior) -> Tuple[Distribution, int, bool]:
    """Return PyTorch distribution-like prior adapted from user-provided prior.

    Args:
        prior: prior object provided by the user.

    Raises:
        AttributeError: If prior objects lacks .sample() or .log_prob()

    Returns:
        prior: prior that emits samples and log probabilities as PyTorch tensors.
        theta_dim: event shape of the prior, number of parameters.
        prior_returns_numpy: whether the return type of the prior was a numpy array.
    """

    if isinstance(prior, Distribution):
        return process_pytorch_prior(prior)

    # If prior is given as scipy.stats object, wrap to PyTorch.
    elif isinstance(prior, (rv_frozen, multi_rv_frozen)):
        event_shape = torch.Size([prior.rvs().size])
        # batch_shape is passed as default
        prior = ScipyPytorchWrapper(
            prior, batch_shape=torch.Size([]), event_shape=event_shape
        )
        return process_pytorch_prior(prior)

    # Otherwise it is a custom prior - check for .sample and .log_prob methods.
    else:
        return process_custom_prior(prior)


def process_custom_prior(prior) -> Tuple[Distribution, int, bool]:
    """Check and return corrected prior object defined by the user.

    Args:
        prior: prior object with .sample and .log_prob methods.

    Returns:
        prior: corrected prior.
        theta_dim: event dimension of the prior, size of single parameter vector.
        is_prior_numpy: whether the prior returned numpy before wrapping.
    """

    check_prior_attributes(prior)

    check_prior_batch_behavior(prior)

    prior, is_prior_numpy = maybe_wrap_prior_to_pytorch(prior)

    check_prior_return_type(prior)

    theta_dim = prior.sample().numel()

    return prior, theta_dim, is_prior_numpy


def maybe_wrap_prior_to_pytorch(prior) -> Tuple[Distribution, bool]:
    """Check prior return type and maybe wrap to PyTorch.

    Args:
        prior: prior object with .sample and .log_prob methods.

    Raises:
        TypeError: if prior return type is PyTorch or numpy.

    Returns:
        prior: prior returning Tensor.
        is_prior_numpy: whether the prior returned numpy before wrapping.
    """

    theta = prior.sample((1,))
    log_probs = prior.log_prob(theta)

    # Check return types
    if isinstance(theta, Tensor) and isinstance(log_probs, Tensor):
        # XXX: We wrap to get a Distribution. But this might interfere with the fact
        # that the custom prior can be a probabilistic program.
        prior = CustomPytorchWrapper(
            custom_prior=prior, event_shape=torch.Size([theta.numel()])
        )
        is_prior_numpy = False
    elif isinstance(theta, ndarray) and isinstance(log_probs, ndarray):
        # infer event shape from single numpy sample.
        event_shape = torch.Size([theta.size])
        prior = CustomPytorchWrapper(custom_prior=prior, event_shape=event_shape)
        is_prior_numpy = True
    else:
        raise TypeError(
            "Prior must return torch.Tensor or ndarray, but returns {type(theta)}"
        )

    return prior, is_prior_numpy


def process_pytorch_prior(prior: Distribution,) -> Tuple[Distribution, int, bool]:
    """Return corrected prior after checking requirements for SBI.

    Args:
        prior: PyTorch distribution prior provided by the user.

    Raises:
        ValueError: If prior is defined over an unwrapped scalar variable.

    Returns:
        prior: PyTorch distribution prior.
        theta_dim: event shape of the prior, number of parameters.
        prior_returns_numpy: False.
    """

    # Reject unwrapped scalar priors.
    if prior.sample().ndim == 0:
        raise ValueError(
            "Detected scalar prior. Please make sure to pass a PyTorch prior with "
            "batch_shape=torch.Size([1]), or event_shape=torch.Size([1])"
        )

    check_prior_batch_behavior(prior)

    prior = maybe_reinterpret_batch_dims(prior)

    if not prior.sample().dtype == float32:
        prior = PytorchReturnTypeWrapper(prior, return_type=float32)

    # This will fail for float64 priors.
    check_prior_return_type(prior)

    theta_dim = prior.sample().numel()

    return prior, theta_dim, False


def maybe_reinterpret_batch_dims(prior) -> Distribution:
    """Return prior with batch shape smaller or equal 1. 

    Reinterprets all batch dimensions as event dimensions if possible, e.g., for
    Uniform, Beta, Gamma priors. 

    Raises: 
        ValueError: if batch_shape larger 1 and reinterpretation is not possible. 

    Returns:
        prior: prior with maybe reinterpreted batch dimensions.
    """

    reinterpretable_priors = (Uniform, Beta, Gamma)

    if prior.batch_shape.numel() <= 1:
        return prior

    elif isinstance(prior, reinterpretable_priors) and prior.batch_shape.numel() > 1:
        warnings.warn(
            f"""The specified prior is a prior on *several scalar parameters*
            (i.e., a batch), not a prior on a multi-dimensional parameter. 

            Because we recognised its type its batch dimensions will be reinterpreted
            as event dimensions using pytorch.distributions.Independent.
            """,
            UserWarning,
        )
        # Reinterpret all batch dimensions as event dimensions.
        return Independent(prior, reinterpreted_batch_ndims=len(prior.batch_shape))

    else:
        raise ValueError(
            f"""The specified prior is a prior on *several scalar parameters*
            (i.e. a batch), not a prior on a multi-dimensional parameter."""
        )


def check_for_possibly_batched_observations(x_o: Tensor):
    """Raise ValueError if dimensionality of data doesn't match requirements.

    SBI does not support multiple observations yet. For 2D observed data the leading
    dimension will be interpreted as batch dimension and a ValueError is raised if the
    batch dimension is larger than 1.
    Multidimensional observations e.g., images, are allowed when they are passed with an
    additional leading batch dimension of size 1.
    """

    # Interpret first dimension as batch dimension.
    inferred_batch_shape, *inferred_data_shape = x_o.shape
    inferred_data_dim = len(inferred_data_shape)

    # Reject multidimensional data with batch_shape > 1.
    if x_o.ndim > 1 and inferred_batch_shape > 1:
        raise ValueError(
            """observed data `x_o` has D>1 dimensions. SBI interprets the leading
                dimension as a batch dimension, but it *currently* only processes
                a single observation, a batch of several observation is not supported
                yet.

                NOTE: below we use list notation to reduce clutter, but observation
                should be of type torch.Tensor or ndarray.

                For example:

                > x_o = [[1]]
                > x_o = [[1, 2, 3]]

                are interpreted as single observation with a leading batch dimension of
                one. However

                > x_o = [ [1], [2] ]
                > x_o = [ [1,2,3], [4,5,6] ]

                are interpreted as a batch of two scalar or vector observations, which
                is not supported yet. The following is interpreted as a matrix-shaped
                observation, i.e a monochromatic image:

                > x_o = [ [[1,2,3], [4,5,6]] ]

                Finally, for convenience,

                > x_o = [1]
                > x_o = [1, 2, 3]

                will be interpreted as a single scalar or single vector observation
                respectively, without the user needing to wrap or unsqueeze them.
                """
        )
    # Warn on multidimensional data with batch_size one.
    elif inferred_data_dim > 1 and inferred_batch_shape == 1:
        warnings.warn(
            f"""Beware: The observed data (x_o) you passed was interpreted to have
            matrix shape: {inferred_data_shape}. The current implementation of SBI
            might not provide stable support for this and result in shape mismatches.
            """
        )
    else:
        pass


def check_prior_attributes(prior):
    """Check for prior attributes needed in sbi.
    
    We check for .sample(sample_shape) .log_prob(value) methods.

    Raises:
        AttributeError: if either of the two methods doesn't exist.
    """

    # Sample a batch of two parameters to check batch behaviour > 1.
    # and to check that .sample can handle a tuple argument.
    num_samples = 2
    try:
        theta = prior.sample((num_samples,))
    except AttributeError:
        raise AttributeError(
            "Prior needs method .sample(). Consider using a PyTorch distribution."
        )
    except TypeError:
        raise TypeError(
            f"""The prior.sample() method must accept Tuple arguments, e.g.,
            prior.sample(({num_samples}, )) to sample a batch of 2 parameters. Consider
            using a PyTorch distribution."""
        )
    except:  # Catch any other error.
        raise ValueError(
            f"""Something went wrong when sampling a batch of parameters
            from the prior as prior.sample(({num_samples}, )). Consider using a PyTorch
            distribution."""
        )
    try:
        prior.log_prob(theta)
    except AttributeError:
        raise AttributeError(
            "Prior needs method .log_prob(). Consider using a PyTorch distribution."
        )
    except:  # Catch any other error.
        raise ValueError(
            """Something went wrong when evaluating a batch of parameters theta
            with prior.log_prob(theta). Consider using a PyTorch distribution."""
        )


def check_prior_return_type(prior, return_type: Optional[torch.dtype] = float32):
    """Check whether prior.sample() returns float32 Tensor."""

    prior_dtype = prior.sample().dtype
    assert (
        prior_dtype == return_type
    ), f"Prior return type must be {return_type}, but is {prior_dtype}."


def check_prior_batch_behavior(prior):
    """Assert that it is possible to sample and evaluate batches of parameters."""

    # Check for correct batch size in .sample and .log_prob
    num_samples = 1
    theta = prior.sample((num_samples,))
    log_probs = prior.log_prob(theta)

    assert (
        len(theta.shape) >= 2
    ), f"""A parameter batch sampled from the prior must be at least 2D,
    (num_samples, parameter_dim), but is {len(theta.shape)}"""

    num_sampled, parameter_dim = theta.shape
    # Using len here because log_prob could be ndarray or torch.Tensor
    num_log_probs = len(log_probs)

    assert (
        num_sampled == num_samples
    ), "prior.sample((batch_size, )) must return batch_size parameters."

    assert (
        num_log_probs == num_samples
    ), "prior.log_prob must return as many log probs as samples."


def process_simulator(
    user_simulator: Callable, prior, is_numpy_simulator: bool
) -> Callable:
    """Return a simulator that meets the requirements for usage in SBI.

    Wraps the simulator to return only torch.Tensor and handle batches of parameters.
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
    """Return a simulator that accepts and returns Tensor arguments."""

    if is_numpy_simulator:
        # Get data to check input type is consistent with data.
        theta = prior.sample().numpy()  # cast to numpy because is in PyTorch already.
        x = simulator(theta)
        assert isinstance(
            x, ndarray
        ), f"Simulator output type {type(x)} must match its input type."
        f"{type(theta)}"

        # Define a wrapper function to PyTorch
        def pytorch_simulator(theta: Tensor):
            return torch.as_tensor(simulator(theta.numpy()), dtype=float32)

    else:
        pytorch_simulator = simulator

    return pytorch_simulator


def wrap_as_batch_simulator(simulator: Callable, prior) -> Callable:
    """Return a batch simulator.

    A batch simulator can handle a batch of parameters and return the
    corresponding batch of simulated data.
    """

    is_batched_simulator = True
    try:
        batch_size = 2
        simulator_batch_size, *_ = simulator(prior.sample((batch_size,))).shape

        assert simulator_batch_size == batch_size
    except:
        is_batched_simulator = False
        warnings.warn(
            """Simulator can't handle batches of parameters. It will loop over batches
            in Python with `map`. Consider vectorising the simulator natively for
            performance."""
        )
        # make sure the simulator does not return batch dim.
        single_simulation = simulator(prior.sample())
        assert (
            not single_simulation.shape[0] == 1
        ), f"""The simulator can't handle batches and returns a singleton batch
            dimension: {single_simulation.shape}. A simulator that can't handle batches
            of parameters must not return a batch dim."""

    return simulator if is_batched_simulator else get_batched_simulator(simulator)


def get_batched_simulator(simulator: Callable) -> Callable:
    """Return simulator wrapped with `map` to handle batches of parameters."""

    # XXX: this should be handled with more care, e.g., enable multiprocessing
    # XXX: with Pool() as p: p.map(...)
    def batched_simulator(theta: Tensor) -> Tensor:
        theta = ensure_theta_batched(theta)
        # use map to get data for every theta in batch
        # use stack to collect list of tensors in tensor
        return torch.stack(list(map(simulator, theta)))

    return batched_simulator


def process_x_o(
    x_o: Union[Tensor, ndarray], simulator: Callable, prior
) -> Tuple[Tensor, int]:
    """Return observed data to sbi's shape and type requirements.

    Args:
        x_o: observed data as provided by the user.
        simulator: simulator function as provided by the user.
        prior: prior object with .sample() and .log_prob() methods.

    Returns:
        x_o: observed data with shape corrected for usage in SBI.
        x_o_dim: number of elements in a single data point.
    """

    # maybe add batch dimension, cast to tensor
    x_o = atleast_2d(x_o)
    x_o = torch.as_tensor(x_o, dtype=float32)

    check_for_possibly_batched_observations(x_o)

    # Get unbatched simulated data by sampling from prior and simulator.
    # cast to tensor for comparison
    simulated_data = torch.as_tensor(simulator(prior.sample()), dtype=float32).squeeze(
        0
    )

    # Get data shape by ommitting the batch dimension.
    x_o_shape = x_o.shape[1:]

    assert x_o_shape == simulated_data.shape, (
        f"Observed data shape ({x_o_shape}) must match "
        f"simulator output shape ({simulated_data.shape})."
    )

    x_o_dim = x_o[0, :].numel()

    return x_o, x_o_dim


def prepare_sbi_problem(
    user_simulator: Callable, user_prior, user_x_o: Union[Tensor, ndarray]
) -> Tuple[Callable, Callable, Tensor]:
    """Prepare simulator, prior and observed data for usage in sbi.

    Attempts to meet the following requirements by reshaping and type casting to PyTorch
    Tensor:

    - the simulator function receives as input and returns a Tensor.
    - the simulator can simulate batches of parameters and return batches of data.
    - the prior does not produce batches and samples and evaluates to Tensor.
    - the observed data is a Tensor and has a leading batch dimension of one.

    If this is not possible, a suitable exception will be raised.

    Args:
        user_simulator: simulator as provided by the user.
        user_prior: prior as provided by the user user_x_o: observed data as
            provided by the user

    Returns:
        simulator: simulator adapted for sbi.
        prior: adapted prior.
        x_o: adapted observed data.
    """

    # Check prior, return PyTorch prior.
    prior, _, prior_returns_numpy = process_prior(user_prior)

    # Check data, returns data with leading batch dimension.
    x_o, _ = process_x_o(user_x_o, user_simulator, user_prior)

    # Check simulator, returns PyTorch simulator able to simulate batches.
    simulator = process_simulator(user_simulator, prior, prior_returns_numpy)

    # Consistency check after making ready for SBI.
    check_sbi_problem(simulator, prior, x_o)

    return simulator, prior, x_o


def check_sbi_problem(simulator: Callable, prior, x_o: Tensor):
    """Assert requirements for simulator, prior and observation for usage in sbi.

    Args:
        simulator: simulator function
        prior: prior (Distribution like)
        x_o: observed data
    """
    num_prior_samples = 2
    theta = prior.sample((num_prior_samples,))
    theta_batch_shape, *_ = theta.shape
    simulation = simulator(theta)
    sim_batch_shape, *sim_event_shape = simulation.shape
    _, *obs_event_shape = x_o.shape

    assert isinstance(theta, Tensor), "Parameters theta must be a Tensor."
    assert isinstance(simulation, Tensor), "Simulator output must be a Tensor."
    assert isinstance(x_o, Tensor), "Observation must be a Tensor."

    assert (
        theta_batch_shape == num_prior_samples
    ), f"""Theta batch shape {theta_batch_shape} must match
        num_samples={num_prior_samples}."""
    assert (
        sim_batch_shape == num_prior_samples
    ), f"""Simulation batch shape {sim_batch_shape} must match
        num_samples={num_prior_samples}."""
    assert (
        obs_event_shape == sim_event_shape
    ), f"""The shape of a single x_o is {obs_event_shape} and it does not match
        that of a single simulation {sim_event_shape}. For a batch size of
        {num_prior_samples} the simulator returns {simulation.shape}
        (should be ({num_prior_samples}, {obs_event_shape})."""
