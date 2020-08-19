# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


import warnings
from typing import (
    Callable,
    Optional,
    Union,
    Tuple,
    cast,
    Sequence,
)

import torch
from numpy import ndarray
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen
from torch import Tensor, float32, nn
from torch.distributions import Distribution, Uniform

from sbi.user_input.user_input_checks_utils import (
    CustomPytorchWrapper,
    MultipleIndependent,
    PytorchReturnTypeWrapper,
    ScipyPytorchWrapper,
)
from sbi.utils.torchutils import BoxUniform, atleast_2d


def process_prior(prior) -> Tuple[Distribution, int, bool]:
    """Return PyTorch distribution-like prior from user-provided prior.

    Args:
        prior: Prior object with `.sample()` and `.log_prob()` as provided by the user.

    Raises:
        AttributeError: If prior objects lacks `.sample()` or `.log_prob()`.

    Returns:
        prior: Prior that emits samples and evaluates log prob as PyTorch Tensors.
        theta_numel: Number of parameters - elements in a single sample from the prior.
        prior_returns_numpy: Whether the return type of the prior was a Numpy array.
    """

    # If prior is a sequence, assume independent components and check as PyTorch prior.
    if isinstance(prior, Sequence):
        warnings.warn(
            f"""Prior was provided as a sequence of {len(prior)} priors. They will be
            interpreted as independent of each other and matched in order to the
            components of the parameter."""
        )
        return process_pytorch_prior(MultipleIndependent(prior))

    if isinstance(prior, Distribution):
        return process_pytorch_prior(prior)

    # If prior is given as `scipy.stats` object, wrap as PyTorch.
    elif isinstance(prior, (rv_frozen, multi_rv_frozen)):
        event_shape = torch.Size([prior.rvs().size])
        # batch_shape is passed as default
        prior = ScipyPytorchWrapper(
            prior, batch_shape=torch.Size([]), event_shape=event_shape
        )
        return process_pytorch_prior(prior)

    # Otherwise it is a custom prior - check for `.sample()` and `.log_prob()`.
    else:
        return process_custom_prior(prior)


def process_custom_prior(prior) -> Tuple[Distribution, int, bool]:
    """Check and return corrected prior object defined by the user.

    Args:
        prior: Prior object with `.sample()` and `.log_prob()` as provided by the user.

    Returns:
        prior: sbi-compatible prior.
        theta_numel: Number of parameters - elements in a single sample from the prior.
        is_prior_numpy: Whether the prior returned Numpy arrays before wrapping.
    """

    check_prior_attributes(prior)
    check_prior_batch_behavior(prior)
    prior, is_prior_numpy = maybe_wrap_prior_as_pytorch(prior)
    check_prior_return_type(prior)
    theta_numel = prior.sample().numel()

    return prior, theta_numel, is_prior_numpy


def maybe_wrap_prior_as_pytorch(prior) -> Tuple[Distribution, bool]:
    """Check prior return type and maybe wrap as PyTorch.

    Args:
        prior: Prior object with `.sample()` and `.log_prob()` as provided by the user.

    Raises:
        TypeError: If prior return type is PyTorch or Numpy.

    Returns:
        prior: Prior that emits samples and evaluates log prob as PyTorch Tensors.
        is_prior_numpy: Whether the prior returned Numpy arrays before wrapping.
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

    return cast(Distribution, prior), is_prior_numpy


def process_pytorch_prior(prior: Distribution) -> Tuple[Distribution, int, bool]:
    """Return PyTorch prior adapted to the requirements for sbi.

    Args:
        prior: PyTorch distribution prior provided by the user.

    Raises:
        ValueError: If prior is defined over an unwrapped scalar variable.

    Returns:
        prior: PyTorch distribution prior.
        theta_numel: Number of parameters - elements in a single sample from the prior.
        prior_returns_numpy: False.
    """

    # Reject unwrapped scalar priors.
    # This will reject Uniform priors with dimension larger than 1.
    if prior.sample().ndim == 0:
        raise ValueError(
            "Detected scalar prior. Please make sure to pass a PyTorch prior with "
            "`batch_shape=torch.Size([1])` or `event_shape=torch.Size([1])`."
        )
    # Cast 1D Uniform to BoxUniform to avoid shape error in mdn log prob.
    elif isinstance(prior, Uniform) and prior.batch_shape.numel() == 1:
        prior = BoxUniform(low=prior.low, high=prior.high)
        warnings.warn(
            "Casting 1D Uniform prior to BoxUniform to match sbi batch requirements."
        )

    check_prior_batch_behavior(prior)
    check_prior_batch_dims(prior)

    if not prior.sample().dtype == float32:
        prior = PytorchReturnTypeWrapper(prior, return_type=float32)

    # This will fail for float64 priors.
    check_prior_return_type(prior)

    theta_numel = prior.sample().numel()

    return prior, theta_numel, False


def check_prior_batch_dims(prior) -> None:
    """Check if batch shape of the prior is smaller or equal to 1.

    Raises:
        ValueError: If `batch_shape` larger than 1.
    """

    if prior.batch_shape.numel() > 1:
        raise ValueError(
            """The specified prior has `batch_shape` larger than 1. Please
            specify a prior with batch_shape smaller equal to 1 and `event_shape`
            equal to number of parameters of your model.

            In case your intention was to pass a univariate distribution like Uniform
            (or Beta, Gamma, etc.) defined over multiple parameters, consider instead
            using `torch.distributions.Independent` to reinterpret batch dimensions as
            event dimensions, or use the `MultipleIndependent` distribution we provide.

            To use `sbi.utils.MultipleIndependent`, just pass a list of priors, e.g. to 
            specify a uniform prior over two parameters, pass as prior:
                prior = [
                            Uniform(torch.zeros(1), torch.ones(1)),
                            Uniform(torch.ones(1), 2 * torch.ones(1))
                        ]
            or, to pass a Gamma over the first parameter and a correlated Gaussian over
            the other two, write:
                prior = [
                            Gamma(torch.ones(1), 2 * torch.ones(1)),
                            MVG(torch.zeros(2), tensor([[1., 0.1], [0.1, 2.]])),
                        ]
            """
        )


def check_for_possibly_batched_x_shape(x_shape):
    """Raise `ValueError` if dimensionality of simulations doesn't match requirements.

    sbi does not support multiple observations yet. For 2D observed data the leading
    dimension will be interpreted as batch dimension and a `ValueError` is raised if the
    batch dimension is larger than 1.

    Multidimensional observations e.g., images, are allowed when they are passed with an
    additional leading batch dimension of size 1.
    """

    x_ndim = len(x_shape)
    inferred_batch_shape, *inferred_data_shape = x_shape
    # Interpret first dimension as batch dimension.
    inferred_data_ndim = len(inferred_data_shape)

    # Reject multidimensional data with batch_shape > 1.
    if x_ndim > 1 and inferred_batch_shape > 1:
        raise ValueError(
            """Observation/simulation `x` has D>1 dimensions. sbi interprets the
            leading dimension as a batch dimension, but it *currently* only processes a
            single observation, a batch of several is not supported yet.

            NOTE: below we use list notation to reduce clutter, but `x` should be of 
            type torch.Tensor or ndarray.

            For example:

            > x_o = [[1]]
            > x_o = [[1, 2, 3]]

            are interpreted as single observations with a leading batch dimension of
            one. However

            > x_o = [ [1], [2] ]
            > x_o = [ [1,2,3], [4,5,6] ]

            are interpreted as a batch of two scalar or vector observations, which
            is not supported yet. The following is interpreted as a matrix-shaped
            observation, e.g. a monochromatic image:

            > x_o = [ [[1,2,3], [4,5,6]] ]

            Finally, for convenience,

            > x_o = [1]
            > x_o = [1, 2, 3]

            will be interpreted as a single scalar or single vector observation
            respectively, without the user needing to wrap or unsqueeze them.
            """
        )
    # Warn on multidimensional data with `batch_size` one.
    elif inferred_data_ndim > 1 and inferred_batch_shape == 1:
        warnings.warn(
            f"""Beware: The observed data (x_o) you passed was interpreted to have
            matrix shape: {inferred_data_shape}. The current implementation of sbi
            might not provide stable support for this and result in shape mismatches.
            """
        )
    else:
        pass


def check_for_possibly_batched_observations(x_o: Tensor):
    """Raise `ValueError` if dimensionality of data doesn't match requirements.

    sbi does not support multiple observations yet. For 2D observed data the leading
    dimension will be interpreted as batch dimension and a ValueError will be raised if
    the batch dimension is larger than 1.
    Multidimensional observations e.g., images, are allowed when they are passed with an
    additional leading batch dimension of size 1.
    """
    check_for_possibly_batched_x_shape(x_o.shape)


def check_prior_attributes(prior) -> None:
    """Check for prior methods sample(sample_shape) .log_prob(value) methods.

    Raises:
        AttributeError: if either of the two methods doesn't exist.
    """

    # Sample a batch of two parameters to check batch behaviour > 1 and that
    # `.sample()` can handle a tuple argument.
    num_samples = 2
    try:
        theta = prior.sample((num_samples,))
    except AttributeError:
        raise AttributeError(
            "Prior needs method `.sample()`. Consider using a PyTorch distribution."
        )
    except TypeError:
        raise TypeError(
            f"""The `prior.sample()` method must accept Tuple arguments, e.g.,
            prior.sample(({num_samples}, )) to sample a batch of 2 parameters. Consider
            using a PyTorch distribution."""
        )
    except:  # Catch any other error.
        raise ValueError(
            f"""Something went wrong when sampling a batch of parameters
            from the prior as `prior.sample(({num_samples}, ))`. Consider using a 
            PyTorch distribution."""
        )
    try:
        prior.log_prob(theta)
    except AttributeError:
        raise AttributeError(
            "Prior needs method `.log_prob()`. Consider using a PyTorch distribution."
        )
    except:  # Catch any other error.
        raise ValueError(
            """Something went wrong when evaluating a batch of parameters theta
            with `prior.log_prob(theta)`. Consider using a PyTorch distribution."""
        )


def check_prior_return_type(
    prior, return_type: Optional[torch.dtype] = float32
) -> None:
    """Check whether prior.sample() returns float32 Tensor."""

    prior_dtype = prior.sample().dtype
    assert (
        prior_dtype == return_type
    ), f"Prior return type must be {return_type}, but is {prior_dtype}."


def check_prior_batch_behavior(prior) -> None:
    """Assert that it is possible to sample and evaluate batches of parameters."""

    # Check for correct batch size in .sample and .log_prob
    num_samples = 1
    theta = prior.sample((num_samples,))
    log_probs = prior.log_prob(theta)

    assert (
        len(theta.shape) >= 2
    ), f"""A parameter batch sampled from the prior must be at least 2D,
    (num_samples, parameter_dim), but is {len(theta.shape)}"""

    num_sampled, *parameter_dim = theta.shape
    # Using len here because `log_prob` could be `ndarray` or `torch.Tensor`.
    num_log_probs = len(log_probs)

    assert (
        num_sampled == num_samples
    ), "prior.sample((batch_size, )) must return batch_size parameters."

    assert (
        num_log_probs == num_samples
    ), "prior.log_prob must return as many log probs as samples."


def process_simulator(
    user_simulator: Callable, prior, is_numpy_simulator: bool,
) -> Callable:
    """Return a simulator that meets the requirements for usage in sbi.

    Wraps the simulator to return only `torch.Tensor` and handle batches of parameters.
    """

    assert isinstance(user_simulator, Callable), "Simulator must be a function."

    pytorch_simulator = wrap_as_pytorch_simulator(
        user_simulator, prior, is_numpy_simulator
    )

    batch_simulator = ensure_batched_simulator(pytorch_simulator, prior)

    return batch_simulator


def wrap_as_pytorch_simulator(
    simulator: Callable, prior, is_numpy_simulator
) -> Callable:
    """Return a simulator that accepts and returns `Tensor` arguments."""

    if is_numpy_simulator:
        # Get data to check input type is consistent with data.
        theta = prior.sample().numpy()  # Cast to numpy because is in PyTorch already.
        x = simulator(theta)
        assert isinstance(
            x, ndarray
        ), f"Simulator output type {type(x)} must match its input type {type(theta)}"

        # Define a wrapper function to PyTorch
        def pytorch_simulator(theta: Tensor) -> Tensor:
            return torch.as_tensor(simulator(theta.numpy()), dtype=float32)

    else:
        # Define a wrapper to make sure that the output of the simulator is `float32`.
        def pytorch_simulator(theta: Tensor) -> Tensor:
            return torch.as_tensor(simulator(theta), dtype=float32)

    return pytorch_simulator


def ensure_batched_simulator(simulator: Callable, prior) -> Callable:
    """Return a simulator with batched output.

    Return the unchanged simulator if it can already simulate multiple parameter
    vectors per call. Otherwise, wrap as simulator with batched output (leading batch
    dimension of shape [1]).
    """

    is_batched_simulator = True
    try:
        batch_size = 2
        simulator_batch_size, *_ = simulator(prior.sample((batch_size,))).shape

        assert simulator_batch_size == batch_size
    except:
        is_batched_simulator = False

    return simulator if is_batched_simulator else get_batch_dim_simulator(simulator)


def get_batch_dim_simulator(simulator: Callable) -> Callable:
    """Return simulator wrapped with `map` to handle batches of parameters.

    Note: this batches the simulator only syntactically, there are no performance
    benefits as with true vectorization."""

    def batch_dim_simulator(theta: Tensor) -> Tensor:
        batch_shape, *_ = theta.shape
        assert (
            batch_shape == 1
        ), f"This simulator can handle one single thetas, theta.shape={theta.shape}."
        # Remove possible singleton dimensions in the user simulator.
        simulation = simulator(theta.squeeze()).squeeze()
        # Return with leading batch dimension.
        return simulation.unsqueeze(0)

    return batch_dim_simulator


def process_x(x: Tensor, x_shape: torch.Size) -> Tensor:
    """Return observed data adapted to match sbi's shape and type requirements.

    Args:
        x: Observed data as provided by the user.
        x_shape: Prescribed shape - either directly provided by the user at init or
            inferred by sbi by running a simulation and checking the output.

    Returns:
        x: Observed data with shape ready for usage in sbi.
    """

    x = torch.as_tensor(atleast_2d(x), dtype=float32)

    check_for_possibly_batched_observations(x)
    input_x_shape = x.shape

    assert input_x_shape == x_shape, (
        f"Observed data shape ({input_x_shape}) must match "
        f"the shape of simulated data x ({x_shape})."
    )

    return x


def prepare_for_sbi(simulator: Callable, prior,) -> Tuple[Callable, Distribution]:
    """Prepare simulator, prior and for usage in sbi.

    One of the goals is to allow you to use sbi with inputs computed in numpy.

    Attempts to meet the following requirements by reshaping and type-casting:
    - the simulator function receives as input and returns a Tensor.
    - the simulator can simulate batches of parameters and return batches of data.
    - the prior does not produce batches and samples and evaluates to Tensor.
    - the output shape is a `torch.Size((1,N))` (i.e, has a leading batch dimension 1).

    If this is not possible, a suitable exception will be raised.

    Args:
        simulator: Simulator as provided by the user.
        prior: Prior as provided by the user.

    Returns:
        Tuple (simulator, prior, x_shape) checked and matching the requirements of sbi.
    """

    # Check prior, return PyTorch prior.
    prior, _, prior_returns_numpy = process_prior(prior)

    # Check simulator, returns PyTorch simulator able to simulate batches.
    simulator = process_simulator(simulator, prior, prior_returns_numpy)

    # Consistency check after making ready for sbi.
    check_sbi_inputs(simulator, prior)

    return simulator, prior


def check_sbi_inputs(simulator: Callable, prior: Distribution) -> None:
    """Assert requirements for simulator, prior and observation for usage in sbi.

    Args:
        simulator: simulator function
        prior: prior (Distribution like)
        x_shape: Shape of single simulation output $x$.
    """
    num_prior_samples = 1
    theta = prior.sample((num_prior_samples,))
    theta_batch_shape, *_ = theta.shape
    simulation = simulator(theta)
    sim_batch_shape, *sim_event_shape = simulation.shape

    assert isinstance(theta, Tensor), "Parameters theta must be a `Tensor`."
    assert isinstance(simulation, Tensor), "Simulator output must be a `Tensor`."

    assert (
        theta_batch_shape == num_prior_samples
    ), f"""Theta batch shape {theta_batch_shape} must match
        num_samples={num_prior_samples}."""
    assert (
        sim_batch_shape == num_prior_samples
    ), f"""Simulation batch shape {sim_batch_shape} must match
        num_samples={num_prior_samples}."""


def check_estimator_arg(estimator: Union[str, Callable]) -> None:
    """Check (density or ratio) estimator argument passed by the user."""
    assert isinstance(estimator, str) or (
        isinstance(estimator, Callable) and not isinstance(estimator, nn.Module)
    ), (
        "The passed density estimator / classifier must be a string or a function "
        f"returning a nn.Module, but is {type(estimator)}"
    )
