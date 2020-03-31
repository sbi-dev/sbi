from typing import Callable
import pytest
import torch
from torch.distributions import Uniform, MultivariateNormal, Distribution
from sbi.simulators.simutils import (
    check_user_input,
    check_prior,
    check_observed_data,
    check_simulator,
    ScipyToPytorchWrapper,
    CustomToPytorchWrapper,
)
from scipy.stats import multivariate_normal, uniform, beta
from sbi.simulators.linear_gaussian import linear_gaussian
from sbi.utils.torchutils import BoxUniform
import numpy as np


class UserUniform:
    """User defined numpy uniform prior. 
    """

    def __init__(self, lower, upper, return_numpy=False):
        self.lower = lower
        self.upper = upper
        self.dist = BoxUniform(lower, upper)
        self.return_numpy = return_numpy

    def sample(self, sample_shape=torch.Size([])):
        samples = self.dist.sample(sample_shape)
        return samples.numpy() if self.return_numpy else samples

    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        log_probs = self.dist.log_prob(values)
        return log_probs.numpy() if self.return_numpy else log_probs


def torch_simulator_no_batch(x):
    """Identity simulator throwing assertion error when called on a batch."""
    assert x.ndim == 1, "cant handle batches, bitches."
    return x


def numpy_simulator(x):
    """Simulator returning zeroed input, Assertion error if input is not numpy."""
    assert isinstance(x, np.ndarray)
    return np.zeros_like(x)


def numpy_linear_gaussian(theta):
    """Linear Gaussian simulator wrapped to get and return numpy."""
    return linear_gaussian(torch.as_tensor(theta, dtype=torch.float32)).numpy()


def list_simulator(x):
    return list(x)


@pytest.mark.parametrize(
    "wrapper, prior",
    (
        (
            CustomToPytorchWrapper,
            UserUniform(torch.zeros(3), torch.ones(3), return_numpy=True),
        ),
        (ScipyToPytorchWrapper, multivariate_normal()),
        (ScipyToPytorchWrapper, uniform()),
        (ScipyToPytorchWrapper, beta(a=1, b=1)),
    ),
)
def test_prior_wrappers(wrapper, prior):

    prior = wrapper(prior)

    theta = prior.sample((10,))
    assert isinstance(theta, torch.Tensor)
    assert theta.shape[0] == 10

    log_probs = prior.log_prob(theta)
    assert isinstance(log_probs, torch.Tensor)
    assert log_probs.shape[0] == 10


@pytest.mark.parametrize(
    "prior",
    (
        pytest.param(Uniform(0.0, 1.0), marks=pytest.mark.xfail),
        pytest.param(Uniform(torch.zeros(3), torch.ones(3)), marks=pytest.mark.xfail),
        pytest.param(
            Uniform(torch.zeros((1, 3)), torch.ones((1, 3))), marks=pytest.mark.xfail
        ),
        Uniform(torch.zeros(1), torch.ones(1)),
        BoxUniform(torch.zeros(3), torch.ones(3)),
        MultivariateNormal(torch.zeros(3), torch.eye(3)),
        UserUniform(torch.zeros(3), torch.ones(3), return_numpy=False),
        UserUniform(torch.zeros(3), torch.ones(3), return_numpy=True),
    ),
)
def test_check_prior(prior):

    prior, parameter_dim, numpy_simulator = check_prior(prior)

    batch_size = 2
    theta = prior.sample((batch_size,))
    assert theta.shape == torch.Size(
        [batch_size, parameter_dim]
    ), "wrong batch behavior after prior check."
    assert (
        prior.log_prob(theta).shape[0] == batch_size
    ), "Wrong log prob shape after prior check."


@pytest.mark.parametrize(
    "prior, observed_data",
    (
        pytest.param(
            BoxUniform(torch.zeros(3), torch.ones(3)),
            torch.ones(3),
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            BoxUniform(torch.zeros(1), torch.ones(1)), 2.0, marks=pytest.mark.xfail
        ),
        pytest.param(
            BoxUniform(torch.zeros(3), torch.ones(3)), [2.0], marks=pytest.mark.xfail
        ),
        (BoxUniform(torch.zeros(3), torch.ones(3)), torch.ones(1, 3)),
        (BoxUniform(torch.zeros(1), torch.ones(1)), torch.ones(1, 1)),
        (BoxUniform(torch.zeros(3), torch.ones(3)), np.zeros((1, 3))),
        (BoxUniform(torch.zeros(1), torch.ones(1)), np.zeros((1, 1))),
        (BoxUniform(torch.zeros(2), torch.ones(2)), [[1.0, 3.0]]),
        (BoxUniform(torch.zeros(1), torch.ones(1)), [[2.0]]),
    ),
)
def test_check_observed_data(
    prior, observed_data, simulator=linear_gaussian,
):
    observed_data, observation_dim = check_observed_data(
        observed_data, simulator, prior
    )

    assert observed_data.shape == torch.Size([1, observation_dim])


@pytest.mark.parametrize(
    "simulator, prior",
    (
        (linear_gaussian, BoxUniform(torch.zeros(1), torch.ones(1))),
        (linear_gaussian, BoxUniform(torch.zeros(2), torch.ones(2))),
        (numpy_simulator, UserUniform(torch.zeros(2), torch.ones(2), True)),
        (numpy_linear_gaussian, UserUniform(torch.zeros(2), torch.ones(2), True)),
        (torch_simulator_no_batch, BoxUniform(torch.zeros(2), torch.ones(2))),
        pytest.param(
            list_simulator,
            BoxUniform(torch.zeros(2), torch.ones(2)),
            marks=pytest.mark.xfail,
        ),
    ),
)
def test_check_simulator(simulator: Callable, prior: Distribution):

    prior, theta_dim, prior_returns_numpy = check_prior(prior)
    simulator = check_simulator(simulator, prior, prior_returns_numpy)

    n_batch = 2
    x = simulator(prior.sample((n_batch,)))

    assert isinstance(x, torch.Tensor), "corrected simulator must return Tensor."
    assert x.shape[0] == n_batch, "incorrected simulator return shape."


@pytest.mark.parametrize(
    "simulator, prior, observed_data",
    (
        (
            torch_simulator_no_batch,
            BoxUniform(torch.zeros(3), torch.ones(3)),
            torch.zeros(1, 3),
        ),
        (
            numpy_simulator,
            UserUniform(torch.zeros(3), torch.ones(3), return_numpy=True),
            np.zeros((1, 3)),
        ),
        (linear_gaussian, BoxUniform(torch.zeros(3), torch.ones(3)), torch.zeros(1, 3)),
        pytest.param(
            linear_gaussian,
            BoxUniform(torch.zeros(3), torch.ones(3)),
            torch.zeros(3),
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            list_simulator,
            BoxUniform(torch.zeros(3), torch.ones(3)),
            torch.zeros(1, 3),
            marks=pytest.mark.xfail,
        ),
        (
            numpy_linear_gaussian,
            UserUniform(torch.zeros(3), torch.ones(3), return_numpy=True),
            np.zeros((1, 3)),
        ),
    ),
)
def test_check_user_input(simulator: Callable, prior, observed_data):
    """Test user interface by passing different kinds of simulator functions.

    Arguments:
        simulator -- simulator function
    """

    simulator, prior, observed_data = check_user_input(simulator, prior, observed_data)

    # check batch sims and type
    n_batch = 2
    assert simulator(prior.sample((n_batch,))).shape[0] == n_batch
    assert isinstance(simulator(prior.sample((1,))), torch.Tensor)

    # check function attributes
    assert simulator.parameter_dim == 3
    assert simulator.observation_dim == 3
