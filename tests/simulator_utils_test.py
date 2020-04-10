from __future__ import annotations

from typing import Callable, Union
import pytest
import torch
from torch.distributions import Uniform, MultivariateNormal, Distribution
from torch import Tensor
from sbi.simulators.simutils import (
    prepare_sbi_problem,
    process_prior,
    process_observed_data,
    process_simulator,
    ScipyPytorchWrapper,
    CustomPytorchWrapper,
    simulate_in_batches,
)
from scipy.stats import multivariate_normal, uniform, beta
from sbi.simulators.linear_gaussian import linear_gaussian
from sbi.utils.torchutils import BoxUniform
from sbi.utils.get_nn_models import posterior_nn
from sbi.inference.snpe import SnpeC
import numpy as np


class UserNumpyUniform:
    """User defined numpy uniform prior. 
    
    Used for testing to mimick a user-defined prior with valid .sample and .log_prob 
    methods. 
    """

    def __init__(self, lower: Tensor, upper: Tensor, return_numpy: bool = False):
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


def list_simulator(theta):
    return list(theta)


def identity_simulator(theta):
    return theta


def matrix_simulator(theta):
    """Return a 2-by-2 matrix."""
    assert theta.numel() == 4
    return theta.reshape(2, 2)


@pytest.mark.parametrize(
    "wrapper, prior",
    (
        (
            CustomPytorchWrapper,
            UserNumpyUniform(torch.zeros(3), torch.ones(3), return_numpy=True),
        ),
        (ScipyPytorchWrapper, multivariate_normal()),
        (ScipyPytorchWrapper, uniform()),
        (ScipyPytorchWrapper, beta(a=1, b=1)),
    ),
)
def test_prior_wrappers(wrapper, prior):
    """Test prior wrappers to pytorch distributions."""
    prior = wrapper(prior)

    # use 2 here to test for minimal case >1
    batch_size = 2
    theta = prior.sample((batch_size,))
    assert isinstance(theta, Tensor)
    assert theta.shape[0] == batch_size

    # Test log prob on batch of thetas.
    log_probs = prior.log_prob(theta)
    assert isinstance(log_probs, Tensor)
    assert log_probs.shape[0] == batch_size


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
        UserNumpyUniform(torch.zeros(3), torch.ones(3), return_numpy=False),
        UserNumpyUniform(torch.zeros(3), torch.ones(3), return_numpy=True),
    ),
)
def test_process_prior(prior):

    prior, parameter_dim, numpy_simulator = process_prior(prior)

    batch_size = 2
    theta = prior.sample((batch_size,))
    assert theta.shape == torch.Size(
        (batch_size, parameter_dim)
    ), "Number of sampled parameters must match batch size."
    assert (
        prior.log_prob(theta).shape[0] == batch_size
    ), "Number of log probs must match number of input values."


@pytest.mark.parametrize(
    "prior, observed_data",
    (
        (BoxUniform(torch.zeros(3), torch.ones(3)), torch.ones(3)),
        (BoxUniform(torch.zeros(3), torch.ones(3)), np.ones(3)),
        pytest.param(
            BoxUniform(torch.zeros(1), torch.ones(1)), 2.0, marks=pytest.mark.xfail
        ),
        pytest.param(
            BoxUniform(torch.zeros(3), torch.ones(3)), [2.0], marks=pytest.mark.xfail
        ),
        pytest.param(
            BoxUniform(torch.zeros(2), torch.ones(2)), [[1]], marks=pytest.mark.xfail,
        ),
        (BoxUniform(torch.zeros(3), torch.ones(3)), torch.ones(1, 3)),
        (BoxUniform(torch.zeros(1), torch.ones(1)), torch.ones(1, 1)),
        (BoxUniform(torch.zeros(3), torch.ones(3)), np.zeros((1, 3))),
        (BoxUniform(torch.zeros(1), torch.ones(1)), np.zeros((1, 1))),
    ),
)
def test_process_observed_data(
    prior: Distribution,
    observed_data: Union[Tensor, np.ndarray],
    simulator: Optional[Callable] = linear_gaussian,
):
    observed_data, observation_dim = process_observed_data(
        observed_data, simulator, prior
    )

    assert observed_data.shape == torch.Size([1, observation_dim])


def test_process_matrix_observation():
    prior = BoxUniform(torch.zeros(4), torch.ones(4))
    observed_data = np.zeros((1, 2, 2))
    simulator = matrix_simulator

    observed_data, observation_dim = process_observed_data(
        observed_data, simulator, prior
    )


@pytest.mark.parametrize(
    "simulator, prior",
    (
        (linear_gaussian, BoxUniform(torch.zeros(1), torch.ones(1))),
        (linear_gaussian, BoxUniform(torch.zeros(2), torch.ones(2))),
        (numpy_simulator, UserNumpyUniform(torch.zeros(2), torch.ones(2), True)),
        (numpy_linear_gaussian, UserNumpyUniform(torch.zeros(2), torch.ones(2), True)),
        (torch_simulator_no_batch, BoxUniform(torch.zeros(2), torch.ones(2))),
        pytest.param(
            list_simulator,
            BoxUniform(torch.zeros(2), torch.ones(2)),
            marks=pytest.mark.xfail,
        ),
    ),
)
def test_process_simulator(simulator: Callable, prior: Distribution):

    prior, theta_dim, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)

    n_batch = 2
    x = simulator(prior.sample((n_batch,)))

    assert isinstance(x, Tensor), "Processed simulator must return Tensor."
    assert (
        x.shape[0] == n_batch
    ), "Processed simulator must return as many data points as parameters in batch."


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
            UserNumpyUniform(torch.zeros(3), torch.ones(3), return_numpy=True),
            np.zeros((1, 3)),
        ),
        (linear_gaussian, BoxUniform(torch.zeros(3), torch.ones(3)), torch.zeros(1, 3)),
        (linear_gaussian, BoxUniform(torch.zeros(3), torch.ones(3)), torch.zeros(3)),
        pytest.param(
            list_simulator,
            BoxUniform(torch.zeros(3), torch.ones(3)),
            torch.zeros(1, 3),
            marks=pytest.mark.xfail,
        ),
        (
            numpy_linear_gaussian,
            UserNumpyUniform(torch.zeros(3), torch.ones(3), return_numpy=True),
            np.zeros((1, 3)),
        ),
    ),
)
def test_prepare_sbi_problem(
    simulator: Callable, prior, observed_data: Union[Tensor, np.ndarray]
):
    """Test user interface by passing different kinds of simulators, prior and data.

    Args:
        simulator: simulator function
        prior: prior as defined by the user (pytorch, scipy, custom)
        observed_data: data as defined by the user. 
    """

    simulator, prior, observed_data = prepare_sbi_problem(
        simulator, prior, observed_data
    )

    # check batch sims and type
    n_batch = 2
    assert simulator(prior.sample((n_batch,))).shape[0] == n_batch
    assert isinstance(simulator(prior.sample((1,))), Tensor)


@pytest.mark.parametrize(
    "num_samples", (pytest.param(0, marks=pytest.mark.xfail), 100, 1000)
)
@pytest.mark.parametrize("batch_size", (1, 100, 1000))
def test_simulate_in_batches(
    num_samples,
    batch_size,
    simulator=linear_gaussian,
    prior=BoxUniform(torch.zeros(5), torch.ones(5)),
):
    """Test combinations of num_samples and simulation_batch_size. """

    simulate_in_batches(
        simulator,
        lambda n: prior.sample((n,)),
        num_samples,
        batch_size,
        torch.Size([5]),
    )


def test_inference_with_pilot_samples_samples():
    """Test whether num_pilot_samples can be same as num_simulations_per_round."""

    num_dim = 3
    true_observation = torch.zeros((1, num_dim))

    prior = MultivariateNormal(
        loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
    )

    infer = SnpeC(
        simulator=linear_gaussian,
        true_observation=true_observation,
        density_estimator=posterior_nn(
            model="maf", prior=prior, context=true_observation,
        ),
        prior=prior,
        simulation_batch_size=100,
    )

    # Run inference.
    num_rounds, num_simulations_per_round = 2, 100
    posterior = infer(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )
