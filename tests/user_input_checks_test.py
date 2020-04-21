from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import pytest
import torch
import warnings
from scipy.stats import beta, multivariate_normal, uniform
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal, Uniform, Gamma, Beta

from sbi.inference.snpe import SnpeC
from sbi.simulators.linear_gaussian import linear_gaussian
from sbi.simulators.user_input_checks import (
    prepare_sbi_problem,
    process_prior,
    process_simulator,
    process_x_o,
)
from sbi.simulators.user_input_checks_utils import (
    CustomPytorchWrapper,
    PytorchReturnTypeWrapper,
    ScipyPytorchWrapper,
    IndependentJoint,
)
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils.torchutils import BoxUniform

# use cpu by default
torch.set_default_tensor_type("torch.FloatTensor")


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


def linear_gaussian_no_batch(theta):
    """Identity simulator throwing assertion error when called on a batch."""
    assert theta.ndim == 1, "cant handle batches."
    return MultivariateNormal(theta, torch.eye(theta.numel())).sample()


def numpy_linear_gaussian(theta):
    """Linear Gaussian simulator wrapped to get and return numpy."""
    return linear_gaussian(torch.as_tensor(theta, dtype=torch.float32)).numpy()


def list_simulator(theta):
    return list(theta)


def matrix_simulator(theta):
    """Return a 2-by-2 matrix."""
    assert theta.numel() == 4
    return theta.reshape(2, 2)


def test_reinterpreted_batch_dim_prior():
    """Test whether the right warning and error are raised for reinterpreted priors."""

    with pytest.warns(UserWarning, match="The specified prior is a prior on"):
        process_prior(Uniform(torch.zeros(3), torch.ones(3)))

    # Pass a MVN with batch_shape>1 to enforce ValueError.
    with pytest.raises(ValueError):
        process_prior(MultivariateNormal(torch.zeros(2, 3), torch.ones(3)))

    # This must pass without warnings or errors.
    process_prior(BoxUniform(torch.zeros(3), torch.ones(3)))


@pytest.mark.parametrize(
    "prior",
    (
        pytest.param(Uniform(0.0, 1.0), marks=pytest.mark.xfail),
        pytest.param(
            Uniform(torch.zeros((1, 3)), torch.ones((1, 3))), marks=pytest.mark.xfail
        ),
        pytest.param(
            MultivariateNormal(torch.zeros(3, 3), torch.eye(3)),
            marks=pytest.mark.xfail,
        ),
        Uniform(torch.zeros(3), torch.ones(3)),
        Uniform(torch.zeros(1), torch.ones(1)),
        BoxUniform(torch.zeros(3), torch.ones(3)),
        MultivariateNormal(torch.zeros(3), torch.eye(3)),
        UserNumpyUniform(torch.zeros(3), torch.ones(3), return_numpy=False),
        UserNumpyUniform(torch.zeros(3), torch.ones(3), return_numpy=True),
        BoxUniform(
            torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
        ),
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
    "prior, x_o",
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
def test_process_x_o(
    prior: Distribution,
    x_o: Union[Tensor, np.ndarray],
    simulator: Optional[Callable] = linear_gaussian,
):
    x_o, x_o_dim = process_x_o(x_o, simulator, prior)

    assert x_o.shape == torch.Size([1, x_o_dim])


def test_process_matrix_observation():
    prior = BoxUniform(torch.zeros(4), torch.ones(4))
    x_o = np.zeros((1, 2, 2))
    simulator = matrix_simulator

    x_o, x_o_dim = process_x_o(x_o, simulator, prior)


@pytest.mark.parametrize(
    "simulator, prior",
    (
        (linear_gaussian, BoxUniform(torch.zeros(1), torch.ones(1))),
        (linear_gaussian, BoxUniform(torch.zeros(2), torch.ones(2))),
        (numpy_linear_gaussian, UserNumpyUniform(torch.zeros(2), torch.ones(2), True)),
        (linear_gaussian_no_batch, BoxUniform(torch.zeros(2), torch.ones(2))),
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
    "simulator, prior, x_o",
    (
        (
            linear_gaussian_no_batch,
            BoxUniform(torch.zeros(3), torch.ones(3)),
            torch.zeros(3),
        ),
        (
            numpy_linear_gaussian,
            UserNumpyUniform(torch.zeros(3), torch.ones(3), return_numpy=True),
            np.zeros((1, 3)),
        ),
        (linear_gaussian, BoxUniform(torch.zeros(3), torch.ones(3)), torch.zeros(1, 3)),
        (
            linear_gaussian,
            BoxUniform(
                torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
            ),
            torch.zeros(3),
        ),
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
        pytest.param(  # test simulator always returning batch dim.
            lambda _: torch.zeros(1, 480),
            BoxUniform(torch.zeros(2), torch.ones(2)),
            torch.zeros(1, 480),
            marks=pytest.mark.xfail,
        ),
    ),
)
def test_prepare_sbi_problem(
    simulator: Callable, prior, x_o: Union[Tensor, np.ndarray]
):
    """Test user interface by passing different kinds of simulators, prior and data.

    Args:
        simulator: simulator function
        prior: prior as defined by the user (pytorch, scipy, custom)
        x_o: data as defined by the user.
    """

    simulator, prior, x_o = prepare_sbi_problem(simulator, prior, x_o)

    # check batch sims and type
    n_batch = 2
    assert simulator(prior.sample((n_batch,))).shape[0] == n_batch
    assert isinstance(simulator(prior.sample((1,))), Tensor)
    assert prior.sample().dtype == torch.float32


@pytest.mark.parametrize(
    "user_simulator, user_prior, user_x_o",
    (
        (
            linear_gaussian,
            BoxUniform(
                torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
            ),
            torch.zeros(3),
        ),
        (
            linear_gaussian_no_batch,
            BoxUniform(torch.zeros(3), torch.ones(3)),
            torch.zeros(3),
        ),
        (
            numpy_linear_gaussian,
            UserNumpyUniform(torch.zeros(3), torch.ones(3), return_numpy=True),
            np.zeros((1, 3)),
        ),
        (linear_gaussian, BoxUniform(torch.zeros(3), torch.ones(3)), torch.zeros(1, 3)),
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
        pytest.param(  # test simulator always returning batch dim.
            lambda _: torch.zeros(1, 480),
            BoxUniform(torch.zeros(2), torch.ones(2)),
            torch.zeros(1, 480),
            marks=pytest.mark.xfail,
        ),
    ),
)
def test_inference_with_user_sbi_problems(
    user_simulator: Callable, user_prior, user_x_o: Union[Tensor, np.ndarray]
):
    """
    Test inference with combinations of user defined simulators, priors and x_os.
    """

    infer = SnpeC(
        simulator=user_simulator,
        x_o=user_x_o,
        # XXX: Dont pass density_estimator as that would bypass prior checking.
        # density_estimator=posterior_nn(model="maf", prior=prior, x_o=x_o,),
        prior=user_prior,
        simulation_batch_size=100,
    )

    # Run inference.
    infer(num_rounds=1, num_simulations_per_round=100)


@pytest.mark.parametrize(
    "dists",
    [
        pytest.param([Beta(torch.ones(1), 2 * torch.ones(1))], marks=pytest.mark.xfail),
        pytest.param([Gamma(torch.ones(2), torch.ones(1))], marks=pytest.mark.xfail),
        pytest.param(
            [
                Gamma(torch.ones(2), torch.ones(1)),
                IndependentJoint(
                    [
                        Uniform(torch.zeros(1), torch.ones(1)),
                        Uniform(torch.zeros(1), torch.ones(1)),
                    ]
                ),
            ],
            marks=pytest.mark.xfail,
        ),
        [
            Uniform(torch.zeros(1), torch.ones(1)),
            Uniform(torch.zeros(1), torch.ones(1)),
        ],
        [
            Gamma(torch.ones(1), torch.ones(1)),
            Uniform(torch.zeros(1), torch.ones(1)),
            Beta(torch.ones(1), 2 * torch.ones(1)),
        ],
        [
            MultivariateNormal(torch.zeros(3), torch.eye(3)),
            Gamma(torch.ones(1), torch.ones(1)),
        ],
    ],
)
def test_independent_joint_shapes_and_samples(dists):
    """Test return shapes and validity of samples by comparing to samples generated
    from underlying distributions individually."""

    # Fix the seed for reseeding within this test.
    seed = 0

    joint = IndependentJoint(dists)

    # Check shape of single sample and log prob
    sample = joint.sample()
    assert sample.shape == torch.Size([joint.ndims])
    assert joint.log_prob(sample).shape == torch.Size([1])

    num_samples = 10
    num_dists = len(dists)

    # seed sampling for later comparison.
    torch.manual_seed(seed)
    samples = joint.sample((num_samples,))
    log_probs = joint.log_prob(samples)

    # Check sample and log_prob return shapes.
    assert samples.shape == torch.Size(
        [num_samples, joint.ndims]
    ) or samples.shape == torch.Size([num_samples])
    assert log_probs.shape == torch.Size([num_samples])

    # Seed again to get same samples by hand.
    torch.manual_seed(seed)
    true_samples = []
    true_log_probs = []

    # Get samples and log probs by hand.
    for idx, d in enumerate(dists):
        sample = d.sample((num_samples,))
        true_samples.append(sample)
        true_log_probs.append(d.log_prob(sample).reshape(num_samples, -1))

    # collect in Tensor.
    true_samples = torch.cat(true_samples, dim=-1)
    true_log_probs = torch.cat(true_log_probs, dim=-1).sum(-1)

    # Check whether independent joint sample equal individual samples.
    assert (true_samples == samples).all()
    assert (true_log_probs == log_probs).all()


def test_invalid_inputs():

    dists = [
        Gamma(torch.ones(1), torch.ones(1)),
        Uniform(torch.zeros(1), torch.ones(1)),
        Beta(torch.ones(1), 2 * torch.ones(1)),
    ]
    joint = IndependentJoint(dists)

    # Test too many input dimensions.
    with pytest.raises(AssertionError):
        joint.log_prob(torch.ones(10, 4))

    # Test nested construction.
    with pytest.raises(AssertionError):
        IndependentJoint([joint])

    # Test 3D value.
    with pytest.raises(AssertionError):
        joint.log_prob(torch.ones(10, 4, 1))
