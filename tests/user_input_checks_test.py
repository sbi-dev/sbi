# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Callable, Optional, Union, Dict, Any, Tuple, Union, cast, List, Sequence, TypeVar

import pytest
import torch
from pyknos.mdn.mdn import MultivariateGaussianMDN
from scipy.stats import beta, multivariate_normal, uniform
from torch import Tensor, eye, nn, ones, zeros
from torch.distributions import Beta, Distribution, Gamma, MultivariateNormal, Uniform

from sbi.inference import SNPE_C
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian
from sbi.user_input.user_input_checks import (
    prepare_for_sbi,
    process_prior,
    process_simulator,
    process_x,
)
from sbi.user_input.user_input_checks_utils import (
    CustomPytorchWrapper,
    MultipleIndependent,
    PytorchReturnTypeWrapper,
    ScipyPytorchWrapper,
)
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils.torchutils import BoxUniform


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
    return MultivariateNormal(theta, eye(theta.numel())).sample()


def numpy_linear_gaussian(theta):
    """Linear Gaussian simulator wrapped to get and return numpy."""
    return diagonal_linear_gaussian(torch.as_tensor(theta, dtype=torch.float32)).numpy()


def list_simulator(theta):
    return list(theta)


def matrix_simulator(theta):
    """Return a 2-by-2 matrix."""
    assert theta.numel() == 4
    return theta.reshape(1, 2, 2)


@pytest.mark.parametrize(
    "wrapper, prior",
    (
        (CustomPytorchWrapper, UserNumpyUniform(zeros(3), ones(3), return_numpy=True),),
        (ScipyPytorchWrapper, multivariate_normal()),
        (ScipyPytorchWrapper, uniform()),
        (ScipyPytorchWrapper, beta(a=1, b=1)),
        (
            PytorchReturnTypeWrapper,
            BoxUniform(zeros(3, dtype=torch.float64), ones(3, dtype=torch.float64)),
        ),
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

    # Test return type
    assert prior.sample().dtype == torch.float32


def test_reinterpreted_batch_dim_prior():
    """Test whether the right warning and error are raised for reinterpreted priors."""

    # Both must raise ValueError because we don't reinterpret batch dims anymore.
    with pytest.raises(ValueError):
        process_prior(Uniform(zeros(3), ones(3)))
    with pytest.raises(ValueError):
        process_prior(MultivariateNormal(zeros(2, 3), ones(3)))

    # This must pass without warnings or errors.
    process_prior(BoxUniform(zeros(3), ones(3)))


@pytest.mark.parametrize(
    "prior",
    (
        pytest.param(Uniform(0.0, 1.0), marks=pytest.mark.xfail),  # scalar prior.
        pytest.param(
            Uniform(zeros((1, 3)), ones((1, 3))), marks=pytest.mark.xfail
        ),  # batch shape > 1.
        pytest.param(
            MultivariateNormal(zeros(3, 3), eye(3)), marks=pytest.mark.xfail,
        ),  # batch shape > 1.
        pytest.param(
            Uniform(zeros(3), ones(3)), marks=pytest.mark.xfail
        ),  # batched uniform.
        Uniform(zeros(1), ones(1)),
        BoxUniform(zeros(3), ones(3)),
        MultivariateNormal(zeros(3), eye(3)),
        UserNumpyUniform(zeros(3), ones(3), return_numpy=False),
        UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
        BoxUniform(zeros(3, dtype=torch.float64), ones(3, dtype=torch.float64)),
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
    "x, x_shape", ((ones(3), torch.Size([1, 3])), (ones(1, 3), torch.Size([1, 3])),),
)
def test_process_x(x, x_shape):
    process_x(x, x_shape)


@pytest.mark.parametrize(
    "simulator, prior",
    (
        (diagonal_linear_gaussian, BoxUniform(zeros(1), ones(1))),
        (diagonal_linear_gaussian, BoxUniform(zeros(2), ones(2))),
        (numpy_linear_gaussian, UserNumpyUniform(zeros(2), ones(2), True)),
        (linear_gaussian_no_batch, BoxUniform(zeros(2), ones(2))),
        pytest.param(list_simulator, BoxUniform(zeros(2), ones(2)),),
    ),
)
def test_process_simulator(simulator: Callable, prior: Distribution):

    prior, theta_dim, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)

    n_batch = 1
    x = simulator(prior.sample((n_batch,)))

    assert isinstance(x, Tensor), "Processed simulator must return Tensor."
    assert (
        x.shape[0] == n_batch
    ), "Processed simulator must return as many data points as parameters in batch."


@pytest.mark.parametrize(
    "simulator, prior",
    (
        (linear_gaussian_no_batch, BoxUniform(zeros(3), ones(3)),),
        (
            numpy_linear_gaussian,
            UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
        ),
        (diagonal_linear_gaussian, BoxUniform(zeros(3), ones(3))),
        (
            diagonal_linear_gaussian,
            BoxUniform(zeros(3, dtype=torch.float64), ones(3, dtype=torch.float64)),
        ),
        (
            numpy_linear_gaussian,
            UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
        ),
        (
            diagonal_linear_gaussian,
            [
                Gamma(ones(1), ones(1)),
                Beta(ones(1), ones(1)),
                MultivariateNormal(zeros(2), eye(2)),
            ],
        ),
        pytest.param(list_simulator, BoxUniform(zeros(3), ones(3)),),
    ),
)
def test_prepare_sbi_problem(simulator: Callable, prior):
    """Test user interface by passing different kinds of simulators, prior and data.

    Args:
        simulator: simulator function
        prior: prior as defined by the user (pytorch, scipy, custom)
        x_shape: shape of data as defined by the user.
    """

    simulator, prior = prepare_for_sbi(simulator, prior)

    # check batch sims and type
    n_batch = 1
    assert simulator(prior.sample((n_batch,))).shape[0] == n_batch
    assert isinstance(simulator(prior.sample((1,))), Tensor)
    assert prior.sample().dtype == torch.float32


@pytest.mark.parametrize(
    "user_simulator, user_prior",
    (
        (
            diagonal_linear_gaussian,
            BoxUniform(zeros(3, dtype=torch.float64), ones(3, dtype=torch.float64)),
        ),
        (linear_gaussian_no_batch, BoxUniform(zeros(3), ones(3))),
        (
            numpy_linear_gaussian,
            UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
        ),
        (diagonal_linear_gaussian, BoxUniform(zeros(3), ones(3))),
        (linear_gaussian_no_batch, BoxUniform(zeros(3), ones(3))),
        (list_simulator, BoxUniform(-ones(3), ones(3))),
        (
            numpy_linear_gaussian,
            UserNumpyUniform(zeros(3), ones(3), return_numpy=True),
        ),
        (
            diagonal_linear_gaussian,
            (
                Gamma(ones(1), ones(1)),
                Beta(ones(1), ones(1)),
                MultivariateNormal(zeros(2), eye(2)),
            ),
        ),
    ),
)
def test_inference_with_user_sbi_problems(user_simulator: Callable, user_prior):
    """
    Test inference with combinations of user defined simulators, priors and x_os.
    """

    infer = SNPE_C(
        *prepare_for_sbi(user_simulator, user_prior),
        density_estimator="maf",
        simulation_batch_size=1,
        show_progress_bars=False,
    )

    # Run inference.
    _ = infer(num_simulations=100, max_num_epochs=2)


@pytest.mark.parametrize(
    "dists",
    [
        pytest.param(
            [Beta(ones(1), 2 * ones(1))], marks=pytest.mark.xfail
        ),  # single dist.
        pytest.param(
            [Gamma(ones(2), ones(1))], marks=pytest.mark.xfail
        ),  # single batched dist.
        pytest.param(
            [
                Gamma(ones(2), ones(1)),
                MultipleIndependent(
                    [Uniform(zeros(1), ones(1)), Uniform(zeros(1), ones(1))]
                ),
            ],
            marks=pytest.mark.xfail,
        ),  # nested definition.
        pytest.param(
            [Uniform(0, 1), Beta(1, 2)], marks=pytest.mark.xfail
        ),  # scalar dists.
        [Uniform(zeros(1), ones(1)), Uniform(zeros(1), ones(1))],
        (
            Gamma(ones(1), ones(1)),
            Uniform(zeros(1), ones(1)),
            Beta(ones(1), 2 * ones(1)),
        ),
        [MultivariateNormal(zeros(3), eye(3)), Gamma(ones(1), ones(1))],
    ],
)
def test_independent_joint_shapes_and_samples(dists):
    """Test return shapes and validity of samples by comparing to samples generated
    from underlying distributions individually."""

    # Fix the seed for reseeding within this test.
    seed = 0

    joint = MultipleIndependent(dists)

    # Check shape of single sample and log prob
    sample = joint.sample()
    assert sample.shape == torch.Size([joint.ndims])
    assert joint.log_prob(sample).shape == torch.Size([1])

    num_samples = 10

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
    for d in dists:
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
        Gamma(ones(1), ones(1)),
        Uniform(zeros(1), ones(1)),
        Beta(ones(1), 2 * ones(1)),
    ]
    joint = MultipleIndependent(dists)

    # Test too many input dimensions.
    with pytest.raises(AssertionError):
        joint.log_prob(ones(10, 4))

    # Test nested construction.
    with pytest.raises(AssertionError):
        MultipleIndependent([joint])

    # Test 3D value.
    with pytest.raises(AssertionError):
        joint.log_prob(ones(10, 4, 1))




@pytest.mark.parametrize(
    "arg",
    [
        ("maf"),
        ("MAF"),
        pytest.param(
            "nn",
            marks=pytest.mark.xfail(
                AssertionError,
                reason=(
                    "custom density estimator must be a function return nn.Module, "
                    "not the nn.Module."
                ),
            ),
        ),
        ("fun"),
    ],
)
def test_passing_custom_density_estimator(arg):

    x_numel = 2
    y_numel = 2
    hidden_features = 10
    num_components = 1
    mdn = MultivariateGaussianMDN(
        features=x_numel,
        context_features=y_numel,
        hidden_features=hidden_features,
        hidden_net=nn.Sequential(
            nn.Linear(y_numel, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ),
        num_components=num_components,
        custom_initialization=True,
    )
    if arg == "nn":
        # Just the nn.Module.
        density_estimator = mdn
    elif arg == "fun":
        # A function returning the nn.Module.
        density_estimator = lambda batch_theta, batch_x: mdn
    else:
        density_estimator = arg
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    SNPE_C(diagonal_linear_gaussian, prior, density_estimator=density_estimator)
