# TODO ESPECIALLY THE VI METHODS AND VARIATIONAL FAMILIES WHEN FINSIHED SHOULD BE
# THESTED HERE!!!!

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.distributions.transforms as torch_tf
from torch import eye, ones, zeros
from torch.distributions import Beta, Binomial, Gamma, MultivariateNormal

from sbi.inference import SNLE, likelihood_estimator_based_potential
from sbi.inference.posteriors import VIPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.simulators.linear_gaussian import true_posterior_linear_gaussian_mvn_prior
from sbi.utils import MultipleIndependent
from tests.test_utils import check_c2st


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("vi_method", ("rKL", "fKL", "IW", "alpha"))
@pytest.mark.parametrize("sampling_method", ("naive", "sir"))
def test_c2st_vi_on_Gaussian(num_dim: int, vi_method: str, sampling_method: str):
    """Test VI on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        vi_method: different vi methods
        sampling_method: Different sampling methods

    """

    if sampling_method == "naive" and vi_method == "IW":
        return  # This is not meant to perform goood ...

    num_samples = 2000

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)

    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    class TractablePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return target_distribution.log_prob(
                torch.as_tensor(theta, dtype=torch.float32)
            )

        def allow_iid_x(self) -> bool:
            return True

    prior = MultivariateNormal(prior_mean, prior_cov)
    potential_fn = TractablePotential(prior=prior)
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(potential_fn, prior, theta_transform=theta_transform)
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))
    posterior.vi_method = vi_method
    posterior.train()
    samples = posterior.sample((num_samples,), method=sampling_method)
    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("q", ("maf", "nsf", "gaussian_diag", "gaussian", "mcf", "scf"))
def test_c2st_vi_flows_on_Gaussian(num_dim: int, q: str):
    """Test VI on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        vi_method: different vi methods
        sampling_method: Different sampling methods

    """
    # Coupling flows undefined at 1d
    if num_dim == 1 and q in ["mcf", "scf"]:
        return

    num_samples = 2000

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)

    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    class TractablePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return target_distribution.log_prob(
                torch.as_tensor(theta, dtype=torch.float32)
            )

        def allow_iid_x(self) -> bool:
            return True

    prior = MultivariateNormal(prior_mean, prior_cov)
    potential_fn = TractablePotential(prior=prior)
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(potential_fn, prior, theta_transform=theta_transform, q=q)
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))
    posterior.train(n_particles=1000, eps=1e-8)
    samples = posterior.sample((num_samples,))
    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.slow
@pytest.mark.parametrize("num_dim", (1, 2))
def test_c2st_vi_external_distributions_on_Gaussian(num_dim: int):
    """Test VI on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        vi_method: different vi methods
        sampling_method: Different sampling methods

    """
    num_samples = 2000

    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)
    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)

    x_o = zeros((1, num_dim))
    target_distribution = true_posterior_linear_gaussian_mvn_prior(
        x_o[0], likelihood_shift, likelihood_cov, prior_mean, prior_cov
    )
    target_samples = target_distribution.sample((num_samples,))

    class TractablePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return target_distribution.log_prob(
                torch.as_tensor(theta, dtype=torch.float32)
            )

        def allow_iid_x(self) -> bool:
            return True

    prior = MultivariateNormal(prior_mean, prior_cov)
    potential_fn = TractablePotential(prior=prior)
    theta_transform = torch_tf.identity_transform

    mu = zeros(num_dim, requires_grad=True)
    scale = ones(num_dim, requires_grad=True)
    q = torch.distributions.Independent(torch.distributions.Normal(mu, scale), 1)
    posterior = VIPosterior(
        potential_fn,
        prior,
        theta_transform=theta_transform,
        q=q,
        vi_method="rKL",
        parameters=[mu, scale],
    )
    posterior.set_default_x(x_o)
    posterior.train(check_for_convergence=False)
    samples = posterior.sample((num_samples,))
    samples = torch.as_tensor(samples, dtype=torch.float32)

    check_c2st(samples, target_samples, alg="slice_np")


@pytest.mark.parametrize("q", ("maf", "nsf", "gaussian_diag", "gaussian", "mcf", "scf"))
def test_deepcopy_support(q: str):
    """Tests if the variational does support deepcopy.

    Args:
        q: Different variational posteriors.
    """

    num_dim = 2

    class FakePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return torch.ones_like(torch.as_tensor(theta, dtype=torch.float32))

        def allow_iid_x(self) -> bool:
            return True

    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    potential_fn = FakePotential(prior=prior)
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(
        potential_fn,
        prior,
        theta_transform=theta_transform,
        q=q,
    )
    posterior_copy = deepcopy(posterior)
    posterior.set_default_x(torch.tensor(np.zeros((num_dim,)).astype(np.float32)))
    assert posterior._x != posterior_copy._x, "Mhh, something with the copy is strange"
    posterior_copy = deepcopy(posterior)
    assert (
        posterior._x == posterior_copy._x
    ).all(), "Mhh, something with the copy is strange"

    # Produces nonleaf tensors in the cache... -> Can lead to failure of deepcopy.
    posterior.q.rsample()
    posterior_copy = deepcopy(posterior)


def test_vi_posterior_inferface():
    num_dim = 2

    class FakePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return torch.ones_like(torch.as_tensor(theta[:, 0], dtype=torch.float32))

        def allow_iid_x(self) -> bool:
            return True

    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    potential_fn = FakePotential(prior=prior)
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(
        potential_fn,
        theta_transform=theta_transform,
    )
    posterior.set_default_x(torch.zeros((1, num_dim)))

    posterior2 = VIPosterior(potential_fn)

    # Raising errors if untrained
    assert isinstance(
        posterior.q.support, type(posterior2.q.support)
    ), "The support indicated by 'theta_transform' is different than that of the 'prior'."

    with pytest.raises(Exception) as execinfo:
        posterior.sample()

    assert (
        "The variational posterior was not fit" in execinfo.value.args[0]
    ), "An expected error was raised but the error message is different than expected..."

    with pytest.raises(Exception) as execinfo:
        posterior.log_prob(prior.sample())

    assert (
        "The variational posterior was not fit" in execinfo.value.args[0]
    ), "An expected error was raised but the error message is different than expected..."

    # Passing Hyperparameters in train
    posterior.train(max_num_iters=20)

    posterior.train(max_num_iters=20, optimizer=torch.optim.SGD)
    assert isinstance(
        posterior._optimizer._optimizer, torch.optim.SGD
    ), "Assert chaning the optimizer base class did not work"
    posterior.train(max_num_iters=20, stick_the_landing=True)

    assert (
        posterior._optimizer.stick_the_landing
    ), "The sticking_the_landing argument is not correctly passed."

    posterior.vi_method = "alpha"
    posterior.train(max_num_iters=20)
    posterior.train(max_num_iters=20, alpha=0.9)

    assert (
        posterior._optimizer.alpha == 0.9
    ), "The Hyperparameter alpha is not passed to the corresponding optmizer"

    posterior.vi_method = "IW"
    posterior.train(max_num_iters=20)
    posterior.train(max_num_iters=20, K=32)

    assert (
        posterior._optimizer.K == 32
    ), "The Hyperparameter K is not passed to the corresponding optmizer"

    # Passing Hyperparameters in sample
    posterior.sample()
    posterior.sample(method="sir")
    posterior.sample(method="sir", K=128)

    # Testing evaluate
    posterior.evaluate()
    posterior.evaluate("prop")
    posterior.evaluate("prop_prior")

    # Test log_prob and potential
    posterior.log_prob(posterior.sample())
    posterior.potential(posterior.sample())


def test_vi_with_multiple_independent_prior():
    prior = MultipleIndependent(
        [
            Gamma(torch.tensor([1.0]), torch.tensor([0.5])),
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),
        ],
        validate_args=False,
    )

    def simulator(theta):
        return Binomial(probs=theta[:, 1]).sample().reshape(-1, 1)

    num_simulations = 100
    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    trainer = SNLE(prior)
    nle = trainer.append_simulations(theta, x).train()
    potential, transform = likelihood_estimator_based_potential(nle, prior, x[0])
    posterior = VIPosterior(
        potential,
        prior=prior,  # type: ignore
        theta_transform=transform,
    )
    posterior.set_default_x(x[0])
    posterior.train()

    posterior.sample(
        sample_shape=(10,),
        show_progress_bars=False,
    )
