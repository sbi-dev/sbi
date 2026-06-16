# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Unit tests for the stateless PotentialFunction protocol and bind_observation.

These tests verify:
1. Protocol compliance for potential functions
2. bind_observation correctly binds x_o
3. IID handling (sum_iid parameter)
4. Device validation
5. Statelessness (repeated calls yield identical outputs)
"""

import pytest
import torch
from torch import Tensor

from sbi.inference.potentials.binding import BoundPotential, bind_observation
from sbi.inference.potentials.protocol import (
    validate_potential,
)


class SimplePotential:
    """Simple potential function for testing."""

    def __init__(self):
        self.device = torch.device("cpu")

    def __call__(self, theta: Tensor, x: Tensor) -> Tensor:
        return -0.5 * ((theta - x.mean()).pow(2)).sum(dim=-1)


class DeviceAwarePotential:
    """Potential with explicit device attribute."""

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, theta: Tensor, x: Tensor) -> Tensor:
        return -0.5 * ((theta - x.mean()).pow(2)).sum(dim=-1)


def stateless_potential(theta: Tensor, x: Tensor) -> Tensor:
    """Simple stateless potential function."""
    return -0.5 * ((theta - x.mean()).pow(2)).sum(dim=-1)


class TestPotentialFunctionProtocol:
    """Tests for PotentialFunction protocol."""

    def test_validate_callable_function(self):
        """Test that validate_potential accepts valid functions."""
        assert validate_potential(stateless_potential) is True

    def test_validate_callable_class(self):
        """Test that validate_potential accepts callable classes."""
        assert validate_potential(SimplePotential()) is True

    def test_validate_non_callable_raises(self):
        """Test that validate_potential rejects non-callables."""
        with pytest.raises(TypeError, match="must be callable"):
            validate_potential("not a potential")

    def test_validate_missing_call_raises(self):
        """Test that validate_potential catches missing __call__."""

        class NoCall:
            pass

        with pytest.raises(TypeError, match="must be callable"):
            validate_potential(NoCall())


class TestBindObservation:
    """Tests for bind_observation utility."""

    def test_bind_simple_potential(self):
        """Test binding a simple stateless potential."""
        x_o = torch.tensor([1.0, 2.0, 3.0])
        bound_fn = bind_observation(stateless_potential, x_o, sum_iid=True)

        theta = torch.tensor([[0.0], [1.0], [2.0]])
        result = bound_fn(theta)

        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    def test_bind_preserves_device(self):
        """Test that bound function works on correct device."""
        x_o = torch.tensor([1.0, 2.0, 3.0])
        bound_fn = bind_observation(stateless_potential, x_o, sum_iid=True)

        theta = torch.tensor([[0.0], [1.0]])
        result = bound_fn(theta)

        assert result.device == x_o.device

    def test_iid_summing(self):
        """Test that sum_iid=True correctly sums over IID observations."""
        x_o = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        def potential(theta: Tensor, x: Tensor) -> Tensor:
            return -((theta.unsqueeze(0) - x.unsqueeze(1)).pow(2)).sum(dim=-1)

        bound_fn_sum = bind_observation(potential, x_o, sum_iid=True)
        bound_fn_no_sum = bind_observation(potential, x_o, sum_iid=False)

        theta = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        result_sum = bound_fn_sum(theta)
        result_no_sum = bound_fn_no_sum(theta)

        assert result_sum.shape == (2,)
        assert result_no_sum.shape == (2, 2)

    def test_single_observation(self):
        """Test binding with single observation (no IID dimension)."""
        x_o = torch.tensor([1.0, 2.0])

        def potential(theta: Tensor, x: Tensor) -> Tensor:
            return -((theta - x).pow(2)).sum(dim=-1)

        bound_fn = bind_observation(potential, x_o, sum_iid=True)
        theta = torch.tensor([[0.0, 0.0]])
        result = bound_fn(theta)

        assert result.shape == (1,)
        assert torch.isfinite(result).all()

    def test_device_mismatch_raises(self):
        """Test that device mismatch raises clear error."""
        potential = DeviceAwarePotential(torch.device("cpu"))
        x_o = torch.tensor([1.0, 2.0, 3.0])

        if torch.cuda.is_available():
            x_o_cuda = x_o.to("cuda")
            with pytest.raises(ValueError, match="Device mismatch"):
                bind_observation(potential, x_o_cuda, sum_iid=True)


class TestBoundPotential:
    """Tests for BoundPotential class."""

    def test_bound_potential_callable(self):
        """Test that BoundPotential is callable."""
        x_o = torch.tensor([1.0, 2.0, 3.0])
        bound = BoundPotential(stateless_potential, x_o, sum_iid=True)

        theta = torch.tensor([[0.0], [1.0], [2.0]])
        result = bound(theta)

        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    def test_bound_potential_device_property(self):
        """Test device property returns correct device."""
        x_o = torch.tensor([1.0, 2.0, 3.0])
        bound = BoundPotential(stateless_potential, x_o, sum_iid=True)

        assert bound.device == x_o.device

    def test_bound_potential_to_method(self):
        """Test .to() method changes device."""
        x_o = torch.tensor([1.0, 2.0, 3.0])
        bound = BoundPotential(stateless_potential, x_o, sum_iid=True)

        if torch.cuda.is_available():
            bound_cuda = bound.to("cuda")
            assert bound_cuda.device.type == "cuda"


class TestStatelessness:
    """Tests verifying stateless behavior - key property of the new design."""

    def test_repeated_calls_same_output(self):
        """Test that repeated calls with same inputs yield identical outputs."""
        x_o = torch.tensor([1.0, 2.0, 3.0, 4.0])
        bound_fn = bind_observation(stateless_potential, x_o, sum_iid=True)

        theta = torch.tensor([[0.0], [1.0], [2.0]])

        result1 = bound_fn(theta.clone())
        result2 = bound_fn(theta.clone())
        result3 = bound_fn(theta.clone())

        assert torch.allclose(result1, result2)
        assert torch.allclose(result2, result3)

    def test_different_theta_different_output(self):
        """Test that different theta values yield different outputs."""
        x_o = torch.tensor([1.0, 2.0, 3.0])
        bound_fn = bind_observation(stateless_potential, x_o, sum_iid=True)

        theta1 = torch.tensor([[0.0]])
        theta2 = torch.tensor([[1.0]])
        theta3 = torch.tensor([[2.0]])

        result1 = bound_fn(theta1)
        result2 = bound_fn(theta2)
        result3 = bound_fn(theta3)

        assert not torch.allclose(result1, result2)
        assert not torch.allclose(result2, result3)


class TestIntegration:
    """Integration tests with realistic potential functions."""

    def test_with_prior_and_likelihood(self):
        """Test binding a potential combining prior and likelihood."""
        prior = torch.distributions.Normal(0.0, 1.0)

        class MockEstimator:
            def log_prob(self, x: Tensor, context: Tensor) -> Tensor:
                return -((x - context).pow(2)).sum(dim=-1)

        estimator = MockEstimator()

        def potential(theta: Tensor, x: Tensor) -> Tensor:
            likelihood = estimator.log_prob(x, context=theta)
            prior_log_prob = prior.log_prob(theta).sum(dim=-1)
            return likelihood + prior_log_prob

        x_o = torch.tensor([1.0, 2.0, 3.0])
        bound_fn = bind_observation(potential, x_o, sum_iid=True)

        theta = torch.randn(10, 1)
        result = bound_fn(theta)

        assert result.shape == (10,)
        assert torch.isfinite(result).all()

    def test_with_gradient_tracking(self):
        """Test that gradients can be tracked through bound potential."""
        x_o = torch.tensor([1.0, 2.0, 3.0])
        bound_fn = bind_observation(stateless_potential, x_o, sum_iid=True)

        theta = torch.randn(5, 1, requires_grad=True)
        result = bound_fn(theta)

        assert result.requires_grad
        result.sum().backward()
        assert theta.grad is not None


class TestNewStatelessAPI:
    """Tests for the new stateless potential functions (likelihood_potential, etc.)."""

    def test_likelihood_potential_creates_valid_function(self):
        """Test that likelihood_potential returns a valid PotentialFunction."""
        from sbi.inference import NLE
        from sbi.inference.potentials import bind_observation, likelihood_potential
        from sbi.utils import BoxUniform

        torch.manual_seed(42)

        prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
        theta = prior.sample((50,))
        x = theta + torch.randn_like(theta) * 0.1
        x_o = torch.randn(1, 2)

        trainer = NLE()
        estimator = trainer.append_simulations(theta, x).train(
            training_batch_size=16, max_num_epochs=2
        )

        stateless_fn = likelihood_potential(estimator, prior)

        assert hasattr(stateless_fn, "device")
        assert callable(stateless_fn)

        bound_fn = bind_observation(stateless_fn, x_o)
        theta_test = torch.rand(5, 2)
        result = bound_fn(theta_test)

        assert result.shape == (5,)
        assert torch.isfinite(result).all()

    def test_new_stateless_functions_satisfy_protocol(self):
        """Test that new stateless functions satisfy PotentialFunction protocol."""
        from sbi.inference import NLE
        from sbi.inference.potentials import likelihood_potential, validate_potential
        from sbi.utils import BoxUniform

        torch.manual_seed(42)

        prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
        theta = prior.sample((30,))
        x = theta + torch.randn_like(theta) * 0.1

        trainer = NLE()
        estimator = trainer.append_simulations(theta, x).train(
            training_batch_size=16, max_num_epochs=2
        )

        stateless_fn = likelihood_potential(estimator, prior)
        assert validate_potential(stateless_fn) is True


class TestBackwardCompatibility:
    """Tests verifying backward compatibility with existing sbi workflows."""

    def test_existing_likelihood_potential_still_works(self):
        """Test that existing likelihood_estimator_based_potential works."""
        from sbi.inference import NLE
        from sbi.inference.potentials.likelihood_based_potential import (
            likelihood_estimator_based_potential,
        )
        from sbi.utils import BoxUniform

        torch.manual_seed(42)

        prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))
        theta = prior.sample((50,))
        x = theta + torch.randn_like(theta) * 0.1
        x_o = torch.randn(1, 3)

        trainer = NLE()
        likelihood_estimator = trainer.append_simulations(theta, x).train(
            training_batch_size=16, max_num_epochs=2
        )

        potential_fn, parameter_transform = likelihood_estimator_based_potential(
            likelihood_estimator, prior, x_o
        )

        assert hasattr(potential_fn, "x_o")
        assert hasattr(potential_fn, "set_x")

        theta_test = torch.randn(5, 3)
        result = potential_fn(theta_test)
        assert result.shape == (5,)

    def test_validate_potential_with_real_potential(self):
        """Test that validate_potential works with existing potentials."""
        from sbi.inference import NLE
        from sbi.inference.potentials.likelihood_based_potential import (
            LikelihoodBasedPotential,
        )
        from sbi.utils import BoxUniform

        torch.manual_seed(42)

        prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
        theta = prior.sample((30,))
        x = theta + torch.randn_like(theta) * 0.1

        trainer = NLE()
        likelihood_estimator = trainer.append_simulations(theta, x).train(
            training_batch_size=16, max_num_epochs=2
        )

        potential = LikelihoodBasedPotential(likelihood_estimator, prior)
        potential.set_x(torch.randn(1, 2))

        assert validate_potential(potential) is True
