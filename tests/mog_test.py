# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Unit tests for the MoG (Mixture of Gaussians) dataclass."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from sbi.neural_nets.estimators.mog import MoG


class TestMoGBasics:
    """Test basic MoG properties and creation."""

    def test_mog_creation(self):
        """Test that MoG can be created with valid tensors."""
        batch_size, num_components, dim = 4, 3, 2

        logits = torch.randn(batch_size, num_components)
        means = torch.randn(batch_size, num_components, dim)
        # Create valid precision matrices (positive definite)
        L = torch.randn(batch_size, num_components, dim, dim)
        precision_factors = torch.triu(L)
        # Ensure positive diagonal
        precision_factors[..., range(dim), range(dim)] = torch.abs(
            precision_factors[..., range(dim), range(dim)]
        ) + 0.1
        precisions = torch.matmul(precision_factors.transpose(-2, -1), precision_factors)

        mog = MoG(
            logits=logits,
            means=means,
            precisions=precisions,
            precision_factors=precision_factors,
        )

        assert mog.num_components == num_components
        assert mog.dim == dim
        assert mog.batch_shape == torch.Size([batch_size])

    def test_weights_sum_to_one(self):
        """Test that mixture weights sum to 1."""
        logits = torch.randn(5, 10)  # 5 batches, 10 components
        means = torch.randn(5, 10, 3)
        precisions = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(5, 10, -1, -1)

        mog = MoG(logits=logits, means=means, precisions=precisions)

        weights = mog.weights
        assert weights.shape == (5, 10)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-6)


class TestMoGLogProb:
    """Test MoG log probability computation."""

    def test_log_prob_single_component_matches_mvn(self):
        """Test that single-component MoG matches MultivariateNormal."""
        dim = 3
        batch_size = 2

        mean = torch.randn(batch_size, dim)
        # Create positive definite covariance
        L = torch.randn(batch_size, dim, dim)
        cov = torch.matmul(L, L.transpose(-2, -1)) + 0.1 * torch.eye(dim)
        precision = torch.linalg.inv(cov)

        # Create MoG from Gaussian
        mog = MoG.from_gaussian(mean, cov)

        # Sample test points
        test_points = torch.randn(batch_size, dim)

        # Compare log probs
        mog_log_prob = mog.log_prob(test_points)

        mvn_log_probs = []
        for i in range(batch_size):
            mvn = MultivariateNormal(loc=mean[i], covariance_matrix=cov[i])
            mvn_log_probs.append(mvn.log_prob(test_points[i]))
        mvn_log_prob = torch.stack(mvn_log_probs)

        assert torch.allclose(mog_log_prob, mvn_log_prob, atol=1e-4)

    @pytest.mark.parametrize("input_shape,expected_shape", [
        ((4, 2), (4,)),  # 2D input: (batch_size, dim)
        ((10, 4, 2), (10, 4)),  # 3D input: (sample_size, batch_size, dim)
    ])
    def test_log_prob_shape(self, input_shape, expected_shape):
        """Test log_prob output shape for different input shapes."""
        batch_size, num_components, dim = 4, 3, 2
        mog = _create_random_mog(batch_size, num_components, dim)

        inputs = torch.randn(*input_shape)
        log_prob = mog.log_prob(inputs)

        assert log_prob.shape == expected_shape
        assert torch.all(torch.isfinite(log_prob))


class TestMoGSample:
    """Test MoG sampling."""

    @pytest.mark.parametrize("sample_shape,expected_shape", [
        (torch.Size([]), (4, 2)),  # Default (empty) sample_shape
        (torch.Size([100]), (100, 4, 2)),  # 1D sample_shape
        (torch.Size([10, 20]), (10, 20, 4, 2)),  # 2D sample_shape
    ])
    def test_sample_shape(self, sample_shape, expected_shape):
        """Test sample shape with various sample_shape values."""
        batch_size, num_components, dim = 4, 3, 2
        mog = _create_random_mog(batch_size, num_components, dim)

        samples = mog.sample(sample_shape)
        assert samples.shape == expected_shape
        assert torch.all(torch.isfinite(samples))

    def test_sample_mean_convergence(self):
        """Test that sample mean converges to mixture mean."""
        batch_size, num_components, dim = 1, 2, 2
        num_samples = 10000

        # Create a simple MoG with known mean
        logits = torch.tensor([[0.0, 0.0]])  # Equal weights
        means = torch.tensor([[[1.0, 2.0], [-1.0, -2.0]]])  # Mean of mixture = [0, 0]
        precisions = torch.eye(dim).unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1)
        precision_factors = torch.eye(dim).unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1)

        mog = MoG(
            logits=logits,
            means=means,
            precisions=precisions,
            precision_factors=precision_factors,
        )

        samples = mog.sample(torch.Size([num_samples]))
        sample_mean = samples.mean(dim=0)

        # Expected mean is weighted average: 0.5 * [1, 2] + 0.5 * [-1, -2] = [0, 0]
        expected_mean = torch.tensor([[0.0, 0.0]])

        assert torch.allclose(sample_mean, expected_mean, atol=0.1)

    def test_sample_without_explicit_precision_factors(self):
        """Test that sampling works even without explicit precision_factors.

        Since __post_init__ now computes precision_factors from precisions,
        sampling should work without explicitly providing precision_factors.
        """
        logits = torch.randn(2, 3)
        means = torch.randn(2, 3, 4)
        precisions = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(2, 3, -1, -1)

        mog = MoG(logits=logits, means=means, precisions=precisions)

        # Should work - precision_factors computed automatically
        samples = mog.sample()
        assert samples.shape == (2, 4)
        assert mog.precision_factors is not None


class TestMoGCondition:
    """Test MoG conditioning."""

    def test_condition_reduces_dimension(self):
        """Test that conditioning reduces the dimensionality."""
        batch_size, num_components, dim = 2, 3, 4
        mog = _create_random_mog(batch_size, num_components, dim)

        condition = torch.randn(batch_size, dim)
        dims_to_sample = [0, 2]  # Keep dimensions 0 and 2

        cond_mog = mog.condition(condition, dims_to_sample)

        assert cond_mog.dim == len(dims_to_sample)
        assert cond_mog.num_components == num_components
        assert cond_mog.batch_shape == mog.batch_shape

    def test_condition_weights_sum_to_one(self):
        """Test that conditioned MoG weights still sum to 1."""
        mog = _create_random_mog(batch_size=3, num_components=5, dim=4)
        condition = torch.randn(3, 4)
        dims_to_sample = [1, 3]

        cond_mog = mog.condition(condition, dims_to_sample)

        assert torch.allclose(cond_mog.weights.sum(dim=-1), torch.ones(3), atol=1e-6)


class TestMoGFromGaussian:
    """Test MoG.from_gaussian class method."""

    def test_from_gaussian_creates_single_component(self):
        """Test that from_gaussian creates a single-component MoG."""
        mean = torch.randn(3)
        cov = torch.eye(3)

        mog = MoG.from_gaussian(mean, cov)

        assert mog.num_components == 1
        assert mog.dim == 3
        assert mog.batch_shape == torch.Size([1])

    def test_from_gaussian_batched(self):
        """Test from_gaussian with batched inputs."""
        batch_size, dim = 5, 3
        mean = torch.randn(batch_size, dim)
        cov = torch.eye(dim).unsqueeze(0).expand(batch_size, -1, -1)

        mog = MoG.from_gaussian(mean, cov)

        assert mog.num_components == 1
        assert mog.dim == dim
        assert mog.batch_shape == torch.Size([batch_size])

    def test_from_gaussian_precision_is_inverse_of_cov(self):
        """Test that precision is the inverse of covariance."""
        mean = torch.randn(2)
        L = torch.randn(2, 2)
        cov = torch.matmul(L, L.T) + 0.5 * torch.eye(2)

        mog = MoG.from_gaussian(mean, cov)

        # precision should be inverse of covariance
        expected_precision = torch.linalg.inv(cov)
        actual_precision = mog.precisions[0, 0]  # (batch=1, component=1, dim, dim)

        assert torch.allclose(actual_precision, expected_precision, atol=1e-5)


class TestMoGDeviceAndDetach:
    """Test MoG device transfer and detach."""

    def test_detach(self):
        """Test detaching MoG from computation graph."""
        logits = torch.randn(2, 3, requires_grad=True)
        means = torch.randn(2, 3, 4, requires_grad=True)
        precisions = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(2, 3, -1, -1)

        mog = MoG(logits=logits, means=means, precisions=precisions)
        detached = mog.detach()

        assert not detached.logits.requires_grad
        assert not detached.means.requires_grad


# Helper function for creating random valid MoG
def _create_random_mog(
    batch_size: int,
    num_components: int,
    dim: int,
    device: torch.device = torch.device("cpu"),
) -> MoG:
    """Create a random valid MoG for testing."""
    logits = torch.randn(batch_size, num_components, device=device)
    means = torch.randn(batch_size, num_components, dim, device=device)

    # Create valid precision factors (upper triangular with positive diagonal)
    precision_factors = torch.zeros(
        batch_size, num_components, dim, dim, device=device
    )
    # Fill upper triangular
    for i in range(dim):
        for j in range(i, dim):
            if i == j:
                # Diagonal: positive values
                precision_factors[..., i, j] = torch.abs(
                    torch.randn(batch_size, num_components, device=device)
                ) + 0.5
            else:
                # Off-diagonal: any values
                precision_factors[..., i, j] = torch.randn(
                    batch_size, num_components, device=device
                ) * 0.1

    precisions = torch.matmul(
        precision_factors.transpose(-2, -1), precision_factors
    )

    return MoG(
        logits=logits,
        means=means,
        precisions=precisions,
        precision_factors=precision_factors,
    )


class TestMoGNumericalStability:
    """Test MoG numerical stability with ill-conditioned matrices."""

    def test_ill_conditioned_precision_with_epsilon_stabilization(self):
        """Test that MoG handles ill-conditioned precision matrices via epsilon."""
        batch_size, num_components, dim = 2, 3, 4

        # Create an ill-conditioned precision matrix
        # (one eigenvalue much smaller than others)
        logits = torch.randn(batch_size, num_components)
        means = torch.randn(batch_size, num_components, dim)

        # Create nearly singular precision (condition number ~1e6)
        eigenvalues = torch.tensor([1e-6, 0.1, 1.0, 10.0])
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))  # Random orthogonal matrix
        base_precision = Q @ torch.diag(eigenvalues) @ Q.T

        precisions = base_precision.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_components, -1, -1
        ).clone()

        # MoG should handle this via epsilon stabilization in Cholesky
        mog = MoG(logits=logits, means=means, precisions=precisions)

        # Should be able to compute log_prob and sample without errors
        test_samples = torch.randn(10, batch_size, dim)
        log_probs = mog.log_prob(test_samples)
        assert torch.all(torch.isfinite(log_probs))

        samples = mog.sample((5,))
        assert torch.all(torch.isfinite(samples))

    def test_nan_logits_rejected(self):
        """Test that NaN values in logits are rejected."""
        batch_size, num_components, dim = 2, 3, 4

        logits = torch.randn(batch_size, num_components)
        logits[0, 0] = float("nan")
        means = torch.randn(batch_size, num_components, dim)
        precisions = torch.eye(dim).unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_components, -1, -1
        )

        with pytest.raises(ValueError, match="logits contains NaN"):
            MoG(logits=logits, means=means, precisions=precisions)

    def test_inf_means_rejected(self):
        """Test that Inf values in means are rejected."""
        batch_size, num_components, dim = 2, 3, 4

        logits = torch.randn(batch_size, num_components)
        means = torch.randn(batch_size, num_components, dim)
        means[0, 0, 0] = float("inf")
        precisions = torch.eye(dim).unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_components, -1, -1
        )

        with pytest.raises(ValueError, match="means contains NaN"):
            MoG(logits=logits, means=means, precisions=precisions)

    def test_nan_precisions_rejected(self):
        """Test that NaN values in precisions are rejected."""
        batch_size, num_components, dim = 2, 3, 4

        logits = torch.randn(batch_size, num_components)
        means = torch.randn(batch_size, num_components, dim)
        precisions = torch.eye(dim).unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_components, -1, -1
        ).clone()
        precisions[0, 0, 0, 0] = float("nan")

        with pytest.raises(ValueError, match="precisions contains NaN"):
            MoG(logits=logits, means=means, precisions=precisions)

    def test_non_positive_definite_precision_rejected(self):
        """Test that non-positive-definite precisions are rejected."""
        batch_size, num_components, dim = 2, 3, 4

        logits = torch.randn(batch_size, num_components)
        means = torch.randn(batch_size, num_components, dim)

        # Create a non-positive-definite matrix (negative eigenvalue)
        eigenvalues = torch.tensor([-1.0, 0.1, 1.0, 10.0])
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))
        bad_precision = Q @ torch.diag(eigenvalues) @ Q.T

        precisions = bad_precision.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_components, -1, -1
        ).clone()

        with pytest.raises(ValueError, match="Cholesky decomposition"):
            MoG(logits=logits, means=means, precisions=precisions)

    def test_conditioning_with_near_singular_precision(self):
        """Test that conditioning works with near-singular precisions."""
        batch_size, num_components, dim = 1, 2, 4

        logits = torch.zeros(batch_size, num_components)
        means = torch.zeros(batch_size, num_components, dim)

        # Create a precision matrix that is nearly singular in one block
        # but well-conditioned overall
        precision = torch.eye(dim) * 1.0
        precision[0, 0] = 1e-4  # Small but not zero
        precisions = precision.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_components, -1, -1
        ).clone()

        mog = MoG(logits=logits, means=means, precisions=precisions)

        # Condition on dimensions 2 and 3 (keep dimensions 0 and 1 free)
        # dims_to_sample specifies which dimensions to KEEP (not condition on)
        dims_to_sample = [0, 1]
        # condition tensor has values for ALL dimensions, free dims will be ignored
        condition_values = torch.zeros(batch_size, dim)

        conditioned_mog = mog.condition(condition_values, dims_to_sample)

        # Verify results are finite
        assert torch.all(torch.isfinite(conditioned_mog.logits))
        assert torch.all(torch.isfinite(conditioned_mog.means))
        assert torch.all(torch.isfinite(conditioned_mog.precisions))
