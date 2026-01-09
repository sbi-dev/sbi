# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Unit tests for MultivariateGaussianMDN and MixtureDensityEstimator."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sbi.neural_nets.estimators.mixture_density_estimator import (
    MixtureDensityEstimator,
    MultivariateGaussianMDN,
)
from sbi.neural_nets.estimators.mog import MoG


class TestMultivariateGaussianMDNCreation:
    """Test MDN creation and initialization."""

    def test_mdn_creation_default_hidden_net(self):
        """Test MDN creation with default hidden network."""
        mdn = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=50,
            num_components=10,
        )

        assert mdn.features == 3
        assert mdn.context_features == 5
        assert mdn.num_components == 10
        assert mdn.hidden_features == 50

    def test_mdn_creation_custom_hidden_net(self):
        """Test MDN creation with custom hidden network."""
        hidden_net = nn.Sequential(
            nn.Linear(5, 50),
            nn.ReLU(),
        )

        mdn = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=50,
            num_components=10,
            hidden_net=hidden_net,
        )

        assert mdn.features == 3
        assert mdn.num_components == 10

    def test_mdn_creation_mismatched_hidden_net_raises(self):
        """Test that mismatched hidden_net dimensions raise error."""
        hidden_net = nn.Sequential(
            nn.Linear(5, 30),  # Outputs 30, but we say hidden_features=50
            nn.ReLU(),
        )

        with pytest.raises(ValueError, match="does not match"):
            MultivariateGaussianMDN(
                features=3,
                context_features=5,
                hidden_features=50,  # Mismatch!
                num_components=10,
                hidden_net=hidden_net,
            )

    def test_mdn_custom_initialization(self):
        """Test MDN with custom initialization."""
        mdn = MultivariateGaussianMDN(
            features=2,
            context_features=3,
            hidden_features=20,
            num_components=5,
            custom_initialization=True,
        )

        # Should create without error
        context = torch.randn(4, 3)
        mog = mdn(context)
        assert mog.num_components == 5


class TestMultivariateGaussianMDNForward:
    """Test MDN forward pass."""

    def test_forward_returns_mog(self):
        """Test that forward returns a MoG object."""
        mdn = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        )

        context = torch.randn(8, 5)
        mog = mdn(context)

        assert isinstance(mog, MoG)
        assert mog.batch_shape == torch.Size([8])
        assert mog.num_components == 4
        assert mog.dim == 3

    def test_forward_shapes(self):
        """Test output tensor shapes."""
        batch_size = 16
        features = 4
        context_features = 10
        num_components = 6

        mdn = MultivariateGaussianMDN(
            features=features,
            context_features=context_features,
            hidden_features=32,
            num_components=num_components,
        )

        context = torch.randn(batch_size, context_features)
        mog = mdn(context)

        assert mog.logits.shape == (batch_size, num_components)
        assert mog.means.shape == (batch_size, num_components, features)
        assert mog.precisions.shape == (batch_size, num_components, features, features)
        assert mog.precision_factors.shape == (
            batch_size, num_components, features, features
        )

    def test_precisions_positive_definite(self):
        """Test that output precisions are positive definite."""
        mdn = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        )

        context = torch.randn(8, 5)
        mog = mdn(context)

        # Check positive definiteness via eigenvalues
        for b in range(8):
            for k in range(4):
                prec = mog.precisions[b, k]
                eigenvalues = torch.linalg.eigvalsh(prec)
                assert torch.all(eigenvalues > 0), "Precision not positive definite"

    def test_precision_factors_upper_triangular(self):
        """Test that precision factors are upper triangular."""
        mdn = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        )

        context = torch.randn(8, 5)
        mog = mdn(context)

        # Check upper triangular structure
        precfs = mog.precision_factors
        lower_mask = torch.tril(torch.ones(3, 3), diagonal=-1).bool()

        for b in range(8):
            for k in range(4):
                lower_elements = precfs[b, k][lower_mask]
                assert torch.allclose(
                    lower_elements, torch.zeros_like(lower_elements)
                ), "Precision factor not upper triangular"


class TestMultivariateGaussianMDNLogProb:
    """Test MDN log probability computation."""

    def test_log_prob_shape(self):
        """Test log_prob output shape."""
        batch_size = 8
        features = 3
        context_features = 5

        mdn = MultivariateGaussianMDN(
            features=features,
            context_features=context_features,
            hidden_features=20,
            num_components=4,
        )

        context = torch.randn(batch_size, context_features)
        inputs = torch.randn(batch_size, features)

        log_prob = mdn.log_prob(inputs, context)

        assert log_prob.shape == (batch_size,)

    def test_log_prob_finite(self):
        """Test that log_prob returns finite values."""
        mdn = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        )

        context = torch.randn(8, 5)
        inputs = torch.randn(8, 3)

        log_prob = mdn.log_prob(inputs, context)

        assert torch.all(torch.isfinite(log_prob))

    def test_log_prob_gradient_flows(self):
        """Test that gradients flow through log_prob."""
        mdn = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        )

        context = torch.randn(8, 5)
        inputs = torch.randn(8, 3)

        log_prob = mdn.log_prob(inputs, context)
        loss = -log_prob.mean()
        loss.backward()

        # Check that gradients exist
        for param in mdn.parameters():
            assert param.grad is not None


class TestMultivariateGaussianMDNSample:
    """Test MDN sampling."""

    def test_sample_shape(self):
        """Test sample output shape."""
        batch_size = 8
        features = 3
        context_features = 5
        num_samples = 100

        mdn = MultivariateGaussianMDN(
            features=features,
            context_features=context_features,
            hidden_features=20,
            num_components=4,
        )

        context = torch.randn(batch_size, context_features)
        samples = mdn.sample(torch.Size([num_samples]), context)

        # MDN.sample returns (*sample_shape, batch_size, features) - same as MoG
        assert samples.shape == (num_samples, batch_size, features)

    def test_sample_finite(self):
        """Test that samples are finite."""
        mdn = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        )

        context = torch.randn(8, 5)
        samples = mdn.sample(torch.Size([100]), context)

        assert torch.all(torch.isfinite(samples))


class TestMultivariateGaussianMDNDimensions:
    """Test MDN with various dimensions."""

    @pytest.mark.parametrize("features", [1, 2, 5, 10])
    def test_different_feature_dims(self, features):
        """Test MDN with different output dimensions."""
        mdn = MultivariateGaussianMDN(
            features=features,
            context_features=5,
            hidden_features=20,
            num_components=3,
        )

        context = torch.randn(4, 5)
        mog = mdn(context)

        assert mog.dim == features
        # Test log_prob and sample work
        inputs = torch.randn(4, features)
        log_prob = mdn.log_prob(inputs, context)
        samples = mdn.sample(torch.Size([10]), context)

        assert log_prob.shape == (4,)
        assert samples.shape == (10, 4, features)  # (*sample_shape, batch, features)

    @pytest.mark.parametrize("num_components", [1, 2, 5, 20])
    def test_different_num_components(self, num_components):
        """Test MDN with different number of components."""
        mdn = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=num_components,
        )

        context = torch.randn(4, 5)
        mog = mdn(context)

        assert mog.num_components == num_components


class TestMultivariateGaussianMDNDevice:
    """Test MDN device handling."""

    def test_mdn_on_cpu(self):
        """Test MDN works on CPU."""
        mdn = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        )

        context = torch.randn(8, 5)
        mog = mdn(context)

        assert mog.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mdn_on_cuda(self):
        """Test MDN works on CUDA."""
        mdn = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        ).cuda()

        context = torch.randn(8, 5).cuda()
        mog = mdn(context)

        assert mog.device.type == "cuda"


class TestMixtureDensityEstimatorFunctional:
    """Functional tests for MixtureDensityEstimator accuracy.

    These tests validate the full stack: MixtureDensityEstimator wrapping
    MultivariateGaussianMDN, using the standard sbi training interface.
    """

    @pytest.mark.slow
    def test_estimator_learns_simple_mog(self):
        """Test that estimator can learn a simple 2-component MoG distribution.

        This test creates a ground truth 2-component MoG where the mixture
        weights and means depend on the context. The estimator is trained to learn
        this conditional distribution and validated using c2st.
        """
        from sbi.utils.metrics import c2st

        torch.manual_seed(42)

        # Problem setup
        features = 2  # Output dimension
        context_features = 3  # Input dimension
        num_components = 2
        num_train = 5000
        num_test_samples = 1000

        # Generate training data from a context-dependent MoG
        # The MoG parameters depend linearly on the context
        def sample_from_true_mog(context: torch.Tensor) -> torch.Tensor:
            """Sample from ground truth MoG given context."""
            batch_size = context.shape[0]

            # Context-dependent mixture weights (via softmax of linear transform)
            weight_transform = torch.tensor([[0.5, -0.5], [0.3, -0.3], [0.2, -0.2]])
            logits = context @ weight_transform  # (batch, 2)
            weights = torch.softmax(logits, dim=-1)

            # Context-dependent means
            # Component 1: mean shifts with first context dim
            # Component 2: mean shifts with second context dim
            mean1 = torch.stack([context[:, 0], context[:, 1]], dim=-1)
            mean2 = torch.stack([-context[:, 0], -context[:, 1]], dim=-1)

            # Fixed covariances (identity scaled)
            cov = 0.3 * torch.eye(features)

            # Sample component assignments
            component = torch.multinomial(weights, num_samples=1).squeeze(-1)

            # Sample from chosen component
            samples = torch.zeros(batch_size, features)
            for i in range(batch_size):
                if component[i] == 0:
                    mean = mean1[i]
                else:
                    mean = mean2[i]
                samples[i] = torch.distributions.MultivariateNormal(
                    mean, cov
                ).sample()

            return samples

        # Generate training data
        train_context = torch.randn(num_train, context_features)
        train_samples = sample_from_true_mog(train_context)

        # Create MixtureDensityEstimator (full stack)
        mdn_net = MultivariateGaussianMDN(
            features=features,
            context_features=context_features,
            hidden_features=64,
            num_components=num_components,
            num_hidden_layers=2,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([features]),
            condition_shape=torch.Size([context_features]),
        )

        optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)

        # Training loop using estimator.loss()
        batch_size = 256
        num_epochs = 100

        estimator.train()
        for epoch in range(num_epochs):
            # Shuffle data
            perm = torch.randperm(num_train)
            train_context_shuffled = train_context[perm]
            train_samples_shuffled = train_samples[perm]

            for i in range(0, num_train, batch_size):
                batch_context = train_context_shuffled[i:i + batch_size]
                batch_samples = train_samples_shuffled[i:i + batch_size]

                optimizer.zero_grad()
                loss = estimator.loss(batch_samples, batch_context).mean()
                loss.backward()
                optimizer.step()

        # Evaluate on test context
        estimator.eval()
        test_context = torch.randn(1, context_features)  # Single test context

        # Get samples from trained estimator (sbi convention: sample_shape, batch, input)
        with torch.no_grad():
            samples = estimator.sample(torch.Size([num_test_samples]), test_context)
            # Shape: (num_test_samples, 1, features) -> squeeze batch dim
            estimator_samples = samples.squeeze(1)

        # Get samples from true distribution
        test_context_expanded = test_context.expand(num_test_samples, -1)
        true_samples = sample_from_true_mog(test_context_expanded)

        # Compare using c2st
        c2st_score = c2st(estimator_samples, true_samples).item()

        # c2st should be close to 0.5 (chance level) if distributions match
        assert 0.4 <= c2st_score <= 0.6, (
            f"Estimator c2st={c2st_score:.3f} is too far from chance level (0.5). "
            "The estimator may not have learned the true distribution accurately."
        )

    def test_estimator_learns_simple_gaussian(self):
        """Test that estimator can learn a simple Gaussian (single component).

        This is a simpler test that verifies basic learning capability.
        """
        from sbi.utils.metrics import c2st

        torch.manual_seed(123)

        features = 2
        context_features = 2
        num_train = 2000
        num_test_samples = 500

        # Ground truth: Gaussian with context-dependent mean
        def sample_true(context: torch.Tensor) -> torch.Tensor:
            mean = context  # Mean equals context
            cov = 0.2 * torch.eye(features)
            return torch.distributions.MultivariateNormal(mean, cov).sample()

        # Generate training data
        train_context = torch.randn(num_train, context_features)
        train_samples = torch.stack([sample_true(c.unsqueeze(0)).squeeze(0)
                                     for c in train_context])

        # Create MixtureDensityEstimator with single component
        mdn_net = MultivariateGaussianMDN(
            features=features,
            context_features=context_features,
            hidden_features=32,
            num_components=1,  # Single component for Gaussian
            num_hidden_layers=2,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([features]),
            condition_shape=torch.Size([context_features]),
        )

        optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)

        estimator.train()
        for _ in range(50):
            perm = torch.randperm(num_train)
            for i in range(0, num_train, 128):
                batch_ctx = train_context[perm[i:i + 128]]
                batch_x = train_samples[perm[i:i + 128]]

                optimizer.zero_grad()
                loss = estimator.loss(batch_x, batch_ctx).mean()
                loss.backward()
                optimizer.step()

        # Test
        estimator.eval()
        test_ctx = torch.tensor([[0.5, -0.5]])

        with torch.no_grad():
            samples = estimator.sample(torch.Size([num_test_samples]), test_ctx)
            # Shape: (num_test_samples, 1, features) -> squeeze batch dim
            estimator_samples = samples.squeeze(1)

        true_samples = torch.stack([
            sample_true(test_ctx).squeeze(0) for _ in range(num_test_samples)
        ])

        c2st_score = c2st(estimator_samples, true_samples).item()

        assert 0.35 <= c2st_score <= 0.65, (
            f"Estimator c2st={c2st_score:.3f} indicates poor learning."
        )


# ============================================================================
# MixtureDensityEstimator Tests
# ============================================================================


class TestMixtureDensityEstimatorCreation:
    """Test MixtureDensityEstimator creation and initialization."""

    def test_estimator_creation_basic(self):
        """Test basic estimator creation."""
        mdn_net = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([3]),
            condition_shape=torch.Size([5]),
        )

        assert estimator.input_shape == torch.Size([3])
        assert estimator.condition_shape == torch.Size([5])

    def test_estimator_creation_with_embedding_net(self):
        """Test estimator creation with embedding network."""
        embedding_net = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        mdn_net = MultivariateGaussianMDN(
            features=3,
            context_features=5,  # Must match embedding output
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([3]),
            condition_shape=torch.Size([10]),  # Raw condition dim
            embedding_net=embedding_net,
        )

        assert estimator.embedding_net is embedding_net
        assert estimator.condition_shape == torch.Size([10])

    def test_estimator_creation_mismatched_embedding_raises(self):
        """Test that mismatched embedding dimensions raise error."""
        embedding_net = nn.Sequential(
            nn.Linear(10, 7),  # Outputs 7
            nn.ReLU(),
        )
        mdn_net = MultivariateGaussianMDN(
            features=3,
            context_features=5,  # Expects 5
            hidden_features=20,
            num_components=4,
        )

        with pytest.raises(ValueError, match="embedding_net output dimension"):
            MixtureDensityEstimator(
                net=mdn_net,
                input_shape=torch.Size([3]),
                condition_shape=torch.Size([10]),
                embedding_net=embedding_net,
            )


class TestMixtureDensityEstimatorLogProb:
    """Test MixtureDensityEstimator log_prob method."""

    def test_log_prob_shape_2d_input(self):
        """Test log_prob with 2D input (batch_dim, input_dim)."""
        batch_size = 8
        input_dim = 3
        condition_dim = 5

        mdn_net = MultivariateGaussianMDN(
            features=input_dim,
            context_features=condition_dim,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([input_dim]),
            condition_shape=torch.Size([condition_dim]),
        )

        condition = torch.randn(batch_size, condition_dim)
        inputs = torch.randn(batch_size, input_dim)

        log_prob = estimator.log_prob(inputs, condition)

        assert log_prob.shape == (batch_size,)
        assert torch.all(torch.isfinite(log_prob))

    def test_log_prob_shape_3d_input(self):
        """Test log_prob with 3D input (sample_dim, batch_dim, input_dim)."""
        sample_dim = 10
        batch_size = 8
        input_dim = 3
        condition_dim = 5

        mdn_net = MultivariateGaussianMDN(
            features=input_dim,
            context_features=condition_dim,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([input_dim]),
            condition_shape=torch.Size([condition_dim]),
        )

        condition = torch.randn(batch_size, condition_dim)
        inputs = torch.randn(sample_dim, batch_size, input_dim)

        log_prob = estimator.log_prob(inputs, condition)

        assert log_prob.shape == (sample_dim, batch_size)
        assert torch.all(torch.isfinite(log_prob))

    def test_log_prob_with_embedding_net(self):
        """Test log_prob with embedding network."""
        batch_size = 8
        input_dim = 3
        raw_condition_dim = 10
        embedded_dim = 5

        embedding_net = nn.Linear(raw_condition_dim, embedded_dim)
        mdn_net = MultivariateGaussianMDN(
            features=input_dim,
            context_features=embedded_dim,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([input_dim]),
            condition_shape=torch.Size([raw_condition_dim]),
            embedding_net=embedding_net,
        )

        condition = torch.randn(batch_size, raw_condition_dim)
        inputs = torch.randn(batch_size, input_dim)

        log_prob = estimator.log_prob(inputs, condition)

        assert log_prob.shape == (batch_size,)


class TestMixtureDensityEstimatorLoss:
    """Test MixtureDensityEstimator loss method."""

    def test_loss_shape(self):
        """Test that loss returns correct shape."""
        batch_size = 16
        input_dim = 3
        condition_dim = 5

        mdn_net = MultivariateGaussianMDN(
            features=input_dim,
            context_features=condition_dim,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([input_dim]),
            condition_shape=torch.Size([condition_dim]),
        )

        condition = torch.randn(batch_size, condition_dim)
        inputs = torch.randn(batch_size, input_dim)

        loss = estimator.loss(inputs, condition)

        assert loss.shape == (batch_size,)
        assert torch.all(torch.isfinite(loss))

    def test_loss_equals_negative_log_prob(self):
        """Test that loss equals negative log probability."""
        mdn_net = MultivariateGaussianMDN(
            features=2,
            context_features=3,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([2]),
            condition_shape=torch.Size([3]),
        )

        condition = torch.randn(8, 3)
        inputs = torch.randn(8, 2)

        loss = estimator.loss(inputs, condition)
        log_prob = estimator.log_prob(inputs, condition)

        assert torch.allclose(loss, -log_prob)

    def test_loss_gradient_flows(self):
        """Test that gradients flow through loss."""
        mdn_net = MultivariateGaussianMDN(
            features=2,
            context_features=3,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([2]),
            condition_shape=torch.Size([3]),
        )

        condition = torch.randn(8, 3)
        inputs = torch.randn(8, 2)

        loss = estimator.loss(inputs, condition).mean()
        loss.backward()

        # Check gradients exist on MDN parameters
        for param in estimator.net.parameters():
            assert param.grad is not None


class TestMixtureDensityEstimatorSample:
    """Test MixtureDensityEstimator sample method."""

    def test_sample_shape(self):
        """Test sample output shape follows sbi convention."""
        batch_size = 8
        input_dim = 3
        condition_dim = 5
        num_samples = 100

        mdn_net = MultivariateGaussianMDN(
            features=input_dim,
            context_features=condition_dim,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([input_dim]),
            condition_shape=torch.Size([condition_dim]),
        )

        condition = torch.randn(batch_size, condition_dim)
        samples = estimator.sample(torch.Size([num_samples]), condition)

        # sbi convention: (*sample_shape, batch_dim, *input_shape)
        assert samples.shape == (num_samples, batch_size, input_dim)

    def test_sample_shape_multi_dim(self):
        """Test sample with multi-dimensional sample_shape."""
        batch_size = 4
        input_dim = 2
        condition_dim = 3

        mdn_net = MultivariateGaussianMDN(
            features=input_dim,
            context_features=condition_dim,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([input_dim]),
            condition_shape=torch.Size([condition_dim]),
        )

        condition = torch.randn(batch_size, condition_dim)
        samples = estimator.sample(torch.Size([10, 5]), condition)

        # sbi convention: (*sample_shape, batch_dim, *input_shape)
        assert samples.shape == (10, 5, batch_size, input_dim)

    def test_sample_finite(self):
        """Test that samples are finite."""
        mdn_net = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([3]),
            condition_shape=torch.Size([5]),
        )

        condition = torch.randn(8, 5)
        samples = estimator.sample(torch.Size([100]), condition)

        assert torch.all(torch.isfinite(samples))


class TestMixtureDensityEstimatorGetMog:
    """Test MixtureDensityEstimator get_mog method."""

    def test_get_mog_returns_mog(self):
        """Test that get_mog returns a MoG object."""
        mdn_net = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([3]),
            condition_shape=torch.Size([5]),
        )

        condition = torch.randn(8, 5)
        mog = estimator.get_mog(condition)

        assert isinstance(mog, MoG)
        assert mog.batch_shape == torch.Size([8])
        assert mog.num_components == 4
        assert mog.dim == 3

    def test_get_mog_with_embedding_net(self):
        """Test get_mog applies embedding network."""
        embedding_net = nn.Linear(10, 5)
        mdn_net = MultivariateGaussianMDN(
            features=3,
            context_features=5,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([3]),
            condition_shape=torch.Size([10]),
            embedding_net=embedding_net,
        )

        # Should accept raw condition shape
        condition = torch.randn(8, 10)
        mog = estimator.get_mog(condition)

        assert isinstance(mog, MoG)
        assert mog.batch_shape == torch.Size([8])


class TestMixtureDensityEstimatorSampleAndLogProb:
    """Test MixtureDensityEstimator sample_and_log_prob method."""

    def test_sample_and_log_prob_consistency(self):
        """Test that sample_and_log_prob is consistent with separate calls."""
        mdn_net = MultivariateGaussianMDN(
            features=2,
            context_features=3,
            hidden_features=20,
            num_components=4,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([2]),
            condition_shape=torch.Size([3]),
        )

        torch.manual_seed(42)
        condition = torch.randn(4, 3)

        # Get samples and log_probs in one call
        torch.manual_seed(123)
        samples, log_probs = estimator.sample_and_log_prob(
            torch.Size([10]), condition
        )

        # Compute log_probs separately
        log_probs_separate = estimator.log_prob(samples, condition)

        assert samples.shape == (10, 4, 2)
        assert log_probs.shape == (10, 4)
        assert torch.allclose(log_probs, log_probs_separate)
