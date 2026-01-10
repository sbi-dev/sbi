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
                mean = mean1[i] if component[i] == 0 else mean2[i]
                samples[i] = torch.distributions.MultivariateNormal(mean, cov).sample()

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
        for _epoch in range(num_epochs):
            # Shuffle data
            perm = torch.randperm(num_train)
            train_context_shuffled = train_context[perm]
            train_samples_shuffled = train_samples[perm]

            for i in range(0, num_train, batch_size):
                batch_context = train_context_shuffled[i : i + batch_size]
                batch_samples = train_samples_shuffled[i : i + batch_size]

                optimizer.zero_grad()
                loss = estimator.loss(batch_samples, batch_context).mean()
                loss.backward()
                optimizer.step()

        # Evaluate on test context
        estimator.eval()
        test_context = torch.randn(1, context_features)  # Single test context

        # Get samples from trained estimator (sbi: sample_shape, batch, input)
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
        train_samples = torch.stack([
            sample_true(c.unsqueeze(0)).squeeze(0) for c in train_context
        ])

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
                batch_ctx = train_context[perm[i : i + 128]]
                batch_x = train_samples[perm[i : i + 128]]

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

    @pytest.mark.parametrize(
        "input_shape,expected_shape",
        [
            ((8, 3), (8,)),  # 2D: (batch_dim, input_dim)
            ((10, 8, 3), (10, 8)),  # 3D: (sample_dim, batch_dim, input_dim)
        ],
    )
    def test_log_prob_shape(self, input_shape, expected_shape):
        """Test log_prob output shape for different input shapes."""
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
        inputs = torch.randn(*input_shape)

        log_prob = estimator.log_prob(inputs, condition)

        assert log_prob.shape == expected_shape
        assert torch.all(torch.isfinite(log_prob))


class TestMixtureDensityEstimatorLoss:
    """Test MixtureDensityEstimator loss method."""

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

    @pytest.mark.parametrize(
        "sample_shape,expected_shape",
        [
            (torch.Size([100]), (100, 8, 3)),  # 1D sample_shape
            (torch.Size([10, 5]), (10, 5, 8, 3)),  # 2D sample_shape
        ],
    )
    def test_sample_shape(self, sample_shape, expected_shape):
        """Test sample output shape follows sbi convention."""
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
        samples = estimator.sample(sample_shape, condition)

        # sbi convention: (*sample_shape, batch_dim, *input_shape)
        assert samples.shape == expected_shape
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
        samples, log_probs = estimator.sample_and_log_prob(torch.Size([10]), condition)

        # Compute log_probs separately
        log_probs_separate = estimator.log_prob(samples, condition)

        assert samples.shape == (10, 4, 2)
        assert log_probs.shape == (10, 4)
        assert torch.allclose(log_probs, log_probs_separate)


# ============================================================================
# SNPE-A Correction Tests
# ============================================================================


class TestMixtureDensityEstimatorCorrection:
    """Test SNPE-A correction methods in MixtureDensityEstimator."""

    def test_apply_correction_invalid_prior_raises(self):
        """Test that apply_correction raises for invalid prior type."""
        mdn_net = MultivariateGaussianMDN(
            features=2,
            context_features=3,
            hidden_features=20,
            num_components=2,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([2]),
            condition_shape=torch.Size([3]),
        )

        proposal_mog = MoG(
            logits=torch.zeros(1, 2),
            means=torch.zeros(1, 2, 2),
            precisions=torch.eye(2).unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1),
        )

        # Using a Uniform distribution (not supported)
        invalid_prior = torch.distributions.Uniform(
            low=torch.zeros(2),
            high=torch.ones(2),
        )

        with pytest.raises(
            ValueError, match="must be MultivariateNormal or BoxUniform"
        ):
            estimator.apply_correction(proposal_mog, invalid_prior)


class TestSNPEACorrectionMath:
    """Regression tests for SNPE-A correction math (Eqs. 23-26)."""

    def test_correction_single_component_identity(self):
        """Test correction when proposal equals density (should return prior-weighted).

        When proposal == density, posterior = density * prior / proposal = prior.
        """
        torch.manual_seed(42)
        dim = 2

        # Create estimator
        mdn_net = MultivariateGaussianMDN(
            features=dim,
            context_features=3,
            hidden_features=20,
            num_components=1,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([dim]),
            condition_shape=torch.Size([3]),
        )

        # Get density MoG for a fixed condition
        condition = torch.randn(1, 3)
        density_mog = estimator.get_uncorrected_mog(condition)

        # Use same MoG as proposal (single component)
        proposal_mog = MoG(
            logits=density_mog.logits.clone(),
            means=density_mog.means.clone(),
            precisions=density_mog.precisions.clone(),
            precision_factors=density_mog.precision_factors.clone(),
        )

        # Prior: standard Gaussian
        prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dim),
            covariance_matrix=torch.eye(dim),
        )

        # Apply correction
        estimator.apply_correction(proposal_mog, prior)

        # Get corrected MoG
        corrected_mog = estimator.get_mog(condition)

        # With proposal == density, the posterior precision should be:
        # prec_post = prec_density - prec_proposal + prec_prior = prec_prior
        expected_precision = torch.eye(dim).unsqueeze(0).unsqueeze(0)
        assert torch.allclose(
            corrected_mog.precisions, expected_precision, atol=1e-4
        ), "Posterior precision should equal prior precision when proposal==density"


class TestBoxUniformPriorCorrection:
    """Test SNPE-A correction with BoxUniform (uniform) priors."""

    def test_apply_correction_with_boxuniform_prior(self):
        """Test that apply_correction accepts BoxUniform prior."""
        from sbi.utils.torchutils import BoxUniform

        mdn_net = MultivariateGaussianMDN(
            features=2,
            context_features=3,
            hidden_features=20,
            num_components=2,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([2]),
            condition_shape=torch.Size([3]),
        )

        proposal_mog = MoG(
            logits=torch.zeros(1, 2),
            means=torch.zeros(1, 2, 2),
            precisions=0.5
            * torch.eye(2).unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1),
        )
        prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))

        # Should not raise
        estimator.apply_correction(proposal_mog, prior)
        assert estimator.correction_applied
        assert estimator._prior_is_uniform


class TestManyComponentProposal:
    """Test SNPE-A correction with many-component proposals (typical use case)."""

    def test_correction_with_10_component_proposal(self):
        """Test correction with 10-component proposal (typical SNPE-A final round)."""
        torch.manual_seed(42)
        dim = 2
        num_proposal_components = 10

        mdn_net = MultivariateGaussianMDN(
            features=dim,
            context_features=3,
            hidden_features=20,
            num_components=2,  # Density has 2 components
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([dim]),
            condition_shape=torch.Size([3]),
        )

        # Create 10-component proposal (typical final round of SNPE-A)
        proposal_mog = MoG(
            logits=torch.randn(1, num_proposal_components),
            means=torch.randn(1, num_proposal_components, dim) * 0.1,
            precisions=0.1
            * torch.eye(dim)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_proposal_components, -1, -1),
        )
        prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dim),
            covariance_matrix=torch.eye(dim),
        )

        estimator.apply_correction(proposal_mog, prior)

        condition = torch.randn(1, 3)
        corrected_mog = estimator.get_mog(condition)

        # Posterior should have 10 * 2 = 20 components
        assert corrected_mog.num_components == num_proposal_components * 2
        assert corrected_mog.batch_shape == torch.Size([1])

        # Samples and log_prob should work
        samples = estimator.sample(torch.Size([50]), condition)
        assert samples.shape == (50, 1, dim)
        assert torch.all(torch.isfinite(samples))

        log_prob = estimator.log_prob(samples, condition)
        assert log_prob.shape == (50, 1)
        assert torch.all(torch.isfinite(log_prob))


class TestMoGPriorValidation:
    """Test validation of MoG priors (not supported)."""

    def test_mog_prior_raises(self):
        """Test that MoG prior raises ValueError (not supported)."""
        mdn_net = MultivariateGaussianMDN(
            features=2,
            context_features=3,
            hidden_features=20,
            num_components=2,
        )
        estimator = MixtureDensityEstimator(
            net=mdn_net,
            input_shape=torch.Size([2]),
            condition_shape=torch.Size([3]),
        )

        proposal_mog = MoG(
            logits=torch.zeros(1, 2),
            means=torch.zeros(1, 2, 2),
            precisions=torch.eye(2).unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1),
        )

        # MoG prior (not supported - use MultivariateNormal instead)
        mog_prior = MoG(
            logits=torch.zeros(1, 1),
            means=torch.zeros(1, 1, 2),
            precisions=torch.eye(2).unsqueeze(0).unsqueeze(0),
        )

        with pytest.raises(
            ValueError, match="must be MultivariateNormal or BoxUniform"
        ):
            estimator.apply_correction(proposal_mog, mog_prior)
