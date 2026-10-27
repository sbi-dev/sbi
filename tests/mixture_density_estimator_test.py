# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Unit tests for MultivariateGaussianMDN and MixtureDensityEstimator."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.distributions.transforms import ExpTransform

from sbi.neural_nets.estimators.mixture_density_estimator import (
    MixtureDensityEstimator,
    MultivariateGaussianMDN,
    ProposalCorrectedMDN,
)
from sbi.neural_nets.estimators.mog import MoG, correct_for_proposal

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def standard_mdn():
    """Standard MDN for testing (3 features, 5 context, 4 components)."""
    return MultivariateGaussianMDN(
        features=3,
        context_features=5,
        hidden_features=20,
        num_components=4,
    )


@pytest.fixture
def standard_estimator(standard_mdn):
    """Standard MixtureDensityEstimator wrapping standard_mdn."""
    return MixtureDensityEstimator(
        net=standard_mdn,
        input_shape=torch.Size([3]),
        condition_shape=torch.Size([5]),
    )


@pytest.fixture
def standard_condition():
    """Standard test condition tensor (batch_size=8, dim=5)."""
    return torch.randn(8, 5)


@pytest.fixture
def standard_inputs():
    """Standard test input tensor (batch_size=8, dim=3)."""
    return torch.randn(8, 3)


@pytest.fixture
def make_mdn():
    """Factory for creating MDN with custom dimensions."""

    def _make(features=3, context_features=5, hidden_features=20, num_components=4):
        return MultivariateGaussianMDN(
            features=features,
            context_features=context_features,
            hidden_features=hidden_features,
            num_components=num_components,
        )

    return _make


# =============================================================================
# MultivariateGaussianMDN Tests
# =============================================================================


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

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
            pytest.param(
                "mps",
                marks=pytest.mark.skipif(
                    not (
                        hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ),
                    reason="MPS not available",
                ),
            ),
        ],
    )
    def test_mdn_on_device(self, standard_mdn, device):
        """Test MDN works on different devices and migrates internal buffers."""
        mdn = standard_mdn.to(device)
        assert mdn._row_ix.device.type == device
        assert mdn._column_ix.device.type == device
        assert mdn._diag_ix.device.type == device
        assert not {"_row_ix", "_column_ix", "_diag_ix"} & set(mdn.state_dict())

        context = torch.randn(8, 5, device=device)
        mog = mdn(context)
        assert mog.device.type == device


class TestMultivariateGaussianMDNExpandComponents:
    """Test the expansion of a single component for NPE-A's final round."""

    @pytest.mark.parametrize("features", [1, 3])
    def test_expansion_keeps_the_density(self, make_mdn, features):
        """Copies with equal weights have the density of the single component."""
        mdn = make_mdn(features=features, num_components=1)
        context, inputs = torch.randn(8, 5), torch.randn(8, features)
        log_prob_before = mdn.log_prob(inputs, context)

        mdn.expand_components(4, perturbation=0.0)

        assert mdn.num_components == 4
        assert mdn.get_mixture_components(context).means.shape == (8, 4, features)
        assert torch.allclose(mdn.log_prob(inputs, context), log_prob_before, atol=1e-5)

    def test_perturbation_separates_the_copies(self, make_mdn):
        """The noise breaks the symmetry, so training can move the copies apart."""
        mdn = make_mdn(num_components=1)
        mdn.expand_components(4, perturbation=5e-3)

        means = mdn.get_mixture_components(torch.randn(8, 5)).means
        assert not torch.allclose(means[:, 0], means[:, 1])

    def test_expanded_layers_are_trained(self, make_mdn):
        """The new layers are registered, so the optimizer and gradients see them."""
        mdn = make_mdn(num_components=1)
        mdn.expand_components(4, perturbation=5e-3)

        mdn.log_prob(torch.randn(8, 3), torch.randn(8, 5)).sum().backward()

        for name in ("_logits_layer", "_means_layer", "_upper_layer"):
            layer = getattr(mdn, name)
            assert any(param is layer.weight for param in mdn.parameters())
            assert layer.weight.grad is not None

    def test_expansion_needs_a_single_component(self, standard_mdn):
        """Only one Gaussian can be copied; a mixture has no unique component."""
        with pytest.raises(ValueError, match="single component"):
            standard_mdn.expand_components(8, perturbation=0.0)


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

    def test_shared_z_score_scale_counts_every_dimension(
        self, standard_mdn, standard_estimator, standard_inputs, standard_condition
    ):
        """A z-score scale shared by all dimensions enters the log-det per dimension.

        Also covers the 0-dim buffers of estimators pickled with sbi <= 0.27.
        """
        estimator = MixtureDensityEstimator(
            net=standard_mdn,
            input_shape=torch.Size([3]),
            condition_shape=torch.Size([5]),
            transform_input=torch.tensor([1.0, 10.0]),
        )
        assert estimator._transform_scale.shape == torch.Size([3])
        inputs = standard_inputs.unsqueeze(0)
        with torch.no_grad():
            # Same net without z-scoring, plus the log-det of z = (x - 1) / 10.
            expected = standard_estimator.log_prob(
                (inputs - 1.0) / 10.0, standard_condition
            ) - 3 * torch.log(torch.tensor(10.0))
            assert torch.allclose(
                estimator.log_prob(inputs, standard_condition), expected
            )
            estimator._transform_shift = torch.tensor(1.0)
            estimator._transform_scale = torch.tensor(10.0)
            assert torch.allclose(
                estimator.log_prob(inputs, standard_condition), expected
            )


class TestMixtureDensityEstimatorGetUncorrectedMog:
    """Test MixtureDensityEstimator get_uncorrected_mog method."""

    def test_get_uncorrected_mog_returns_mog(
        self, standard_estimator, standard_condition
    ):
        """Test that get_uncorrected_mog returns a MoG object."""
        mog = standard_estimator.get_uncorrected_mog(standard_condition)

        assert isinstance(mog, MoG)
        assert mog.batch_shape == torch.Size([8])
        assert mog.num_components == 4
        assert mog.dim == 3

    def test_get_uncorrected_mog_with_embedding_net(self):
        """Test get_uncorrected_mog applies embedding network."""
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
        mog = estimator.get_uncorrected_mog(condition)

        assert isinstance(mog, MoG)
        assert mog.batch_shape == torch.Size([8])

    def test_transform_mog_maps_between_input_and_transformed_space(
        self, standard_mdn, standard_condition, standard_inputs
    ):
        """The MoG mapped to inputs has the estimator's density; the maps invert."""
        estimator = MixtureDensityEstimator(
            net=standard_mdn,
            input_shape=torch.Size([3]),
            condition_shape=torch.Size([5]),
            transform_input=torch.tensor([[1.0, -2.0, 0.5], [10.0, 0.5, 2.0]]),
        )
        with torch.no_grad():
            mog = estimator.get_uncorrected_mog(standard_condition)
            input_mog = estimator.inverse_transform_mog(mog)
            assert torch.allclose(
                input_mog.log_prob(standard_inputs),
                estimator.log_prob(standard_inputs, standard_condition),
                atol=1e-5,
            )
            # A float64 MoG, e.g. from a float64 prior, comes back as float32.
            mapped = estimator.transform_mog(input_mog.to(dtype=torch.float64))
            assert mapped.dtype == torch.float32
            for name in ("means", "precisions", "precision_factors"):
                assert torch.allclose(
                    getattr(mapped, name), getattr(mog, name), atol=1e-5
                )

    def test_mog_maps_keep_small_offsets_at_a_large_shift(self, standard_mdn):
        """With a shift far above the scale, a MoG is z-scored like the inputs, and
        the map between two equal z-scored spaces keeps its small offsets. At a
        scale whose square overflows float32, that map keeps the precisions."""
        shift, scale = torch.full((3,), 1e4), torch.full((3,), 3e-3)
        estimator = MixtureDensityEstimator(
            net=standard_mdn,
            input_shape=torch.Size([3]),
            condition_shape=torch.Size([5]),
            transform_input=torch.stack([shift, scale]),
        )
        mog = MoG.from_gaussian(shift, torch.diag(scale**2))
        assert torch.allclose(
            estimator.transform_mog(mog).means[0, 0],
            estimator._transform_input(shift.unsqueeze(0))[0],
            atol=1e-3,
        )
        z_mog = MoG.from_gaussian(torch.full((3,), 0.13), torch.eye(3))
        mapped = estimator.transform_mog_from(estimator, z_mog)
        assert torch.allclose(mapped.means, z_mog.means, atol=1e-5)

        estimator = MixtureDensityEstimator(
            net=standard_mdn,
            input_shape=torch.Size([3]),
            condition_shape=torch.Size([5]),
            transform_input=torch.stack([torch.zeros(3), torch.full((3,), 1e20)]),
        )
        mapped = estimator.transform_mog_from(estimator, z_mog)
        assert torch.allclose(mapped.precisions, z_mog.precisions)

    def test_transform_mog_from_another_estimator(self, make_mdn):
        """The direct map between two z-scored spaces equals the way through inputs."""

        def estimator(shift, scale):
            return MixtureDensityEstimator(
                net=make_mdn(),
                input_shape=torch.Size([3]),
                condition_shape=torch.Size([5]),
                transform_input=torch.stack([shift, scale]),
            )

        source = estimator(torch.tensor([1.0, -2.0, 0.5]), torch.tensor([3.0, 0.5, 2]))
        target = estimator(torch.tensor([0.0, 1.0, -1.0]), torch.tensor([1.0, 2.0, 4]))
        with torch.no_grad():
            mog = source.get_uncorrected_mog(torch.randn(4, 5))
            direct = target.transform_mog_from(source, mog)
            via_inputs = target.transform_mog(source.inverse_transform_mog(mog))
        for name in ("means", "precisions", "precision_factors"):
            assert torch.allclose(
                getattr(direct, name), getattr(via_inputs, name), atol=1e-4
            )

    @pytest.mark.parametrize(
        "map_name", ["transform_mog", "inverse_transform_mog", "transform_mog_from"]
    )
    def test_mog_maps_raise_for_non_affine_transform(self, standard_mdn, map_name):
        """The MoG maps need an affine z-score, so they raise for an unconstraining
        transform."""
        estimator = MixtureDensityEstimator(
            net=standard_mdn,
            input_shape=torch.Size([3]),
            condition_shape=torch.Size([5]),
            prior_transform=ExpTransform(),
        )
        mog = MoG.from_gaussian(torch.zeros(3), torch.eye(3))
        args = (estimator, mog) if map_name == "transform_mog_from" else (mog,)
        with pytest.raises(NotImplementedError, match="affine z-score"):
            getattr(estimator, map_name)(*args)


class TestProposalCorrectedMDN:
    """Test the NPE-A proposal correction around a MixtureDensityEstimator."""

    def test_correction_rejects_a_mixture_proposal(self, standard_estimator):
        """Dividing by a mixture gives no MoG, so both entry points raise."""
        proposal = MoG(
            logits=torch.zeros(1, 2),
            means=torch.stack([torch.zeros(3), torch.ones(3)]).unsqueeze(0),
            precisions=torch.eye(3).expand(1, 2, 3, 3).clone(),
        )
        with pytest.raises(ValueError, match="final_round"):
            ProposalCorrectedMDN(standard_estimator, proposal)
        with torch.no_grad():
            density = standard_estimator.get_uncorrected_mog(torch.randn(4, 5))
        with pytest.raises(ValueError, match="final_round"):
            correct_for_proposal(density, proposal)
