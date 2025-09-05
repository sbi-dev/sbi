# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# NOTE: the entire file was drafted by GPT-5 in GH Copilot, then editied by janfb.

from typing import Tuple

import pytest
import torch
from torch import Tensor, nn

from sbi.analysis.sensitivity_analysis import (
    ActiveSubspace,
    Destandardize,
    build_input_output_layer,
    destandardizing_net,
)

# ------------------------
# Fixtures and test helpers
# ------------------------


@pytest.fixture
def toy_theta_property() -> Tuple[Tensor, Tensor]:
    """Small synthetic (theta, property) dataset for regression testing."""
    n, d = 64, 3
    theta = torch.randn(n, d)
    y = theta.sum(dim=1, keepdim=True) + 0.05 * torch.randn(n, 1)
    return theta, y


class _PriorWithStats:
    """Simple Gaussian prior with mean and stddev attributes."""

    def __init__(self, d: int):
        self.mean = torch.zeros(d)
        self.stddev = torch.ones(d)

    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        return torch.randn(*shape, self.mean.numel())


class _PriorWithoutStats:
    """Simple prior without mean and stddev attributes."""

    def __init__(self, d: int):
        self._d = d

    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        return torch.randn(*shape, self._d)


class _PosteriorStub:
    """Stub posterior with simple Gaussian prior and quadratic potential."""

    def __init__(self, d: int, with_stats: bool = True):
        self._device = "cpu"
        self.d = d
        self.prior = _PriorWithStats(d) if with_stats else _PriorWithoutStats(d)

    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        return torch.randn(*shape, self.d, requires_grad=False)

    def potential(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        # Smooth, concave potential with gradient everywhere
        # Return shape (N, 1) to match regression output style
        return -(theta**2).sum(dim=1, keepdim=True) * 0.5


@pytest.fixture(
    params=[
        pytest.param(True, id="prior_with_stats"),
        pytest.param(False, id="prior_without_stats"),
    ]
)
def posterior_stub(request) -> _PosteriorStub:
    """Fixture providing a stub posterior with and without prior stats."""
    return _PosteriorStub(d=3, with_stats=request.param)


@pytest.fixture
def embedding_net_theta() -> nn.Module:
    """Small embedding net for theta."""
    return nn.Sequential(nn.Linear(3, 3), nn.ReLU())


# ------------------------
# Utility layer tests
# ------------------------


def test_destandardize_and_destandardizing_net_forward() -> None:
    """Tests destandardizing_net and Destandardize layer."""
    # Create a batch with near-zero variance in one dim to exercise min-std flooring
    n, d = 50, 2
    col0 = torch.randn(n, 1)
    col1 = torch.zeros(n, 1)
    batch = torch.cat([col0, col1], dim=1)

    min_std = 0.5
    net = destandardizing_net(batch, min_std=min_std)
    # Expected mean and clamped std
    mean = batch.mean(dim=0)
    std = batch.std(dim=0)
    std = torch.where(std < min_std, torch.full_like(std, min_std), std)

    # Check that zero maps to mean, ones maps to mean + std
    out0 = net(torch.zeros(1, d))
    out1 = net(torch.ones(1, d))
    assert torch.allclose(out0, mean.unsqueeze(0), atol=1e-6)
    assert torch.allclose(out1, (mean + std).unsqueeze(0), atol=1e-6)

    # Also test Destandardize directly
    dn = Destandardize(mean, std)
    assert torch.allclose(dn(torch.zeros(1, d)), out0)


def test_destandardizing_net_single_sample_branch() -> None:
    """Tests destandardizing_net when batch has a single row."""
    # When batch has a single row, we enter the else-branch and use t_std = 1
    batch = torch.tensor([[2.0, -3.0]], dtype=torch.float32)
    net = destandardizing_net(batch, min_std=0.5)
    # For standardized input 0, output should equal mean (the single row)
    out0 = net(torch.zeros(1, 2))
    assert torch.allclose(out0, batch, atol=1e-6)
    # For standardized input 1, output should be mean + std (std == 1 here)
    out1 = net(torch.ones(1, 2))
    assert torch.allclose(out1, batch + 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "z_theta, z_prop",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_build_input_output_layer_shapes_and_types(
    toy_theta_property, embedding_net_theta, z_theta, z_prop
) -> None:
    """Sanity check that the built input-output layer can be composed

    Args:
        toy_theta_property: Fixture providing (theta, property) data
        embedding_net_theta: Fixture providing a small embedding net for theta
        z_theta: Whether to z-score standardize theta
        z_prop: Whether to z-score standardize the property
    """
    theta, y = toy_theta_property
    inp, out = build_input_output_layer(
        batch_theta=theta,
        batch_property=y,
        z_score_theta=z_theta,
        z_score_property=z_prop,
        embedding_net_theta=embedding_net_theta,
    )
    # Compose a tiny regression head to ensure shape compatibility end-to-end
    head = nn.Linear(theta.shape[1], 1)
    model = nn.Sequential(inp, head, out)
    preds = model(theta)
    assert preds.shape == y.shape


# ------------------------
# ActiveSubspace tests
# ------------------------


@pytest.mark.parametrize("model_name", ["mlp", "resnet"])
@pytest.mark.parametrize("clip_max_norm", [None, 5.0])
def test_add_property_and_train_models(
    model_name,
    clip_max_norm,
    toy_theta_property,
    posterior_stub,
    embedding_net_theta,
) -> None:
    """Tests that add_property and train run without error for different models."""
    theta, y = toy_theta_property
    a = ActiveSubspace(posterior_stub)
    a.add_property(
        theta=theta,
        emergent_property=y,
        model=model_name,
        hidden_features=16,
        num_blocks=1,
        dropout_probability=0.1,
        z_score_theta=True,
        z_score_property=True,
        embedding_net=embedding_net_theta,
    )
    net = a.train(
        training_batch_size=16,
        learning_rate=1e-3,
        validation_fraction=0.25,
        stop_after_epochs=2,
        max_num_epochs=3,
        clip_max_norm=clip_max_norm,
    )
    assert isinstance(net, nn.Module)
    assert len(a._validation_log_probs) >= 1


def test_add_property_with_callable_model(toy_theta_property, posterior_stub) -> None:
    """Tests that add_property works with a user-defined model builder."""
    theta, y = toy_theta_property

    def builder(batch_theta: Tensor) -> nn.Module:
        d = batch_theta.shape[1]
        return nn.Sequential(nn.Identity(), nn.Linear(d, 1))

    a = ActiveSubspace(posterior_stub)
    a.add_property(theta=theta, emergent_property=y, model=builder)
    net = a.train(training_batch_size=16, stop_after_epochs=1, max_num_epochs=2)
    assert isinstance(net, nn.Module)


def test_add_property_invalid_model_raises(toy_theta_property, posterior_stub) -> None:
    """Tests that add_property raises for invalid model name."""
    theta, y = toy_theta_property
    a = ActiveSubspace(posterior_stub)
    with pytest.raises(NameError):
        a.add_property(theta=theta, emergent_property=y, model="unknown")


def test_train_reuses_existing_net(toy_theta_property, posterior_stub) -> None:
    theta, y = toy_theta_property
    a = ActiveSubspace(posterior_stub)
    a.add_property(theta=theta, emergent_property=y, model="mlp", hidden_features=8)
    _ = a.train(training_batch_size=16, stop_after_epochs=1, max_num_epochs=2)
    first_id = id(a._regression_net)
    _ = a.train(training_batch_size=16, stop_after_epochs=1, max_num_epochs=2)
    assert id(a._regression_net) == first_id  # net is reused, not rebuilt


@pytest.mark.parametrize("norm_gradients", [True, False])
def test_find_directions_with_regression_net(
    norm_gradients, toy_theta_property, posterior_stub
) -> None:
    """Tests that find_directions runs and returns correctly shaped outputs."""
    theta, y = toy_theta_property
    a = ActiveSubspace(posterior_stub)
    a.add_property(theta=theta, emergent_property=y, model="mlp", hidden_features=8)
    a.train(training_batch_size=16, stop_after_epochs=1, max_num_epochs=2)

    evals, evecs = a.find_directions(
        posterior_log_prob_as_property=False,
        norm_gradients_to_prior=norm_gradients,
        num_monte_carlo_samples=128,
    )
    d = theta.shape[1]
    assert evals.shape == (d,)
    assert evecs.shape == (d, d)
    # Ascending order
    assert torch.all(evals[1:] >= evals[:-1])
    # Columns are unit vectors
    assert torch.allclose(torch.linalg.norm(evecs, dim=0), torch.ones(d), atol=1e-5)


def test_find_directions_with_posterior_log_prob_warns(
    toy_theta_property, posterior_stub
) -> None:
    """Tests that find_directions with posterior log-prob issues a warning."""
    theta, y = toy_theta_property
    a = ActiveSubspace(posterior_stub)
    a.add_property(theta=theta, emergent_property=y, model="mlp", hidden_features=8)
    a.train(training_batch_size=16, stop_after_epochs=1, max_num_epochs=2)

    with pytest.warns(UserWarning):
        evals, evecs = a.find_directions(
            posterior_log_prob_as_property=True,
            norm_gradients_to_prior=True,
            num_monte_carlo_samples=64,
        )
    assert evals.numel() == evecs.shape[0]


def test_find_directions_raises_without_property(posterior_stub) -> None:
    """Tests that find_directions raises if no property was added."""
    a = ActiveSubspace(posterior_stub)
    with pytest.raises(ValueError):
        _ = a.find_directions(
            posterior_log_prob_as_property=False, num_monte_carlo_samples=8
        )


@pytest.mark.parametrize("norm_gradients", [True, False])
def test_project_after_find_directions(
    norm_gradients, toy_theta_property, posterior_stub
) -> None:
    """Tests that project works after find_directions and returns correct shapes."""
    theta, y = toy_theta_property
    a = ActiveSubspace(posterior_stub)
    a.add_property(theta=theta, emergent_property=y, model="mlp", hidden_features=8)
    a.train(training_batch_size=16, stop_after_epochs=1, max_num_epochs=2)
    _evals, _evecs = a.find_directions(
        posterior_log_prob_as_property=False,
        norm_gradients_to_prior=norm_gradients,
        num_monte_carlo_samples=64,
    )

    proj1 = a.project(theta, num_dimensions=1)
    proj2 = a.project(theta, num_dimensions=2)
    assert proj1.shape == (theta.shape[0], 1)
    assert proj2.shape == (theta.shape[0], 2)
    # Different dimensionality gives different projections
    assert not torch.allclose(proj1, proj2[:, :1])
