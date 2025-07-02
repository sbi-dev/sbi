import pytest
import torch

from sbi.inference.abc.distances import (
    l1_distance,
    l2_distance,
    mmd,
    mse_distance,
    wasserstein,
)


# Test the fundamental property: equal inputs should give zero distance
@pytest.mark.parametrize(
    "distance_func, identical_inputs",
    [
        # Pairwise distances (2D inputs)
        (mse_distance, (torch.ones(1, 3), torch.ones(3))),
        (l2_distance, (torch.ones(1, 3), torch.ones(3))),
        (l1_distance, (torch.ones(1, 3), torch.ones(3))),
        # Statistical distances (3D inputs for x)
        (mmd, (torch.ones(2, 3, 2), torch.ones(3, 2))),
        (wasserstein, (torch.ones(2, 3, 2), torch.ones(3, 2))),
    ],
)
def test_distance_identity_property(distance_func, identical_inputs):
    """Test that d(x, x) = 0 for all distance functions."""
    x, xo = identical_inputs
    result = distance_func(xo, x)
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


def test_distance_basic_functionality():
    """Test that distance functions work with different inputs."""
    # Test pairwise distances
    xo_pair = torch.zeros(3)
    x_pair = torch.ones(2, 3)  # Different from xo

    assert mse_distance(xo_pair, x_pair).min() > 0
    assert l2_distance(xo_pair, x_pair).min() > 0
    assert l1_distance(xo_pair, x_pair).min() > 0

    # Test statistical distances
    xo_stat = torch.zeros(2, 3)
    x_stat = torch.ones(3, 2, 3)  # Different from xo

    assert mmd(xo_stat, x_stat).min() >= 0  # MMD should be non-negative
    assert wasserstein(xo_stat, x_stat).min() >= 0


def test_distance_output_shapes():
    """Test that distance functions return correct output shapes."""
    # Test pairwise distances
    batch_size = 5
    xo = torch.zeros(3)
    x = torch.ones(batch_size, 3)

    assert mse_distance(xo, x).shape == (batch_size,)
    assert l2_distance(xo, x).shape == (batch_size,)
    assert l1_distance(xo, x).shape == (batch_size,)

    # Test statistical distances
    xo_stat = torch.zeros(2, 3)
    x_stat = torch.ones(batch_size, 2, 3)

    assert mmd(xo_stat, x_stat).shape == (batch_size,)
    assert wasserstein(xo_stat, x_stat).shape == (batch_size,)
