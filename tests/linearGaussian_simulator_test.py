# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
import torch

from sbi.simulators.linear_gaussian import diagonal_linear_gaussian, linear_gaussian


@pytest.mark.parametrize("D, N", ((1, 10000), (5, 100000)))
def test_standardlinearGaussian_simulator(D: int, N: int):
    """Test linear Gaussian simulator.

    Args:
        D: parameter dimension
        N: number of samples
    """

    true_parameters = torch.zeros(D)
    num_simulations = N
    parameters = true_parameters.repeat(num_simulations).reshape(-1, D)
    xs = diagonal_linear_gaussian(parameters)

    # Check shapes.
    assert parameters.shape == torch.Size(
        (N, D)
    ), f"wrong shape of parameters: {parameters.shape} != {torch.Size((N, D))}"
    assert xs.shape == torch.Size([N, D])

    # Check mean and std.
    assert torch.allclose(
        xs.mean(axis=0), true_parameters, atol=5e-2
    ), f"Expected mean of zero, obtained {xs.mean(axis=0)}"
    assert torch.allclose(
        xs.std(axis=0), torch.ones(D), atol=5e-2
    ), f"Expected std of one, obtained {xs.std(axis=0)}"


@pytest.mark.parametrize("D, N", ((1, 10000), (5, 100000)))
def test_linearGaussian_simulator(D: int, N: int):
    """Test linear Gaussian simulator.

    Args:
        D: parameter dimension
        N: number of samples
    """

    true_parameters = torch.zeros(D)
    num_simulations = N
    parameters = true_parameters.repeat(num_simulations).reshape(-1, D)
    xs = linear_gaussian(parameters, torch.zeros(D), torch.eye(D))

    # Check shapes.
    assert parameters.shape == torch.Size(
        (N, D)
    ), f"wrong shape of parameters: {parameters.shape} != {torch.Size((N, D))}"
    assert xs.shape == torch.Size([N, D])

    # Check mean and std.
    assert torch.allclose(
        xs.mean(axis=0), true_parameters, atol=5e-2
    ), f"Expected mean of zero, obtained {xs.mean(axis=0)}"
    assert torch.allclose(
        xs.std(axis=0), torch.ones(D), atol=5e-2
    ), f"Expected std of one, obtained {xs.std(axis=0)}"
