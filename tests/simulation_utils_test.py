# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings

import numpy as np
import pytest
import torch

from sbi.utils.simulation_utils import parallelize_simulator, simulate_from_theta

# --- Simulators for testing ---


def simple_simulator_torch(theta):
    """Simple simulator utilizing Torch operations.
    Input: Theta (N, D) or (D,)
    Output: (N, D) or (D,)
    """
    return theta * 2.0


def simple_simulator_numpy(theta):
    """Simple simulator utilizing Numpy operations.
    Input: Theta (Tensor or Array)
    Output: Array
    """
    if isinstance(theta, torch.Tensor):
        theta = theta.numpy()
    return theta * 2.0


def list_simulator(theta):
    """Simulator that returns a list (e.g. file paths)."""
    if isinstance(theta, torch.Tensor):
        # If batched, return list of strings
        if theta.ndim > 1:
            return [f"file_{t.sum()}" for t in theta]
        else:
            return f"file_{theta.sum()}"
    return f"file_{theta}"


# --- Tests for parallelize_simulator ---


def test_parallelize_simulator_decorator():
    """Test parallelize_simulator used as a decorator."""

    # 1. No arguments
    @parallelize_simulator
    def decorated_sim(theta):
        return theta * 2.0

    thetas = torch.ones(10, 2)
    # By default, decorator assumes simulator_is_batched=False, batch_size=10?
    # Checking default args in implementation:
    # simulator_is_batched = False
    # simulation_batch_size = 10
    # But if simulator_is_batched is False, and batch_size > 1,
    # it warns and runs sequentially.

    with pytest.warns(UserWarning, match="Simulation batch size is greater than 1"):
        result = decorated_sim(thetas)

    assert result.shape == (10, 2)
    assert torch.allclose(result, thetas * 2.0)

    # 2. With arguments
    @parallelize_simulator(simulation_batch_size=2, simulator_is_batched=True)
    def decorated_sim_batched(theta):
        return theta * 2.0

    result_batched = decorated_sim_batched(thetas)
    assert result_batched.shape == (10, 2)
    assert torch.allclose(result_batched, thetas * 2.0)


def test_parallelize_simulator_function():
    """Test parallelize_simulator used as a function wrapper."""
    sim = parallelize_simulator(
        simple_simulator_torch, simulation_batch_size=5, simulator_is_batched=True
    )
    thetas = torch.ones(10, 2)
    result = sim(thetas)
    assert result.shape == (10, 2)
    assert torch.allclose(result, thetas * 2.0)


def test_parallelize_simulator_numpy_recommendation_warning():
    """Test that warning suggests using Numpy."""
    thetas = torch.ones(2, 2)
    with pytest.warns(UserWarning, match="Joblib is used for parallelization"):
        wrapper = parallelize_simulator(simple_simulator_torch)
        # Warning is emitted at wrapper creation time

    # Run to ensure no error
    wrapper(thetas)


# --- Tests for simulate_from_thetas ---


@pytest.mark.parametrize("simulator", [simple_simulator_torch, simple_simulator_numpy])
@pytest.mark.parametrize("simulator_is_batched", [True, False])
@pytest.mark.parametrize("simulation_batch_size", [1, 5])
def test_simulate_from_theta_correctness(
    simulator, simulator_is_batched, simulation_batch_size
):
    """Test correctness of simulate_from_theta with various configs."""
    num_sims = 10
    dim = 2
    thetas = torch.rand(num_sims, dim)

    # Compute expected result using the simulator directly
    # ensuring we handle both numpy and torch outputs
    raw_expected = simulator(thetas)
    if isinstance(raw_expected, np.ndarray):
        expected_result_torch = torch.from_numpy(raw_expected)
    else:
        expected_result_torch = raw_expected

    # Run simulation
    # We use catch_warnings to handle the expected warnings
    # without failing or printing too much
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        theta_out, x_out = simulate_from_theta(
            simulator,
            thetas,
            simulator_is_batched=simulator_is_batched,
            simulation_batch_size=simulation_batch_size,
            show_progress_bar=False,
        )

        # Check Joblib warning presence
        assert any("Joblib is used" in str(warn.message) for warn in w)

        # Check configuration warning: if batch_size > 1 and not batched
        if simulation_batch_size > 1 and not simulator_is_batched:
            assert any(
                "Simulation batch size is greater than 1" in str(warn.message)
                for warn in w
            )

    # Check shapes
    assert theta_out.shape == (num_sims, dim)
    # output shape depends on simulator return type logic in parallelize_simulator
    # It stacks output.
    assert x_out.shape == (num_sims, dim)

    # Check values
    if isinstance(x_out, torch.Tensor):
        assert torch.allclose(x_out, expected_result_torch)
    elif isinstance(x_out, np.ndarray):
        assert np.allclose(x_out, expected_result_torch.numpy())


def test_simulate_from_thetas_list_return():
    """Test that list returns (e.g. file paths) are handled correctly."""
    thetas = torch.arange(5).float().unsqueeze(1)  # [[0], [1], [2], [3], [4]]

    # 1. Batched Simulator
    # Simulator receives batch, returns list of strings.
    # parallelize_simulator should flatten these lists.

    def list_sim_batched(t):
        return [f"path_{v.item()}" for v in t]

    theta_out, x_out = simulate_from_theta(
        list_sim_batched,
        thetas,
        simulator_is_batched=True,
        simulation_batch_size=2,
        show_progress_bar=False,
    )

    assert len(x_out) == 5
    assert x_out[0] == "path_0.0"
    assert x_out[-1] == "path_4.0"

    def list_sim_unbatched(t):
        return f"path_{t.item()}"

    theta_out2, x_out2 = simulate_from_theta(
        list_sim_unbatched,
        thetas,
        simulator_is_batched=False,
        simulation_batch_size=1,
        show_progress_bar=False,
    )

    assert len(x_out2) == 5
    assert x_out2[0] == "path_0.0"
