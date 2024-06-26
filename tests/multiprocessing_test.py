# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import time
import warnings

import pytest
import torch

from sbi.inference import simulate_for_sbi
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian
from sbi.utils.torchutils import BoxUniform
from sbi.utils.user_input_checks import process_prior, process_simulator

warnings.simplefilter(action="ignore", category=FutureWarning)


def slow_linear_gaussian(theta):
    """Linear Gaussian simulator with a sleep statement."""
    x = []
    for th in theta:
        time.sleep(0.05)
        x.append(diagonal_linear_gaussian(th.reshape(1, -1)))

    return torch.cat(x)


@pytest.mark.slow
@pytest.mark.parametrize("num_workers", [2])
@pytest.mark.parametrize("sim_batch_size", ((1, 10, 100)))
def test_benchmarking_parallel_simulation(sim_batch_size, num_workers):
    """Test whether joblib is faster than serial processing."""
    num_simulations = 100

    prior = BoxUniform(low=torch.tensor([0]), high=torch.tensor([1]), device='cpu')
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(slow_linear_gaussian, prior, prior_returns_numpy)

    tic = time.time()
    simulate_for_sbi(
        simulator,
        proposal=prior,
        num_simulations=num_simulations,
        num_workers=1,
        simulation_batch_size=sim_batch_size,
    )
    toc_sp = time.time() - tic

    tic = time.time()
    simulate_for_sbi(
        simulator,
        proposal=prior,
        num_simulations=num_simulations,
        num_workers=num_workers,
        simulation_batch_size=sim_batch_size,
    )
    toc_joblib = time.time() - tic

    # Allow joblib to be 50 percent slower due to overhead.
    assert toc_joblib <= toc_sp * 1.5
