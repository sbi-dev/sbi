# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
# This test is conceived to compare the main branch before and after patching
# #1175. In the old main:
# 1. open the process manager / htop, an start monitoring the cpu usage
# 2. run the test, allow the processes to spawn correctly and take note of the
#    CPU usage/iterations per second. THERE IS NO NEED TO RUN THE TEST UNTIL THE
#    END as it does not check anything per se.
# Run this same test in the patched branch. There should be a significant
# improvement in performance, due to higher CPU saturation.

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
        x.append(diagonal_linear_gaussian(th.reshape(1, -1)))

    return torch.cat(x)


@pytest.mark.slow
@pytest.mark.parametrize("num_simulations", (1000000,))
def test_benchmarking_parallel_simulation(num_simulations):
    """Test whether joblib is faster than serial processing."""

    prior = BoxUniform(low=torch.tensor([0]), high=torch.tensor([1]), device='cpu')
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(slow_linear_gaussian, prior, prior_returns_numpy)

    tic = time.time()
    simulate_for_sbi(
        simulator,
        proposal=prior,
        num_simulations=num_simulations,
        num_workers=-1,
        simulation_batch_size=1,
    )
    toc = time.time() - tic
    print(f'Runtime: {toc - tic:.3}')
