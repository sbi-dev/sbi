import pytest
import torch
import time
import warnings

from sbi.simulators.linear_gaussian import diagonal_linear_gaussian
from sbi.simulators.simutils import (
    simulate_in_batches_mp,
    simulate_in_batches,
    simulate_in_batches_joblib,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def slow_linear_gaussian(theta):
    x = []
    for th in theta:
        time.sleep(0.05)
        x.append(diagonal_linear_gaussian(th.reshape(1, -1)))

    return torch.cat(x)


@pytest.mark.parametrize("num_workers", ((5, 10)))
def test_benchmarking_mp(num_workers):

    num_simulations = 500
    theta = torch.zeros(num_simulations, 2)
    show_pbar = False
    sim_batch_size = 1

    tic = time.time()
    theta, x = simulate_in_batches_mp(
        slow_linear_gaussian,
        theta,
        sim_batch_size,
        num_workers,
        show_progressbar=show_pbar,
        worker_batch_size=50,
    )
    toc_mp = time.time() - tic
    print(toc_mp)

    tic = time.time()
    theta, x = simulate_in_batches_joblib(
        slow_linear_gaussian,
        theta,
        sim_batch_size,
        num_workers,
        show_progressbar=show_pbar,
    )
    toc_joblib = time.time() - tic

    print(toc_joblib)

    # Allow joblib to be 10 percent slower.
    assert toc_joblib <= toc_mp * 1.1


@pytest.mark.parametrize("num_simulations", ((100, 200)))
def test_benchmarking_sp(num_simulations):

    theta = torch.zeros(num_simulations, 2)
    show_pbar = False
    sim_batch_size = 1

    tic = time.time()
    theta, x = simulate_in_batches(
        slow_linear_gaussian, theta, sim_batch_size, show_progressbar=show_pbar,
    )
    toc_sp = time.time() - tic
    print(toc_sp)

    tic = time.time()
    theta, x = simulate_in_batches_joblib(
        slow_linear_gaussian, theta, sim_batch_size, 1, show_progressbar=show_pbar
    )
    toc_joblib = time.time() - tic

    print(toc_joblib)

    # Allow joblib to be 10 percent slower.
    assert toc_joblib <= toc_sp * 1.1
