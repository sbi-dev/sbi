# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import time
import warnings

import pytest
import torch

from sbi.simulators.linear_gaussian import diagonal_linear_gaussian
from sbi.simulators.simutils import simulate_in_batches

warnings.simplefilter(action="ignore", category=FutureWarning)


def slow_linear_gaussian(theta):
    x = []
    for th in theta:
        time.sleep(0.05)
        x.append(diagonal_linear_gaussian(th.reshape(1, -1)))

    return torch.cat(x)


@pytest.mark.slow
@pytest.mark.parametrize("num_workers", [10, -2])
@pytest.mark.parametrize("sim_batch_size", ((1, 10, 100)))
def test_benchmarking_sp(sim_batch_size, num_workers):

    num_simulations = 100
    theta = torch.zeros(num_simulations, 2)
    show_pbar = True

    tic = time.time()
    simulate_in_batches(
        slow_linear_gaussian,
        theta,
        sim_batch_size,
        num_workers=1,
        show_progress_bars=show_pbar,
    )
    toc_sp = time.time() - tic

    tic = time.time()
    simulate_in_batches(
        slow_linear_gaussian,
        theta,
        sim_batch_size,
        num_workers=num_workers,
        show_progress_bars=show_pbar,
    )
    toc_joblib = time.time() - tic

    # Allow joblib to be 10 percent slower.
    assert toc_joblib <= toc_sp * 1.1
