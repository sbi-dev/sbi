# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


import contextlib
from typing import Callable

import joblib
import torch
from joblib import Parallel, delayed
from torch import Tensor
from tqdm.auto import tqdm


def simulate_in_batches(
    simulator: Callable,
    theta: Tensor,
    sim_batch_size: int = 1,
    num_workers: int = 1,
    show_progress_bars: bool = True,
) -> Tensor:
    r"""
    Return simulations $x$ for parameters $\theta$ conducted batchwise.

    Parameters are batched with size `sim_batch_size` (default whole theta at once).
    Multiprocessing is used when `num_workers > 1`.

    Args:
        simulator: Simulator callable (a function or a class with `__call__`).
        theta: All parameters $\theta$ sampled from prior or posterior.
        sim_batch_size: Number of simulations per batch. Default is to simulate
            the entire theta in a single batch. When using multiple workers, increasing
            this batch size can further speed up simulations by reducing overhead.
        num_workers: Number of workers for multiprocessing.
        show_progress_bars: Whether to show a progress bar during simulation.

    Returns:
        Parameters theta and simulations $x$.
    """

    num_sims, *_ = theta.shape

    if num_sims == 0:
        x = torch.tensor([])
    elif sim_batch_size is not None and sim_batch_size < num_sims:
        # Dev note: pyright complains of torch.split lacking a type stub
        # as of PyTorch 1.4.0, see https://github.com/microsoft/pyright/issues/291
        batches = torch.split(theta, sim_batch_size, dim=0)

        if num_workers != 1:
            # Parallelize the sequence of batches across workers.
            # We use the solution proposed here: https://stackoverflow.com/a/61689175
            # to update the pbar only after the workers finished a task.
            with tqdm_joblib(
                tqdm(
                    batches,
                    disable=not show_progress_bars,
                    desc=f"Running {num_sims} simulations in {len(batches)} batches.",
                    total=len(batches),
                )
            ) as progress_bar:
                simulation_outputs = Parallel(n_jobs=num_workers)(
                    delayed(simulator)(batch) for batch in batches
                )
        else:
            pbar = tqdm(
                total=num_sims,
                disable=not show_progress_bars,
                desc=f"Running {num_sims} simulations.",
            )

            with pbar:
                simulation_outputs = []
                for batch in batches:
                    simulation_outputs.append(simulator(batch))
                    pbar.update(sim_batch_size)

        x = torch.cat(simulation_outputs, dim=0)
    else:
        x = simulator(theta)

    return x


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as
    argument

    This wrapped context manager obtains the number of finished tasks from the tqdm
    print function and uses it to update the pbar, as suggested in
    https://stackoverflow.com/a/61689175. See #419, #421
    """

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()
