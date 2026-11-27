# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import contextlib
from typing import Callable, Optional

import joblib
import torch
from torch import Tensor

from sbi.utils.simulation_utils import simulate_from_theta


def simulate_in_batches(
    simulator: Callable[[Tensor], Tensor],
    theta: Tensor,
    sim_batch_size: Optional[int] = 1,
    num_workers: int = 1,
    seed: Optional[int] = None,
    show_progress_bars: bool = True,
) -> Tensor:
    r"""
    Return simulations $x$ for parameters $\theta$ conducted batchwise.

    Runs `simulate_from_theta` with a simulator that takes a batch of parameter sets.

    Args:
        simulator: Simulator callable (a function or a class with `__call__`) that
            takes a batch of parameter sets.
        theta: All parameters $\theta$ sampled from prior or posterior.
        sim_batch_size: Number of parameter sets per simulator call. If None, all
            are simulated in one call. When using multiple workers, increasing this
            batch size can further speed up simulations by reducing overhead.
        num_workers: Number of workers for multiprocessing.
        seed: Seed for reproducibility. Every simulator call gets its own seed
            derived from it.
        show_progress_bars: Whether to show a progress bar during simulation.

    Returns:
        Simulations $x$.
    """
    if theta.ndim == 1:
        # Scalar parameters, e.g. from MCABC with a scalar prior. The simulator still
        # gets batches of shape (batch_size,).
        return simulate_in_batches(
            lambda theta_batch: simulator(theta_batch[:, 0]),
            theta[:, None],
            sim_batch_size,
            num_workers,
            seed,
            show_progress_bars,
        )
    x = simulate_from_theta(
        simulator,
        theta,
        num_workers=num_workers,
        simulation_batch_size=sim_batch_size,
        simulator_is_batched=True,
        seed=seed,
        show_progress_bar=show_progress_bars,
    )
    return torch.as_tensor(x)


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
