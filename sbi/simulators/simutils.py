# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import (
    Callable,
    Optional,
    Union,
    Dict,
    Any,
    Tuple,
    Union,
    cast,
    List,
    Sequence,
    TypeVar,
)

import torch
from torch import Tensor
from tqdm.auto import tqdm
import logging

from pathos.pools import ProcessPool
from joblib import Parallel, delayed


def simulate_in_batches(
    simulator: Callable,
    theta: Tensor,
    sim_batch_size: int = 1,
    num_workers: int = 1,
    mp_framework: str = "joblib",
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
            the entire theta in a single batch.
        num_workers: Number of workers for multiprocessing.
        mp_framework: Which framework to use for multiprocessing. Can be either of
                [`joblib` | `mp_pathos`]. Ignored if `num_workers==1`.
        show_progress_bars: Whether to show a progress bar during simulation.

    Returns:
        Parameters theta and simulations $x$.
    """

    num_sims, *_ = theta.shape

    if num_sims == 0:
        logging.warning("Zero-length parameter theta implies zero simulations.")
        x = torch.tensor([])
    elif sim_batch_size is not None and sim_batch_size < num_sims:

        if num_workers > 1:
            # Parallelize the sequence of batches across workers.
            if mp_framework == "mp_pathos":
                num_updates_pbar = 10
                batches = torch.split(
                    theta,
                    max(sim_batch_size, int(len(theta) / num_updates_pbar)),
                    dim=0,
                )
                simulation_outputs = []
                for subset_of_batches in tqdm(
                    batches,
                    disable=not show_progress_bars,
                    desc=f"Running {num_sims} simulations in {len(batches)} batches.",
                    total=num_updates_pbar,
                ):
                    proc_batches = torch.split(subset_of_batches, sim_batch_size, dim=0)
                    pool = ProcessPool(processes=num_workers, maxtasksperchild=1)
                    simulation_outputs.append(
                        torch.cat(pool.map(simulator, list(proc_batches)))
                    )
                    pool.clear()
            elif mp_framework == "joblib":
                # Dev note: pyright complains of torch.split lacking a type stub
                # as of PyTorch 1.4.0, see
                # https://github.com/microsoft/pyright/issues/291
                batches = torch.split(theta, sim_batch_size, dim=0)

                # TODO: This usage of tqdm tracks the dispatching of jobs instead of the
                #  moment when they are done, resulting in waiting time at 100% in case
                #  the last jobs takes long. A potential solution can be found here:
                #  https://stackoverflow.com/a/61689175
                simulation_outputs = Parallel(n_jobs=num_workers)(
                    delayed(simulator)(batch)
                    for batch in tqdm(
                        batches,
                        disable=not show_progress_bars,
                        desc=f"Running {num_sims} simulations in {len(batches)} batches.",
                        total=len(batches),
                    )
                )
            else:
                raise NameError(
                    f"mp_framework {mp_framework} not supported. Current "
                    f'supported frameworks are: ["mp_pathos" | "joblib"].'
                )
        else:
            pbar = tqdm(
                total=num_sims,
                disable=not show_progress_bars,
                desc=f"Running {num_sims} simulations.",
            )

            with pbar:
                batches = torch.split(theta, sim_batch_size, dim=0)
                simulation_outputs = []
                for batch in batches:
                    simulation_outputs.append(simulator(batch))
                    pbar.update(sim_batch_size)

        x = torch.cat(simulation_outputs, dim=0)
        print("simulation_outputs", x.shape)
    else:
        x = simulator(theta)

    return x
