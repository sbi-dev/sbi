from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from tqdm.auto import tqdm
import logging
from joblib import Parallel, delayed


def simulate_in_batches(
    simulator: Callable,
    theta: Tensor,
    sim_batch_size: Optional[int] = None,
    show_progressbar: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""
    Return simulations $x$ for parameters $\theta$ conducted batchwise.

    Parameters are batched with size `sim_batch_size` (default whole theta at once).

    Args:
        simulator: Simulator callable (a function or a class with `__call__`).
        theta: Parameters $\theta$ sampled from prior or posterior.
        sim_batch_size: Number of simulations per batch. Default is to simulate
            the entire theta in a single batch.
        show_progressbar: Whether to show a progressbar during simulation.

    Returns:
        Parameters theta and simulations $x$.
    """

    num_sims, *_ = theta.shape

    if num_sims == 0:
        logging.warning("Zero-length parameter theta implies zero simulations.")
        x = torch.tensor([])
    elif sim_batch_size is not None and sim_batch_size < num_sims:
        # Dev note: pyright complains of torch.split lacking a type stub
        # as of PyTorch 1.4.0, see https://github.com/microsoft/pyright/issues/291
        batches = torch.split(theta, sim_batch_size, dim=0)

        pbar = tqdm(
            total=num_sims,
            disable=not show_progressbar,
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

    return theta, x


def simulate_in_batches_joblib(
    simulator: Callable,
    theta: Tensor,
    sim_batch_size: Optional[int] = None,
    num_workers: int = 1,
    show_progressbar: bool = True,
    pbar_steps: int = 10,
) -> Tuple[Tensor, Tensor]:
    r"""
    Return simulations $x$ for parameters $\theta$ conducted batchwise.

    Parameters are batched with size `sim_batch_size` (default whole theta at once).

    If `num_workers > 1` the batches of simulations distributed across `num_workers`
    processes. If the simulation batch size is larger than the overall number of
    simulations no multiprocessing is used.

    The progressbar is presented with a fixed number of steps.

    Args:
        simulator: Simulator callable (a function or a class with `__call__`).
        theta: Parameters $\theta$ sampled from prior or posterior.
        sim_batch_size: Number of simulations per batch. Default is to simulate
            the entire theta in a single batch.
        num_workers: Number of workers for multiprocessing.
        show_progressbar: Whether to show a progressbar during simulation.
        pbar_steps: Number of steps presented in the progressbar.
    Returns:
        Parameters theta and simulations $x$.
    """

    num_sims, *_ = theta.shape

    if num_sims == 0:
        logging.warning("Zero-length parameter theta implies zero simulations.")
        x = torch.tensor([])
    elif sim_batch_size is not None and sim_batch_size < num_sims:
        batches = torch.split(theta, sim_batch_size, dim=0)

        num_batches = len(batches)

        pbar_steps = pbar_steps if show_progressbar else 1
        pbar_step_size = int(num_batches / pbar_steps)

        simulation_outputs = []

        pbar = tqdm(
            total=num_batches,
            disable=not show_progressbar,
            desc=f"Running {num_batches} batches of simulations.",
        )

        with Parallel(n_jobs=num_workers) as parallel:
            for idx in range(pbar_steps):

                pbar_batches = batches[
                    (idx * pbar_step_size) : (idx + 1) * pbar_step_size
                ]

                simulation_outputs += parallel(
                    delayed(simulator)(batch) for batch in pbar_batches
                )
                pbar.update(pbar_step_size)

        pbar.close()

        x = torch.cat(simulation_outputs, dim=0)
    else:
        x = simulator(theta)

    return theta, x
