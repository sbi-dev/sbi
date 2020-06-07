from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from tqdm.auto import tqdm
from sbi.simulators.mp_simutils import simulate_mp
import logging


def simulate_in_batches_mp(
    simulator: Callable,
    theta: Tensor,
    sim_batch_size: Optional[int],
    number_of_workers: int = 4,
    worker_batch_size: int = 20,
    show_progressbar: bool = True,
    logging_level: Union[int, str] = "warning",
) -> Tuple[Tensor, Tensor]:
    """
    Return parameters $\theta$ and data $x$ simulated using multiprocessing.

    Distributes all parameters theta to simulate_in_batches, i.e. splits them between
    cores and then processes them in batches.

    Args:
        simulator: Simulator function.
        theta: Parameters $\theta$ sampled from prior or posterior.
        number_of_workers: Number of parallel workers to start.
        worker_batch_size: Number of parameters each worker handles. This number of
            thetas will then be handled by `simulate_in_batches()`.
        sim_batch_size: Number of simulations per batch. Default is to simulate
            the entire theta in a single batch.
        show_progressbar: Whether to show a progressbar during simulating.
        logging_level: Minimum severity of messages to log. One of the strings
            "info", "warning", "debug", "error" and "critical".

    Returns: Parameters theta and simulation outputs x.

    """

    assert worker_batch_size >= sim_batch_size, (
        "worker_batch_size has to be larger than simulation_batch_size when using"
        " multiprocessing."
    )

    batched_simulator = lambda theta_: simulate_in_batches(
        simulator, theta_, sim_batch_size, show_progressbar=False
    )
    theta, x = simulate_mp(
        simulator=batched_simulator,
        theta=theta,
        num_workers=number_of_workers,
        worker_batch_size=worker_batch_size,
        show_progressbar=show_progressbar,
        logging_level=logging_level,
    )

    return theta, x


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
        logging.warn("Zero-length parameter theta implies zero simulations.")
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
