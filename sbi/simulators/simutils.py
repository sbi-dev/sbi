from __future__ import annotations

from typing import Callable, Optional, Tuple

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
    show_progressbar: Optional[bool] = True,
    logging_level: int = logging.WARNING,
) -> Tuple[Tensor, Tensor]:
    """
    Return parameters $\theta$ and data $x$ simulated using multiprocessing.

    Distributes all parameters theta to simulate_in_batches, i.e. splits them between
     cores and then processes them in batches.

    Args:
        simulator: simulator function.
        theta: parameters $\theta$ sampled from prior or posterior.
        number_of_workers: how many parallel workers to start
        worker_batch_size: how many params are processed on each worker. This number is
            thetas will then be handled by simulate_in_batches()
        sim_batch_size: number of simulations per batch. Default is to simulate
            the entire theta in a single batch.
        show_progressbar: whether to show a progressbar during simulating
        logging_level: The logging level determines the amount of information printed to
            the user. Currently only used for multiprocessing. One of
            logging.[INFO|WARNING|DEBUG|ERROR|CRITICAL].

    Returns: parameters theta and simulation outputs x

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
    sim_batch_size: Optional[int],
    show_progressbar: Optional[bool] = True,
) -> Tuple[Tensor, Tensor]:
    r"""
    Return simulations $x$ for parameters $\theta$ conducted batchwise.
    
    Parameters are batched with size `sim_batch_size` (default whole theta at once).

    Args:
        simulator: simulator function.
        theta: parameters $\theta$ sampled from prior or posterior.
        sim_batch_size: number of simulations per batch. Default is to simulate  
            the entire theta in a single batch.
        show_progressbar: whether to show a progressbar during simulating

    Returns:
        Parameters theta and simulations $x$
    """

    num_sims, *_ = theta.shape

    if num_sims == 0:
        raise ValueError("Zero-length parameter theta implies zero simulations.")
    elif sim_batch_size is not None and sim_batch_size < num_sims:
        # Dev note: pyright complains of torch.split lacking a type stub
        # as of PyTorch 1.4.0, see https://github.com/microsoft/pyright/issues/291
        batches = torch.split(theta, sim_batch_size, dim=0)

        pbar = tqdm(
            total=num_sims,
            disable=not show_progressbar,
            desc=f"Running {num_sims} simulations",
        )

        with pbar:
            simulation_outputs = []
            for batch in batches:
                simulation_outputs.append(simulator(batch))
                pbar.update(sim_batch_size)

        return theta, torch.cat(simulation_outputs, dim=0)
    else:
        return theta, simulator(theta)
