from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor
from tqdm.auto import tqdm


def simulate_in_batches(
    simulator: Callable,
    theta: Tensor,
    sim_batch_size: Optional[int],
    show_progressbar: Optional[bool] = True,
) -> Tensor:
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
        Simulations $x$ with shape (num_sims, shape_of_single_x)
    """
    num_sims, *_ = theta.shape

    if num_sims == 0:
        raise ValueError("Zero-length parameter theta implies zero simulations.")
    elif sim_batch_size is not None and sim_batch_size < num_sims:
        # Dev note: pyright complains of torch.split lacking a type stub
        # as of PyTorch 1.4.0, see https://github.com/microsoft/pyright/issues/291
        batches = torch.split(theta, sim_batch_size, dim=0)

        pbar = tqdm(total=num_sims, disable=not show_progressbar)
        desc = "Running {0} simulations".format(num_sims)
        if type(show_progressbar) == str:
            desc += show_progressbar
        pbar.set_description(desc)

        with pbar:
            simulation_outputs = []
            for batch in batches:
                simulation_outputs.append(simulator(batch))
                pbar.update(sim_batch_size)
        return torch.cat(simulation_outputs, dim=0)
    else:
        return simulator(theta)
