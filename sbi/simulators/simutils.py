from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


def simulate_in_batches(
    simulator: Callable, theta: Tensor, sim_batch_size: Optional[int],
) -> Tensor:
    r"""
    Return simulations $x$ for parameters $\theta$ conducted batchwise.
    
    Parameters are batched with size `sim_batch_size` (default whole theta at once).

    Args:
        simulator: simulator function.
        theta: parameters $\theta$ sampled from prior or posterior.
        sim_batch_size: number of simulations per batch. Default is to simulate  
            the entire theta in a single batch.

    Returns:
        Simulations $x$ with shape (num_sims, shape_of_single_x)
    """
    num_sims, *_ = theta.shape

    if num_sims == 0:
        raise ValueError("Zero-length parameter theta implies zero simulations.")
    elif sim_batch_size is not None and sim_batch_size < num_sims:
        # Pyright complains of lack of a type stub for torch.split as of 1.4.0
        batches = torch.split(theta, sim_batch_size, dim=0)
        return torch.cat([simulator(batch) for batch in batches], dim=0)
    else:
        return simulator(theta)
