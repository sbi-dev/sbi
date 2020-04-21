from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor


def simulate_in_batches(
    simulator: Callable,
    parameter_sample_fn: Callable,
    num_samples: int,
    simulation_batch_size: int,
) -> (Tensor, Tensor):
    r"""
    Return parameters and simulated data for `num_samples` parameter sets $\theta$.

    Simulate them in batches of size `simulation_batch_size`.

    Features:
        Allows to simulate in batches of arbitrary size.
        If `simulation_batch_size==-1`, all simulations are run at the same time.

    Args:
        simulator: simulator function.
        parameter_sample_fn: Function to call for generating $\theta$, e.g. prior
            sampling
        num_samples: Number of simulations to run
        simulation_batch_size: Number of simulations that are run within a single batch
            If `simulation_batch_size == -1`, we run a batch with all simulations
            required, i.e. `simulation_batch_size = num_samples`

    Returns:
        Simulation parameters $\theta$ of shape (num_samples, shape_of_single_theta),
        Simulator outputs $x$ of shape (num_samples, shape_of_single_x)
    """

    assert num_samples > 0, "Number of samples to simulate must be larger than zero."

    # Generate parameters (simulation inputs) by sampling from prior (round 1) or
    # proposal (round > 1).
    parameters = parameter_sample_fn(num_samples)

    if simulation_batch_size == -1:
        # Run all simulations in a single batch.
        simulation_batch_size = num_samples

    # Split parameter set into batches of size (simulation_batch_size,
    # num_dim_parameters).
    n_chunks = math.ceil(num_samples / simulation_batch_size)
    parameter_batches = torch.chunk(parameters, chunks=n_chunks)

    with torch.no_grad():
        xs = torch.cat([simulator(batch) for batch in parameter_batches])

    # XXX Construct tensor because a memory-sharing cast via as_tensor raises
    # XXX   RuntimeError: Trying to backward through the graph a second time
    # XXX when doing multiple rounds in SNPE (gradient-tracking problem to investigate).
    return torch.tensor(parameters), xs
