# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from numpy import ndarray
from torch import Tensor, float32
from tqdm.auto import tqdm

from sbi.utils.sbiutils import seed_all_backends


# Refactoring following #1175. tl:dr: letting joblib iterate over numpy arrays
# allows for a roughly 10x performance gain. The resulting casting necessity
# (cfr. user_input_checks.wrap_as_joblib_efficient_simulator) introduces
# considerable overhead. The simulating pipeline should, therefore, be further
# restructured in the future (PR #1188).
def simulate_for_sbi(
    simulator: Callable,
    proposal: Any,
    num_simulations: int,
    num_workers: int = 1,
    simulation_batch_size: Union[int, None] = 1,
    seed: Optional[int] = None,
    show_progress_bar: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Returns ($\theta, x$) pairs obtained from sampling the proposal and simulating.

    This function performs two steps:

    - Sample parameters $\theta$ from the `proposal`.
    - Simulate these parameters to obtain $x$.

    Args:
        simulator: A function that takes parameters $\theta$ and maps them to
            simulations, or observations, `x`, $\text{sim}(\theta)\to x$. Any
            regular Python callable (i.e. function or class with `__call__` method)
            can be used. Note that the simulator should be able to handle numpy
            arrays for efficient parallelization. You can use
            `process_simulator` to ensure this.
        proposal: Probability distribution that the parameters $\theta$ are sampled
            from.
        num_simulations: Number of simulations that are run.
        num_workers: Number of parallel workers to use for simulations.
        simulation_batch_size: Number of parameter sets of shape
            (simulation_batch_size, parameter_dimension) that the simulator
            receives per call. If None, we set
            simulation_batch_size=num_simulations and simulate all parameter
            sets with one call. Otherwise, we construct batches of parameter
            sets and distribute them among num_workers.
        seed: Seed for reproducibility.
        show_progress_bar: Whether to show a progress bar for simulating. This will not
            affect whether there will be a progressbar while drawing samples from the
            proposal.

    Returns: Sampled parameters $\theta$ and simulation-outputs $x$.
    """

    if num_simulations == 0:
        theta = torch.tensor([], dtype=float32)
        x = torch.tensor([], dtype=float32)

    else:
        # Cast theta to numpy for better joblib performance (seee #1175)
        seed_all_backends(seed)
        theta = proposal.sample((num_simulations,))

        # Parse the simulation_batch_size logic
        if simulation_batch_size is None:
            simulation_batch_size = num_simulations
        else:
            simulation_batch_size = min(simulation_batch_size, num_simulations)

        if num_workers != 1:
            # For multiprocessing, we want to switch to numpy arrays.
            # The batch size will be an approximation, since np.array_split does
            # not take as argument the size of the batch but their total.
            num_batches = num_simulations // simulation_batch_size
            batches = np.array_split(theta.numpy(), num_batches, axis=0)
            batch_seeds = np.random.randint(low=0, high=1_000_000, size=(len(batches),))

            # define seeded simulator.
            def simulator_seeded(theta: ndarray, seed: int) -> Tensor:
                seed_all_backends(seed)
                return simulator(theta)

            try:  # catch TypeError to give more informative error message
                simulation_outputs: list[Tensor] = [  # pyright: ignore
                    xx
                    for xx in tqdm(
                        Parallel(return_as="generator", n_jobs=num_workers)(
                            delayed(simulator_seeded)(batch, seed)
                            for batch, seed in zip(batches, batch_seeds)
                        ),
                        total=num_simulations,
                        disable=not show_progress_bar,
                    )
                ]
            except TypeError as err:
                raise TypeError(
                    "For multiprocessing, we switch to numpy arrays. Make sure to "
                    "preprocess your simulator with `process_simulator` to handle numpy"
                    " arrays."
                ) from err

        else:
            simulation_outputs: list[Tensor] = []
            batches = torch.split(theta, simulation_batch_size)
            for batch in tqdm(batches, disable=not show_progress_bar):
                simulation_outputs.append(simulator(batch))

        # Correctly format the output
        x = torch.cat(simulation_outputs, dim=0)
        theta = torch.as_tensor(theta, dtype=float32)

    return theta, x
