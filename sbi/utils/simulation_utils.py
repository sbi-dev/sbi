# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Any, Callable, List, Optional, Tuple, Union, overload

import numpy as np
import torch
from joblib import Parallel, delayed, parallel_config
from torch import Tensor, float32
from tqdm.auto import tqdm

from sbi.utils.sbiutils import seed_all_backends

Data = Tensor | np.ndarray | List[str]
Theta = Tensor | np.ndarray | List[Any]


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
) -> Tuple[Tensor, Tensor | List[str]]:
    r"""Returns pairs :math:`(\theta, x)` by sampling proposal and running simulations.

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
        return torch.tensor([], dtype=float32), torch.tensor([], dtype=float32)

    seed_all_backends(seed)
    theta = proposal.sample((num_simulations,))
    # Cast to numpy for joblib efficiency
    theta_numpy = theta.cpu().numpy()

    if simulation_batch_size is None:
        simulation_batch_size = num_simulations
    else:
        simulation_batch_size = min(simulation_batch_size, num_simulations)

    # Handle parallel context
    context = parallel_config(n_jobs=num_workers)

    with context:
        # We enforce simulator_is_batched=True because simulate_for_sbi semantics
        # implies that the simulator receives batches (even if size 1).
        try:
            theta, x = simulate_from_thetas(
                simulator,
                theta_numpy,
                simulation_batch_size=simulation_batch_size,
                simulator_is_batched=True,
                show_progress_bar=show_progress_bar,
                seed=seed,
            )
        except TypeError as err:
            if num_workers > 1:
                raise TypeError(
                    "There is a TypeError error in your simulator function. Note: For"
                    " multiprocessing, we switch to numpy arrays. Besides confirming"
                    " your simulator works correctly, make sure to preprocess your"
                    " simulator with `process_simulator` to handle numpy arrays."
                ) from err
            else:
                raise err

    # Correctly format the output to Tensor
    theta = torch.as_tensor(theta, dtype=float32)

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    return theta, x


@overload
def parallelize_simulator(
    simulator: Callable[[Theta], Data],
    simulator_is_batched: bool = ...,
    simulation_batch_size: int = ...,
    show_progress_bar: bool = ...,
    seed: Optional[int] = ...,
) -> Callable[[Theta], Data]: ...


@overload
def parallelize_simulator(
    simulator: None = None,
    simulator_is_batched: bool = ...,
    simulation_batch_size: int = ...,
    show_progress_bar: bool = ...,
    seed: Optional[int] = ...,
) -> Callable[[Callable[[Theta], Data]], Callable[[Theta], Data]]: ...


def parallelize_simulator(
    simulator: Callable[[Theta], Data] | None = None,
    simulator_is_batched: bool = False,
    simulation_batch_size: int = 10,
    show_progress_bar: bool = True,
    seed: Optional[int] = None,
) -> Union[
    Callable[[Theta], Data],
    Callable[[Callable[[Theta], Data]], Callable[[Theta], Data]],
]:
    r"""
    Returns a function that executes simulations in parallel for a given set of
    parameters. Can be used as a function or a decorator.

    Args:
        simulator: Function to run simulations.
        simulator_is_batched: Whether the simulator can handle batches directly.
        simulation_batch_size: Number of simulations to run in each batch.
        show_progress_bar: Whether to show tqdm progress bar.
        seed: Random seed.

    Returns:
        Callable that takes a set of :math:`\theta` and returns simulation outputs.
    """

    def decorator(simulator_func: Callable[[Theta], Data]) -> Callable[[Theta], Data]:
        warnings.warn(
            "Joblib is used for parallelization. It is recommended to use numpy arrays "
            "for the simulator input and output to avoid serialization overhead with "
            "torch tensors.",
            UserWarning,
            stacklevel=2,
        )

        def parallel_simulator(thetas: Theta) -> Data:
            seed_all_backends(seed)

            num_simulations = len(thetas)

            if num_simulations == 0:
                return torch.tensor([], dtype=float32)

            # Create batches
            if simulator_is_batched:
                num_batches = (
                    num_simulations + simulation_batch_size - 1
                ) // simulation_batch_size
                batches = [
                    thetas[i * simulation_batch_size : (i + 1) * simulation_batch_size]
                    for i in range(num_batches)
                ]
            elif simulation_batch_size > 1:
                warnings.warn(
                    "Simulation batch size is greater than 1, but simulator_is_batched "
                    "is False. Simulations will be run sequentially (batch size 1).",
                    UserWarning,
                    stacklevel=2,
                )
                batches = [theta for theta in thetas]
            else:
                batches = [theta for theta in thetas]

            # Run in parallel
            # Generate seeds
            batch_seeds = np.random.randint(low=0, high=1_000_000, size=(len(batches),))

            def run_simulation(batch, seed):
                seed_all_backends(seed)
                return simulator_func(batch)

            # Execute in parallel with joblib
            results = Parallel(return_as="generator")(
                delayed(run_simulation)(batch, seed)
                for batch, seed in zip(batches, batch_seeds, strict=False)
            )

            # Progress bar
            simulation_outputs = []
            if show_progress_bar:
                pbar = tqdm(total=num_simulations)

            for i, res in enumerate(results):
                simulation_outputs.append(res)
                if show_progress_bar:
                    pbar.update(len(batches[i]))

            if show_progress_bar:
                pbar.close()

            # Flatten results
            output_data = []
            if simulator_is_batched:
                for batch_out in simulation_outputs:
                    if isinstance(batch_out, (list, tuple)):
                        output_data.extend(batch_out)
                    elif isinstance(batch_out, (torch.Tensor, np.ndarray)):
                        output_data.extend([x for x in batch_out])
                    else:
                        output_data.append(batch_out)
            else:
                output_data = simulation_outputs

            if not output_data:
                return torch.tensor([], dtype=float32)

            # Handle file paths (strings)
            if isinstance(output_data[0], (str, np.str_, np.bytes_)):
                output_data = [str(f) for f in output_data]
                return output_data

            if isinstance(output_data[0], torch.Tensor):
                return torch.stack(output_data)
            elif isinstance(output_data[0], np.ndarray):
                return np.stack(output_data)

            return output_data

        return parallel_simulator

    if simulator is None:
        return decorator

    return decorator(simulator)


def simulate_from_thetas(
    simulator: Callable[[Theta], Data],
    thetas: Theta,
    simulator_is_batched: bool = False,
    simulation_batch_size: int = 10,
    show_progress_bar: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Theta, Data]:
    r"""
    Execute simulations for a given set of parameters.

    Args:
        simulator: Function to run simulations.
        thetas: Parameters to simulate (Tensor, Numpy array, or list).
        simulator_is_batched: Whether the simulator can handle batches directly.
        simulation_batch_size: Number of simulations to run in each batch.
        show_progress_bar: Whether to show tqdm progress bar.
        seed: Random seed.

    Returns:
        Tuple of (:math:`\theta`, simulation_outputs).
    """
    parallel_sim = parallelize_simulator(
        simulator,
        simulator_is_batched=simulator_is_batched,
        simulation_batch_size=simulation_batch_size,
        show_progress_bar=show_progress_bar,
        seed=seed,
    )

    return thetas, parallel_sim(thetas)
