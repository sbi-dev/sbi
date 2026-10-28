# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from dataclasses import dataclass
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import torch
from joblib import Parallel, delayed
from torch import Tensor, float32
from tqdm.auto import tqdm

from sbi.sbi_types import (
    Array,
    Proposal,
    SimulationOutput,
)
from sbi.utils.sbiutils import seed_all_backends


def simulate_for_sbi(
    simulator: Callable,
    proposal: Proposal,
    num_simulations: int,
    num_workers: int = 1,
    simulation_batch_size: Optional[int] = 1,
    seed: Optional[int] = None,
    show_progress_bar: bool = True,
    simulator_is_batched: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Returns pairs :math:`(\theta, x)` by sampling proposal and running simulations.

    This function performs two steps:

    - Sample parameters $\theta$ from the `proposal`.
    - Simulate these parameters to obtain $x$ with :func:`simulate_from_theta`.

    Args:
        simulator: A function that takes parameters $\theta$ and maps them to
            simulations, or observations, `x`, $\text{sim}(\theta)\to x$. Any
            regular Python callable (i.e. function or class with `__call__` method)
            can be used. It receives the sampled parameters as a tensor and must
            return numbers: a tensor, a numpy array, or a list of numbers per
            parameter set.
        proposal: Probability distribution that the parameters $\theta$ are sampled
            from.
        num_simulations: Number of simulations that are run.
        num_workers: Number of parallel joblib workers; -1 uses all CPUs.
        simulation_batch_size: Number of parameter sets per simulator call, or per
            worker task for a simulator that is not batched. If None, all parameter
            sets go into one call.
        seed: Seed for reproducibility. It seeds the global random state before
            the parameters are sampled and leaves it seeded.
        show_progress_bar: Whether to show a progress bar for simulating. This will not
            affect whether there will be a progressbar while drawing samples from the
            proposal.
        simulator_is_batched: Whether the simulator takes a batch of parameter sets
            per call. If False, it is called once per parameter set. Unlike in
            :func:`simulate_from_theta`, the default is True.

    Returns: Sampled parameters $\theta$ and simulation-outputs $x$.
    """
    settings = _Settings(num_workers, simulation_batch_size, simulator_is_batched)
    if num_simulations == 0:
        return torch.tensor([], dtype=float32), torch.tensor([], dtype=float32)

    seed_all_backends(seed)
    theta = proposal.sample((num_simulations,))
    x = _simulate(
        simulator, theta, settings, seed=None, show_progress_bar=show_progress_bar
    )
    if not isinstance(x, (Tensor, np.ndarray)):
        raise TypeError(
            "`simulate_for_sbi` needs numeric simulator outputs of one shape. For "
            "other outputs, such as file paths, use `simulate_from_theta`."
        )

    return (
        torch.as_tensor(theta, dtype=float32),
        torch.as_tensor(x, dtype=float32, device=theta.device),
    )


def simulate_from_theta(
    simulator: Callable,
    theta: Array,
    *,
    num_workers: int = 1,
    simulation_batch_size: Optional[int] = 1,
    simulator_is_batched: bool = False,
    seed: Optional[int] = None,
    show_progress_bar: bool = True,
) -> SimulationOutput:
    r"""Returns simulator outputs for given parameters, optionally in parallel.

    Args:
        simulator: Function that maps parameters to simulation outputs. By default,
            it takes one parameter set per call. If `simulator_is_batched`, it takes
            a batch of parameter sets and returns outputs with the same leading
            dimension, or a list of that length.
        theta: Parameters of shape (num_simulations, \*parameter_shape), as a
            tensor or numpy array. Other array-likes become numpy arrays.
        num_workers: Number of parallel joblib workers; -1 uses all CPUs. The
            simulator receives parameters of the same type as `theta` for any number
            of workers; with more than one worker, tensors are on the CPU.
        simulation_batch_size: Number of parameter sets per simulator call, or per
            worker task for a simulator that is not batched. If None, all parameter
            sets go into one call.
        simulator_is_batched: Whether the simulator takes a batch of parameter sets
            per call. A batched simulator passed with False gets one parameter set
            per call: indexing such as `theta[:, 0]` raises, and outputs sized by
            `len(theta)` gain a dimension.
        seed: Seed for reproducibility. It seeds the global random state before
            the simulations, and each worker task gets a distinct seed derived from
            it. If None, the current random state is used.
        show_progress_bar: Whether to show a progress bar.

    Returns:
        Simulation outputs in the order of `theta`: tensors or numpy arrays are
        concatenated along the first dimension, other outputs (such as file paths)
        are returned as a list. Tensor outputs are on the device of `theta`.
    """
    settings = _Settings(num_workers, simulation_batch_size, simulator_is_batched)
    return _simulate(simulator, theta, settings, seed, show_progress_bar)


@overload
def parallelize_simulator(
    simulator: Callable,
    /,
    *,
    num_workers: int = ...,
    simulation_batch_size: Optional[int] = ...,
    simulator_is_batched: bool = ...,
    show_progress_bar: bool = ...,
) -> Callable[[Array], SimulationOutput]: ...


@overload
def parallelize_simulator(
    simulator: None = None,
    /,
    *,
    num_workers: int = ...,
    simulation_batch_size: Optional[int] = ...,
    simulator_is_batched: bool = ...,
    show_progress_bar: bool = ...,
) -> Callable[[Callable], Callable[[Array], SimulationOutput]]: ...


def parallelize_simulator(
    simulator: Optional[Callable] = None,
    /,
    *,
    num_workers: int = -1,
    simulation_batch_size: Optional[int] = 1,
    simulator_is_batched: bool = False,
    show_progress_bar: bool = False,
) -> Union[
    Callable[[Array], SimulationOutput],
    Callable[[Callable], Callable[[Array], SimulationOutput]],
]:
    r"""Returns a batched simulator that runs :func:`simulate_from_theta` per call.

    Use it as a function, `parallelize_simulator(simulator)`, or as a decorator,
    `@parallelize_simulator` or `@parallelize_simulator(num_workers=4)`. By default,
    the simulator takes one parameter set per call and runs on all CPUs.

    To reproduce a call, seed beforehand, e.g. with
    :func:`sbi.utils.sbiutils.seed_all_backends`.

    Each call sends the simulator to the workers; for slow calls, see the how-to
    guide "How to run simulations in parallel".

    Args:
        simulator: Function that maps parameters to simulation outputs, see
            :func:`simulate_from_theta`. If None, returns a decorator.
        num_workers: Number of parallel joblib workers; -1 uses all CPUs.
        simulation_batch_size: Number of parameter sets per simulator call, or per
            worker task for a simulator that is not batched, see
            :func:`simulate_from_theta`.
        simulator_is_batched: Whether the simulator takes a batch of parameter sets
            per call.
        show_progress_bar: Whether to show a progress bar on every call.

    Returns:
        Function that takes parameters of shape (num_simulations,
        \*parameter_shape) and returns their simulation outputs, or a decorator
        that creates it.
    """
    if simulator is not None and not callable(simulator):
        raise TypeError(
            "The simulator must be callable. Pass settings as keywords, e.g. "
            "`@parallelize_simulator(num_workers=4)`."
        )
    settings = _Settings(num_workers, simulation_batch_size, simulator_is_batched)

    def decorator(simulator: Callable) -> Callable[[Array], SimulationOutput]:
        @wraps(simulator)
        def parallel_simulator(theta: Array) -> SimulationOutput:
            return _simulate(simulator, theta, settings, None, show_progress_bar)

        return parallel_simulator

    return decorator if simulator is None else decorator(simulator)


@dataclass(frozen=True)
class _Settings:
    """How to run the simulations, checked once when created."""

    num_workers: int
    simulation_batch_size: Optional[int]
    simulator_is_batched: bool

    def __post_init__(self) -> None:
        if not _is_int(self.num_workers) or self.num_workers == 0:
            raise ValueError(
                "`num_workers` must be a nonzero integer, e.g. 4, or -1 for all "
                f"CPUs; got {self.num_workers!r}."
            )
        if self.simulation_batch_size is not None and (
            not _is_int(self.simulation_batch_size) or self.simulation_batch_size < 1
        ):
            raise ValueError(
                "`simulation_batch_size` must be None or a positive integer, got "
                f"{self.simulation_batch_size!r}."
            )

    def batch_size(self, num_simulations: int) -> int:
        if self.simulation_batch_size is None:
            return num_simulations
        return int(self.simulation_batch_size)


class _TensorAsNumpy(NamedTuple):
    """A tensor sent to or from a worker as a numpy array."""

    array: np.ndarray


def _simulate(
    simulator: Callable,
    theta: Array,
    settings: _Settings,
    seed: Optional[int],
    show_progress_bar: bool,
) -> SimulationOutput:
    """Runs the simulations of :func:`simulate_from_theta`."""
    if not isinstance(theta, (Tensor, np.ndarray)):
        theta = np.asarray(theta)
    if theta.ndim < 2:
        raise ValueError(
            "`theta` must have shape (num_simulations, *parameter_shape), got "
            f"{tuple(theta.shape)}."
        )
    num_simulations = len(theta)
    if num_simulations == 0:
        if isinstance(theta, Tensor):
            return torch.empty(0, dtype=float32, device=theta.device)
        return np.empty(0)

    device = theta.device if isinstance(theta, Tensor) else None
    run = simulator if settings.simulator_is_batched else partial(_run_rows, simulator)
    batch_size = settings.batch_size(num_simulations)
    starts = range(0, num_simulations, batch_size)
    sizes = [min(batch_size, num_simulations - start) for start in starts]
    if seed is not None:
        seed_all_backends(seed)
    results: Generator[Any, None, None]
    if settings.num_workers == 1:
        results = (run(part) for part in _split(theta, batch_size))
    else:
        thetas = [_to_transport(part) for part in _split(theta, batch_size)]
        parallel = Parallel(n_jobs=settings.num_workers, return_as="generator")
        results = cast(
            Generator,
            parallel(
                delayed(_run_in_worker)(run, part, worker_seed)
                for part, worker_seed in zip(
                    thetas, _distinct_seeds(len(starts), None), strict=True
                )
            ),
        )

    outputs = []
    try:
        with tqdm(total=num_simulations, disable=not show_progress_bar) as pbar:
            for size, result in zip(sizes, results, strict=True):
                output = _from_transport(result)
                _check_num_outputs(output, size)
                outputs.append(output)
                pbar.update(size)
    finally:
        # Closing joblib's generator cancels the simulations not yet run.
        results.close()

    x = _concatenate(outputs)
    return x.to(device) if device is not None and isinstance(x, Tensor) else x


def _run_rows(simulator: Callable, theta: Array) -> Any:
    """Runs a simulator that is not batched on each parameter set of a batch."""
    outputs = [simulator(row) for row in theta]
    if isinstance(outputs[0], Tensor):
        try:
            return torch.stack(outputs)
        except RuntimeError:  # Outputs of different shapes.
            pass
    return outputs


def _run_in_worker(run: Callable, theta: Any, worker_seed: int) -> Any:
    # Worker processes start from the same torch random state.
    seed_all_backends(worker_seed)
    return _to_transport(run(_from_transport(theta)))


def _to_transport(value: Any) -> Any:
    """Returns a tensor as a numpy array, which is faster to send to a worker.

    Tensors that numpy cannot hold, e.g. bfloat16, and other values stay as they are.
    """
    if not isinstance(value, Tensor):
        return value
    value = value.detach().cpu()
    try:
        return _TensorAsNumpy(value.resolve_conj().resolve_neg().numpy())
    except (TypeError, RuntimeError):  # No numpy dtype, e.g. bfloat16.
        return value


def _from_transport(value: Any) -> Any:
    if not isinstance(value, _TensorAsNumpy):
        return value
    # joblib passes large arrays as read-only memory maps, which torch must copy.
    array = value.array
    return torch.from_numpy(array) if array.flags.writeable else torch.tensor(array)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _check_num_outputs(output: Any, num_parameter_sets: int) -> None:
    num_outputs = None  # One output without a batch dimension.
    if isinstance(output, (Tensor, np.ndarray)) and output.ndim > 0:
        num_outputs = output.shape[0]
    elif isinstance(output, (list, tuple)):
        num_outputs = len(output)
    if num_outputs != num_parameter_sets:
        got = "one output" if num_outputs in (None, 1) else f"{num_outputs} outputs"
        sets = (
            "one parameter set"
            if num_parameter_sets == 1
            else f"a batch of {num_parameter_sets} parameter sets"
        )
        raise ValueError(
            f"The simulator returned {got} for {sets}. If it takes one parameter "
            "set per call, pass `simulator_is_batched=False`."
        )


def _split(theta: Array, batch_size: int) -> List[Array]:
    if isinstance(theta, Tensor):
        return list(theta.split(batch_size))
    return [theta[i : i + batch_size] for i in range(0, len(theta), batch_size)]


def _distinct_seeds(num_seeds: int, seed: Optional[int]) -> np.ndarray:
    """Returns distinct seeds derived from `seed`.

    Distinct seeds keep batches or simulations from repeating each other's noise.
    With `seed=None`, the seeds follow the global numpy random state.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    # `np.random.seed` accepts seeds below 2**32.
    return np.random.default_rng(seed).choice(2**32, size=num_seeds, replace=False)


def _is_numeric(output: Any) -> bool:
    if isinstance(output, Tensor):
        return True
    try:
        return np.asarray(output).dtype.kind in "biufc"
    except ValueError:  # Nested lists of different lengths.
        return False


def _concatenate(outputs: List[Any]) -> SimulationOutput:
    """Concatenates per-batch outputs along the first dimension.

    Other outputs, e.g. paths, are returned as a list.
    """
    try:
        if all(isinstance(output, Tensor) for output in outputs):
            return torch.cat(outputs)
        if all(isinstance(output, np.ndarray) for output in outputs):
            return np.concatenate(outputs)
    except (RuntimeError, ValueError):  # Outputs of different shapes.
        pass
    rows = [row for batch in outputs for row in batch]
    if isinstance(rows[0], Tensor):
        try:
            return torch.stack(rows)
        except RuntimeError:  # Outputs of different shapes.
            return rows
    if all(_is_numeric(row) for row in rows):
        try:
            return np.asarray(rows)
        except ValueError:  # Outputs of different shapes.
            return rows
    return rows
