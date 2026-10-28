# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pickle
import random
import warnings
from dataclasses import dataclass
from functools import partial, wraps
from math import ceil
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import torch
from joblib import Parallel, delayed, effective_n_jobs
from joblib.parallel import get_active_backend
from torch import Tensor, float32
from tqdm.auto import tqdm

from sbi.sbi_types import (
    Array,
    OnError,
    Proposal,
    SimulationBatchSize,
    SimulationOutput,
)
from sbi.utils.sbiutils import seed_all_backends


def simulate_for_sbi(
    simulator: Callable,
    proposal: Proposal,
    num_simulations: int,
    num_workers: int = 1,
    simulation_batch_size: SimulationBatchSize = "auto",
    seed: Optional[int] = None,
    show_progress_bar: bool = True,
    simulator_is_batched: bool = True,
    on_error: OnError = "raise",
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
        num_workers: Number of parallel joblib workers; -1 uses all CPUs, -2 all
            but one, and so on.
        simulation_batch_size: Number of parameter sets per simulator call, or per
            worker task for a simulator that is not batched. "auto" uses one
            parameter set per call with one worker, and four tasks per worker with
            more workers. For all parameter sets in one call, pass their number.
            None is deprecated and means "auto".
        seed: Seed for reproducibility. It seeds the global random state before
            the parameters are sampled and leaves it seeded. A simulator that is not
            batched is also reseeded before each simulation, see
            :func:`simulate_from_theta`. A batched simulator continues the seeded
            random state on one worker; with more workers, each batch gets its own
            seed, so `x` differs between one worker and more.
        show_progress_bar: Whether to show a progress bar for simulating. This will not
            affect whether there will be a progressbar while drawing samples from the
            proposal.
        simulator_is_batched: Whether the simulator takes a batch of parameter sets
            per call. If False, it is called once per parameter set. Unlike in
            :func:`simulate_from_theta`, the default is True.
        on_error: What to do when the simulator raises an exception, "raise" or
            "nan", see :func:`simulate_from_theta`.

    Returns: Sampled parameters $\theta$ and simulation-outputs $x$, both float32
    tensors on the device of $\theta$. One number per simulation gives $x$ of shape
    (num_simulations, 1).
    """
    settings = _Settings(
        num_workers, simulation_batch_size, simulator_is_batched, on_error
    )
    if num_simulations == 0:
        return torch.tensor([], dtype=float32), torch.tensor([], dtype=float32)

    _check_backend(seed, num_workers)
    seed_all_backends(seed)
    theta = proposal.sample((num_simulations,))
    if theta.ndim < 2:
        raise ValueError(
            f"The proposal samples parameters of shape {tuple(theta.shape)}, but "
            "`simulate_for_sbi` needs (num_simulations, *parameter_shape). A "
            "proposal over one parameter needs the event shape (1,), e.g. "
            "`BoxUniform(torch.zeros(1), torch.ones(1))`."
        )
    # Batched simulations continue the seeded random state.
    x = _simulate(
        simulator,
        theta,
        settings,
        seed=None if simulator_is_batched else seed,
        show_progress_bar=show_progress_bar,
    )
    if not isinstance(x, (Tensor, np.ndarray)):
        raise TypeError(
            "`simulate_for_sbi` needs numeric simulator outputs of one shape. For "
            "other outputs, such as file paths, use `simulate_from_theta`."
        )

    x = torch.as_tensor(x, dtype=float32, device=theta.device)
    return torch.as_tensor(theta, dtype=float32), x[:, None] if x.ndim == 1 else x


def simulate_from_theta(
    simulator: Callable,
    theta: Union[Array, Sequence[Any]],
    *,
    num_workers: int = 1,
    simulation_batch_size: SimulationBatchSize = "auto",
    simulator_is_batched: bool = False,
    seed: Optional[int] = None,
    on_error: OnError = "raise",
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
        num_workers: Number of parallel joblib workers; -1 uses all CPUs, -2 all
            but one, and so on. The
            simulator receives parameters of the same type as `theta` for any number
            of workers; with more than one worker, tensors are on the CPU.
        simulation_batch_size: Number of parameter sets per simulator call, or per
            worker task for a simulator that is not batched. "auto" uses one
            parameter set per call with one worker, and four tasks per worker with
            more workers. For all parameter sets in one call, pass their number.
            None is deprecated and means "auto".
        simulator_is_batched: Whether the simulator takes a batch of parameter sets
            per call. A batched simulator passed with False gets one parameter set
            per call: indexing such as `theta[:, 0]` raises, and outputs sized by
            `len(theta)` gain a dimension.
        seed: Seed for reproducibility. The simulator is reseeded before each call:
            before each simulation if it is not batched, before each batch if it
            is. For a simulator that is not batched, the outputs then depend on
            neither `num_workers` nor `simulation_batch_size`; for a batched one,
            they do not depend on `num_workers` at a fixed `simulation_batch_size`.
            This holds for noise drawn on the CPU: with more than one worker, the
            simulator gets tensors on the CPU. The caller's random state is
            restored afterwards. If None, the current random state is used; with
            more than one worker, the workers are seeded from the global numpy
            random state, so seeding before the call reproduces it. Seeding needs
            joblib's default process workers: with a thread backend, which shares
            one random state, a seed raises an error.
        on_error: What to do when the simulator raises an exception. "raise" stops
            all simulations and raises it. "nan" keeps going: a failed simulation
            returns NaN, or None for outputs that are not numeric (for a batched
            simulator, the whole batch), with one warning at the end; integer
            outputs then become float64. If all simulations fail, the first
            exception is raised. With more than one worker, an exception that the
            workers cannot send back, e.g. one whose class is defined in a script,
            arrives as a RuntimeError with its type and message.
        show_progress_bar: Whether to show a progress bar.

    Returns:
        Simulation outputs in the order of `theta`, one per parameter set. Numeric
        outputs of one shape form one tensor or numpy array with a leading
        dimension of num_simulations; other outputs, such as file paths, and outputs
        of different shapes come back as a list. Tensor outputs are on the device
        of `theta`; float64 becomes float32 on MPS, which has no float64.
    """
    settings = _Settings(
        num_workers, simulation_batch_size, simulator_is_batched, on_error
    )
    return _simulate(simulator, theta, settings, seed, show_progress_bar)


@overload
def parallelize_simulator(
    simulator: Callable,
    /,
    *,
    num_workers: int = ...,
    simulation_batch_size: SimulationBatchSize = ...,
    simulator_is_batched: bool = ...,
    on_error: OnError = ...,
    show_progress_bar: bool = ...,
) -> Callable[[Array], SimulationOutput]: ...


@overload
def parallelize_simulator(
    simulator: None = None,
    /,
    *,
    num_workers: int = ...,
    simulation_batch_size: SimulationBatchSize = ...,
    simulator_is_batched: bool = ...,
    on_error: OnError = ...,
    show_progress_bar: bool = ...,
) -> Callable[[Callable], Callable[[Array], SimulationOutput]]: ...


def parallelize_simulator(
    simulator: Optional[Callable] = None,
    /,
    *,
    num_workers: int = -1,
    simulation_batch_size: SimulationBatchSize = "auto",
    simulator_is_batched: bool = False,
    on_error: OnError = "raise",
    show_progress_bar: bool = False,
) -> Union[
    Callable[[Array], SimulationOutput],
    Callable[[Callable], Callable[[Array], SimulationOutput]],
]:
    r"""Returns a batched simulator that runs :func:`simulate_from_theta` per call.

    Use it as a function, `parallelize_simulator(simulator)`, or as a decorator,
    `@parallelize_simulator` or `@parallelize_simulator(num_workers=4)`. By default,
    the simulator takes one parameter set per call and runs on all CPUs.

    Calls are not reseeded. To reproduce a call, seed beforehand, e.g. with
    :func:`sbi.utils.sbiutils.seed_all_backends`; consecutive calls still draw
    different noise. This needs joblib's default process workers.

    Each call sends the simulator to the workers; for slow calls, see the how-to
    guide "How to run simulations in parallel".

    Args:
        simulator: Function that maps parameters to simulation outputs, see
            :func:`simulate_from_theta`. If None, returns a decorator.
        num_workers: Number of parallel joblib workers; -1 uses all CPUs, -2 all
            but one, and so on.
        simulation_batch_size: Number of parameter sets per simulator call, or per
            worker task for a simulator that is not batched, see
            :func:`simulate_from_theta`.
        simulator_is_batched: Whether the simulator takes a batch of parameter sets
            per call.
        on_error: What to do when the simulator raises an exception, "raise" or
            "nan", see :func:`simulate_from_theta`.
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
    settings = _Settings(
        num_workers, simulation_batch_size, simulator_is_batched, on_error
    )

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
    simulation_batch_size: SimulationBatchSize
    simulator_is_batched: bool
    on_error: OnError

    def __post_init__(self) -> None:
        if not _is_int(self.num_workers) or self.num_workers == 0:
            raise ValueError(
                "`num_workers` must be a nonzero integer, e.g. 4, or -1 for all "
                f"CPUs; got {self.num_workers!r}."
            )
        if self.simulation_batch_size is None:
            warnings.warn(
                '`simulation_batch_size=None` is deprecated and now means "auto": '
                "one parameter set per call with one worker, four tasks per worker "
                "with more workers. For all parameter sets in one call, pass their "
                "number.",
                FutureWarning,
                stacklevel=4,
            )
            object.__setattr__(self, "simulation_batch_size", "auto")
        if self.simulation_batch_size != "auto" and (
            not _is_int(self.simulation_batch_size) or self.simulation_batch_size < 1
        ):
            raise ValueError(
                '`simulation_batch_size` must be "auto" or a positive integer, got '
                f"{self.simulation_batch_size!r}."
            )
        if self.on_error not in ("raise", "nan"):
            raise ValueError(
                f'`on_error` must be "raise" or "nan", got {self.on_error!r}.'
            )

    def batch_size(self, num_simulations: int) -> int:
        if self.simulation_batch_size != "auto":
            return int(self.simulation_batch_size)
        if self.num_workers == 1:
            return 1
        return ceil(num_simulations / (4 * effective_n_jobs(self.num_workers)))


class _Batch(NamedTuple):
    """Parameter sets `theta[start : start + len(theta)]` and their call seeds."""

    theta: Array
    start: int
    seeds: Optional[np.ndarray]


class _Failure(NamedTuple):
    where: str
    note: str
    error: Exception


class _BatchResult(NamedTuple):
    output: Any
    num_failed: int
    first_failure: Optional[_Failure]


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
            f"{tuple(theta.shape)}. For one parameter set, pass `theta[None]`. A "
            "proposal over one parameter needs the event shape (1,), e.g. "
            "`BoxUniform(torch.zeros(1), torch.ones(1))`."
        )
    num_simulations = len(theta)
    if num_simulations == 0:
        if isinstance(theta, Tensor):
            return torch.empty(0, dtype=float32, device=theta.device)
        return np.empty(0)

    device = theta.device if isinstance(theta, Tensor) else None
    run = partial(
        _run_batched if settings.simulator_is_batched else _run_rows,
        simulator,
        on_error=settings.on_error,
    )
    batch_size = settings.batch_size(num_simulations)
    starts = range(0, num_simulations, batch_size)
    sizes = [min(batch_size, num_simulations - start) for start in starts]
    uses_threads = _check_backend(seed, settings.num_workers)
    call_seeds = [None] * len(starts)
    if seed is not None and settings.simulator_is_batched:
        call_seeds = _distinct_seeds(len(starts), seed)[:, None]
    elif seed is not None:
        call_seeds = np.split(_distinct_seeds(num_simulations, seed), starts[1:])

    random_state = _get_random_state() if seed is not None else None
    results: Generator[Any, None, None]
    if settings.num_workers == 1:
        batches = map(_Batch, _split(theta, batch_size), starts, call_seeds)
        results = (run(batch) for batch in batches)
    else:
        thetas = [_to_transport(part) for part in _split(theta, batch_size)]
        batches = map(_Batch, thetas, starts, call_seeds)
        worker_seeds = (
            [None] * len(starts) if uses_threads else _distinct_seeds(len(starts), seed)
        )
        parallel = Parallel(n_jobs=settings.num_workers, return_as="generator")
        results = cast(
            Generator,
            parallel(
                delayed(_run_in_worker)(run, batch, worker_seed)
                for batch, worker_seed in zip(batches, worker_seeds, strict=True)
            ),
        )

    outputs = []
    num_failed, first_failure = 0, None
    try:
        with tqdm(total=num_simulations, disable=not show_progress_bar) as pbar:
            for size, result in zip(sizes, results, strict=True):
                output = _from_transport(result.output)
                _check_num_outputs(output, size)
                outputs.append(output)
                num_failed += result.num_failed
                first_failure = first_failure or result.first_failure
                pbar.update(size)
    finally:
        # Closing joblib's generator cancels the simulations not yet run.
        results.close()
        if random_state is not None:
            _set_random_state(random_state)

    if first_failure is not None:
        _report_failures(first_failure, num_failed, num_simulations)
    x = _concatenate(outputs, fill_failed=num_failed > 0)
    if device is None:
        return x
    if isinstance(x, list):
        return [_to_device(row, device) for row in x]
    return _to_device(x, device)


def _to_device(value: Any, device: torch.device) -> Any:
    if not isinstance(value, Tensor):
        return value
    if device.type == "mps" and value.dtype == torch.float64:
        value = value.float()  # MPS has no float64.
    return value.to(device)


def _run_batched(simulator: Callable, batch: _Batch, on_error: OnError) -> _BatchResult:
    """Runs a batched simulator on one batch; a failure fails the whole batch."""
    if batch.seeds is not None:
        seed_all_backends(batch.seeds[0])
    try:
        return _BatchResult(_snapshot(simulator(batch.theta)), 0, None)
    except Exception as error:
        where = f"theta[{batch.start}:{batch.start + len(batch.theta)}]"
        hint = (
            "If it takes one parameter set per call, pass `simulator_is_batched=False`."
        )
        failure = _Failure(where, _failure_note(where, error, hint), error)
        if on_error == "raise":
            error.add_note(failure.note)
            raise
        return _BatchResult([None] * len(batch.theta), len(batch.theta), failure)


def _run_rows(simulator: Callable, batch: _Batch, on_error: OnError) -> _BatchResult:
    """Runs a simulator that is not batched on each parameter set of a batch."""
    outputs = []
    num_failed, first_failure = 0, None
    for i, theta in enumerate(batch.theta):
        if batch.seeds is not None:
            seed_all_backends(batch.seeds[i])
        try:
            outputs.append(_snapshot(simulator(theta)))
        except Exception as error:
            where = f"theta[{batch.start + i}] = {theta.tolist()}"
            failure = _Failure(where, _failure_note(where, error), error)
            if on_error == "raise":
                error.add_note(failure.note)
                raise
            outputs.append(None)
            num_failed += 1
            first_failure = first_failure or failure

    if not num_failed and isinstance(outputs[0], Tensor):
        try:
            return _BatchResult(torch.stack(outputs), 0, None)
        except RuntimeError:  # Outputs of different shapes.
            pass
    return _BatchResult(outputs, num_failed, first_failure)


def _snapshot(output: Any) -> Any:
    """Copies tensor, array, list and tuple outputs, which a simulator may reuse."""
    if isinstance(output, Tensor):
        return output.clone()
    if isinstance(output, np.ndarray):
        return output.copy()
    if type(output) in (list, tuple):
        return type(output)(_snapshot(item) for item in output)
    return output


def _run_in_worker(
    run: Callable, batch: _Batch, worker_seed: Optional[int]
) -> _BatchResult:
    # Without this, worker results would not follow the caller's seed.
    if worker_seed is not None:
        seed_all_backends(worker_seed)
    try:
        result = run(batch._replace(theta=_from_transport(batch.theta)))
    except Exception as error:
        sendable = _sendable(error)
        if sendable is error:
            raise
        raise sendable from None
    if result.first_failure is not None:
        failure = result.first_failure
        result = result._replace(
            first_failure=failure._replace(error=_sendable(failure.error))
        )
    return result._replace(output=_to_transport(result.output))


def _check_backend(seed: Optional[int], num_workers: int) -> bool:
    """Returns whether joblib runs tasks in several threads of this process at once.

    Raises if joblib's backend cannot stream results, or if threads would share one
    random state for a seeded run.
    """
    if num_workers == 1:
        return False
    backend, _ = get_active_backend()
    if not getattr(backend, "supports_return_generator", True):
        raise ValueError(
            f"joblib's {type(backend).__name__} cannot return results as they "
            "finish. Use joblib's default backend."
        )
    uses_threads = getattr(backend, "uses_threads", False)
    runs_threads = uses_threads and backend.effective_n_jobs(num_workers) > 1
    if seed is not None and runs_threads:
        raise ValueError(
            "`seed` needs process workers, but joblib is set to use threads, which "
            "share one random state. Use joblib's default backend, or pass no seed."
        )
    return runs_threads


def _sendable(error: Exception) -> Exception:
    """Returns `error`, or a RuntimeError with its message if it cannot be unpickled.

    An exception that fails to unpickle in the main process breaks joblib's pool,
    which loses the results of all simulations.
    """
    try:
        pickle.loads(pickle.dumps(error))
        return error
    except Exception:
        sendable = RuntimeError(f"{type(error).__name__}: {error}")
        for note in getattr(error, "__notes__", []):
            sendable.add_note(note)
        return sendable


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


def _report_failures(
    first_failure: _Failure, num_failed: int, num_simulations: int
) -> None:
    error = first_failure.error
    if num_failed == num_simulations:
        error.add_note(
            f"{first_failure.note} All {num_simulations} simulations failed."
        )
        raise error
    warnings.warn(
        f"{num_failed} of {num_simulations} simulations failed; their outputs are "
        "NaN, or None for outputs that are not numeric. First failure at "
        f"{first_failure.where}: {type(error).__name__}: {error}",
        UserWarning,
        stacklevel=4,
    )


def _failure_note(where: str, error: Exception, hint: str = "") -> str:
    note = f"The simulator failed for {where}."
    if isinstance(error, AttributeError) and "'Tensor' object" in str(error):
        note += " It receives tensors; convert them with `np.asarray(theta)`."
    return f"{note} {hint}".rstrip()


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _check_num_outputs(output: Any, num_parameter_sets: int) -> None:
    num_outputs = None  # One output without a batch dimension.
    if isinstance(output, (Tensor, np.ndarray)) and output.ndim > 0:
        num_outputs = output.shape[0]
    elif isinstance(output, (list, tuple)):
        num_outputs = len(output)
    if num_outputs != num_parameter_sets:
        got = (
            "a single output without a batch dimension"
            if num_outputs is None
            else f"{num_outputs} output{'' if num_outputs == 1 else 's'}"
        )
        plural = "" if num_parameter_sets == 1 else "s"
        sets = f"a batch of {num_parameter_sets} parameter set{plural}"
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


def _get_random_state() -> Dict[str, Any]:
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif hasattr(torch.cuda, "_lazy_seed_tracker"):
        # Before CUDA starts, torch queues CUDA seeds, which seeding overwrites.
        queue = torch.cuda._lazy_seed_tracker
        state["cuda_queue"] = (
            queue.manual_seed_all_cb,
            queue.manual_seed_cb,
            list(queue.call_order),
        )
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _set_random_state(state: Dict[str, Any]) -> None:
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "cuda_queue" in state:
        queue = torch.cuda._lazy_seed_tracker
        queue.manual_seed_all_cb, queue.manual_seed_cb, queue.call_order = state[
            "cuda_queue"
        ]
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _is_numeric(output: Any) -> bool:
    if isinstance(output, Tensor):
        return True
    try:
        return np.asarray(output).dtype.kind in "biufc"
    except (TypeError, ValueError):  # Ragged lists, or tensors without a numpy dtype.
        return False


def _concatenate(outputs: List[Any], fill_failed: bool) -> SimulationOutput:
    """Concatenates per-batch outputs along the first dimension.

    With `fill_failed`, failed outputs (None) become NaN if the others are numeric.
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
    if fill_failed:
        rows = _fill_failed(rows)
    if isinstance(rows[0], Tensor):
        try:
            return torch.stack(rows)
        except RuntimeError:  # Outputs of different shapes.
            return rows
    if all(_is_numeric(row) for row in rows):
        try:
            return np.asarray(rows)
        except (TypeError, ValueError):  # Different shapes, or no numpy dtype.
            return rows
    return rows


def _fill_failed(rows: List[Any]) -> List[Any]:
    """Replaces failed outputs (None) with NaN if the other outputs are numeric."""
    template = next((row for row in rows if row is not None), None)
    if isinstance(template, Tensor):
        floating = template.is_floating_point() or template.is_complex()
        dtype = template.dtype if floating else torch.float64
        nan = torch.full_like(template, float("nan"), dtype=dtype)
        # Rows of a partly failed worker task can sit on another device.
        return [nan if row is None else row.to(template.device, dtype) for row in rows]
    if template is None or not _is_numeric(template):
        return rows
    nan = np.full(np.shape(template), np.nan)
    return [nan if row is None else row for row in rows]
