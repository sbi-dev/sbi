# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pickle
import random
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from joblib import Parallel, delayed, effective_n_jobs
from joblib.parallel import get_active_backend
from torch import Tensor, float32

from sbi.sbi_types import Array, Proposal, SimulationOutput
from sbi.utils.pbar import tqdm
from sbi.utils.sbiutils import seed_random_states


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
            can be used. It receives the sampled parameters as a tensor, on the
            proposal's device with one worker and on the CPU with more, and must
            return numbers of one shape per parameter set.
        proposal: Probability distribution that the parameters $\theta$ are sampled
            from.
        num_simulations: Number of simulations that are run.
        num_workers: Number of parallel joblib workers; -1 uses all CPUs, -2 all
            but one, and so on.
        simulation_batch_size: Number of parameter sets per call of a batched
            simulator. If None, all parameter sets are simulated in one call.
        seed: Seed for reproducibility. It seeds the global random state before
            the parameters are sampled, and each simulator call, see
            :func:`simulate_from_theta`.
        show_progress_bar: Whether to show a progress bar for simulating. This will not
            affect whether there will be a progressbar while drawing samples from the
            proposal.
        simulator_is_batched: Whether the simulator takes a batch of parameter sets
            per call. If False, it is called once per parameter set.

    Returns:
        Sampled parameters $\theta$ and simulation outputs $x$, both float32
        tensors on the device of $\theta$. One number per simulation gives $x$ of
        shape (num_simulations, 1).
    """
    if num_simulations == 0:
        return torch.tensor([], dtype=float32), torch.tensor([], dtype=float32)

    seed = None if seed is None else int(seed)
    seed_random_states(seed)
    theta = proposal.sample((num_simulations,))
    if theta.ndim < 2:
        raise ValueError(
            f"The proposal samples parameters of shape {tuple(theta.shape)}, but "
            "`simulate_for_sbi` needs (num_simulations, *parameter_shape). A "
            "proposal over one parameter needs the event shape (1,), e.g. "
            "`BoxUniform(torch.zeros(1), torch.ones(1))`."
        )
    x = simulate_from_theta(
        simulator,
        theta,
        num_workers=num_workers,
        simulation_batch_size=simulation_batch_size,
        simulator_is_batched=simulator_is_batched,
        seed=seed,
        show_progress_bar=show_progress_bar,
    )
    try:
        if isinstance(x, list):  # E.g. tensors and arrays mixed.
            x = torch.stack([torch.as_tensor(row).cpu() for row in x])
        x = torch.as_tensor(x)
        if x.is_complex():
            raise TypeError("complex simulator outputs")
        x = x.to(dtype=float32, device=theta.device).detach()
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError(
            "`simulate_for_sbi` needs real numeric simulator outputs of one shape. "
            "For other outputs, such as file paths, use `simulate_from_theta`."
        ) from error

    return torch.as_tensor(theta, dtype=float32), x[:, None] if x.ndim == 1 else x


def simulate_from_theta(
    simulator: Callable,
    theta: Union[Array, Sequence[Any]],
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
            a batch of parameter sets and returns one output per parameter set: a
            tensor or array with that leading dimension, or a list of that length.
        theta: Parameters of shape (num_simulations, \*parameter_shape), as a
            tensor or numpy array. Other array-likes become numpy arrays. The
            simulator receives parameters of the same type; with more than one
            worker, tensors are on the CPU.
        num_workers: Number of parallel joblib workers; -1 uses all CPUs, -2 all
            but one, and so on.
        simulation_batch_size: Number of parameter sets per call of a batched
            simulator. If None, all parameter sets are simulated in one call.
        simulator_is_batched: Whether the simulator takes a batch of parameter sets
            per call. If False, it is called once per parameter set.
        seed: Seed for reproducibility. Every simulator call gets its own seed
            derived from `seed`, for the random states of Python, numpy and
            PyTorch on the CPU. So for noise drawn on the CPU, the outputs do not
            depend on `num_workers`. PyTorch's GPU generators are seeded once
            before the first call, so noise drawn on a GPU is reproducible with one
            worker only. Afterwards, the CPU random states are seeded with another
            seed derived from `seed`. Without a seed, one worker uses the current
            random state; more workers seed every call from numpy's global random
            state. Seeding needs joblib's default process workers: with a thread
            backend, a seed raises an error.
        show_progress_bar: Whether to show a progress bar.

    Returns:
        Simulation outputs in the order of `theta`, one per parameter set. Tensors
        of one shape form one tensor, arrays and numbers of one shape one numpy
        array, with a leading dimension of num_simulations. All other outputs,
        such as file paths, come back as a list.
    """
    if not _is_int(num_workers) or num_workers == 0:
        raise ValueError(
            "`num_workers` must be a nonzero integer, e.g. 4, or -1 for all CPUs; "
            f"got {num_workers!r}."
        )
    if simulation_batch_size is not None and (
        not _is_int(simulation_batch_size) or simulation_batch_size < 1
    ):
        raise ValueError(
            "`simulation_batch_size` must be a positive integer or None, got "
            f"{simulation_batch_size!r}."
        )
    if not simulator_is_batched and simulation_batch_size != 1:
        raise ValueError(
            f"`simulation_batch_size={simulation_batch_size}` needs a simulator that "
            "takes a batch of parameter sets per call; pass "
            "`simulator_is_batched=True` if yours does."
        )
    if not isinstance(theta, (Tensor, np.ndarray)):
        theta = np.asarray(theta)
    if theta.ndim < 2:
        raise ValueError(
            "`theta` must have shape (num_simulations, *parameter_shape), got "
            f"{tuple(theta.shape)}. For one parameter set, pass `theta[None]`; for "
            "one parameter, `theta[:, None]`."
        )
    if len(theta) == 0:
        return theta.new_empty(0) if isinstance(theta, Tensor) else np.empty(0)
    seed = None if seed is None else int(seed)
    num_workers = effective_n_jobs(num_workers)  # E.g. 1 for -1 on one CPU.
    uses_threads = num_workers > 1 and _check_backend(seed)
    # A copy, as a simulator may change its input in place.
    theta = theta.clone() if isinstance(theta, Tensor) else theta.copy()

    # One entry of `thetas` per simulator call.
    all_in_one_call = simulation_batch_size is None
    call_size = len(theta) if all_in_one_call else int(simulation_batch_size)
    starts = range(0, len(theta), call_size)
    if simulator_is_batched:
        thetas: List[Array] = [theta[start : start + call_size] for start in starts]
    else:
        thetas = list(theta)
    if seed is not None:
        # Children of `seed`, so that no draw repeats those that sampled `theta`.
        children = np.random.SeedSequence(seed).spawn(2)
        seed_after_call, gpu_seed = (int(c.generate_state(1)[0]) for c in children)
        torch.manual_seed(gpu_seed)  # Seeds the generators of all devices.
    seeds: Sequence[Optional[int]] = [None] * len(thetas)
    # Without seeds, worker processes would not follow the caller's random state.
    if seed is not None or (num_workers > 1 and not uses_threads):
        seeds = _call_seeds(len(thetas), seed)

    calls = zip(thetas, starts, seeds, strict=True)
    outputs: Iterable[Any]
    if num_workers == 1:
        outputs = (_call(simulator, simulator_is_batched, *call) for call in calls)
    else:
        # No memory maps: joblib would pass large arrays read-only, and the calls
        # share no parameters.
        parallel = Parallel(n_jobs=num_workers, return_as="generator", max_nbytes=None)
        # Copies, as a pickled view of a tensor sends all of `theta`.
        tasks = (
            (t.detach().to("cpu", copy=True) if isinstance(t, Tensor) else t, *rest)
            for t, *rest in calls
        )
        run = _call if uses_threads else _call_in_worker
        outputs = parallel(
            delayed(run)(simulator, simulator_is_batched, *task) for task in tasks
        )
    collected = []
    try:
        with tqdm(total=len(theta), disable=not show_progress_bar) as pbar:
            for theta_of_call, output in zip(thetas, outputs, strict=True):
                collected.append(output)
                pbar.update(len(theta_of_call) if simulator_is_batched else 1)
    except pickle.PicklingError as error:
        error.add_note(
            "Workers need the simulator and its inputs pickled. See the FAQ on "
            "pickling errors, or pass `num_workers=1`."
        )
        raise

    if seed is not None:
        _seed_cpu_random_states(seed_after_call)
    return _join(collected, simulator_is_batched)


def _call(
    simulator: Callable,
    simulator_is_batched: bool,
    theta: Array,
    start: int,
    seed: Optional[int],
) -> Any:
    """Runs one simulator call for `theta`, which starts at row `start`."""
    if seed is not None:
        _seed_cpu_random_states(seed)
    try:
        output = simulator(theta)
        if output is None:
            raise ValueError("The simulator returned None. Does it miss a `return`?")
    except Exception as error:
        if simulator_is_batched:
            note = (
                f"The simulator failed for theta[{start}:{start + len(theta)}]. It "
                "was called with a batch of parameter sets; if it takes one per "
                "call, pass `simulator_is_batched=False`."
            )
        else:
            note = f"The simulator failed for theta[{start}] = {theta.tolist()}."
        error.add_note(note)
        raise

    if simulator_is_batched:
        sized = isinstance(output, list) or (
            isinstance(output, (Tensor, np.ndarray)) and output.ndim > 0
        )
        if not sized or len(output) != len(theta):
            got = f"a {type(output).__name__}"
            got += f" of length {len(output)}" if sized else ""
            raise ValueError(
                f"The simulator returned {got} for a batch of {len(theta)} parameter "
                "sets. It must return one output per parameter set: a tensor, array "
                "or list. If it takes one parameter set per call, pass "
                "`simulator_is_batched=False`."
            )
    return _copy(output)


def _call_in_worker(*args: Any) -> Any:
    try:
        return _call(*args)
    except Exception as error:
        try:
            pickle.loads(pickle.dumps(error))
        except Exception:
            # Such an exception would break joblib's worker pool on its way back.
            notes = " ".join(getattr(error, "__notes__", []))
            raise RuntimeError(f"{type(error).__name__}: {error}. {notes}") from None
        raise


def _copy(output: Any) -> Any:
    """Copies tensors, arrays, lists and tuples, as a simulator may reuse them."""
    if isinstance(output, Tensor):
        return output.clone()
    if isinstance(output, np.ndarray):
        return output.copy()
    if type(output) in (list, tuple):
        return type(output)(_copy(item) for item in output)
    return output


def _join(outputs: List[Any], simulator_is_batched: bool) -> SimulationOutput:
    """Joins the outputs of all calls into a tensor, an array, or else a list."""

    def all_alike(values: List[Any], types: Any, key: Callable) -> bool:
        return all(isinstance(value, types) for value in values) and (
            len({key(value) for value in values}) == 1
        )

    if simulator_is_batched:
        if all_alike(outputs, Tensor, lambda batch: (batch.shape[1:], batch.device)):
            return torch.cat(outputs)
        if all_alike(outputs, np.ndarray, lambda batch: batch.shape[1:]):
            return np.concatenate(outputs)
        outputs = [row for batch in outputs for row in batch]
    if all_alike(outputs, Tensor, lambda row: (row.shape, row.device)):
        return torch.stack(outputs)
    numbers = (np.ndarray, np.number, np.bool_, int, float, complex)
    if all_alike(outputs, numbers, np.shape):
        return np.asarray(outputs)
    return outputs


def _check_backend(seed: Optional[int]) -> bool:
    """Returns whether joblib runs calls in several threads of this process at once.

    Raises if joblib's backend cannot stream results, or if threads would share one
    random state for a seeded run.
    """
    backend, _ = get_active_backend()
    if not getattr(backend, "supports_return_generator", True):
        raise ValueError(
            f"joblib's {type(backend).__name__} cannot return results as they "
            "finish. Use joblib's default backend."
        )
    uses_threads = getattr(backend, "uses_threads", False)
    if seed is not None and uses_threads:
        raise ValueError(
            "`seed` needs process workers, but joblib is set to use threads, which "
            "share one random state. Use joblib's default backend, or pass no seed."
        )
    return uses_threads


def _seed_cpu_random_states(seed: int) -> None:
    """Seeds Python's `random`, numpy and PyTorch's CPU generator.

    Unlike `seed_all_backends`, it leaves cuDNN alone and skips PyTorch's GPU
    generators, which take ten times as long to seed.
    """
    random.seed(seed)
    np.random.seed([seed & 0xFFFFFFFF, seed >> 32])  # It takes 32-bit words.
    torch.default_generator.manual_seed(seed)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _call_seeds(num_seeds: int, seed: Optional[int]) -> List[int]:
    """Returns one 64-bit seed per call; call i gets the same seed for any `num_seeds`.

    With `seed=None`, the seeds follow the global numpy random state.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    return np.random.default_rng(seed).integers(2**63, size=num_seeds).tolist()
