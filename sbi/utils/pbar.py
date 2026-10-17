# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Progress bar helpers shared by the samplers.

Samplers that draw from a proposal, e.g. `accept_reject_sample` or SIR, show a
progress bar of their own. If the proposal is itself a sampler with a progress bar,
e.g. the diffusion sampler inside `accept_reject_sample`, both bars interleave in the
terminal. To show only the outermost bar, the outer sampler wraps every proposal call
in `nested_pbar_context()`, and every sampler disables its bar if `is_nested()`.

All sampling bars build their description with `sampling_desc()`.
"""

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

_state = threading.local()


@contextmanager
def nested_pbar_context() -> Iterator[None]:
    """Marks the enclosed code as running inside an outer sampler.

    The nesting depth is counted per thread. Contexts can be entered repeatedly;
    `is_nested()` stays True until the outermost context exits.
    """
    _state.depth = getattr(_state, "depth", 0) + 1
    try:
        yield
    finally:
        _state.depth -= 1


def is_nested() -> bool:
    """Returns True if the current thread is inside a `nested_pbar_context()`."""
    return getattr(_state, "depth", 0) > 0


def sampling_desc(
    num_samples: int,
    method: str,
    *,
    num_xos: int = 1,
    num_chains: Optional[int] = None,
    num_workers: Optional[int] = None,
    num_steps: Optional[int] = None,
) -> str:
    """Returns the progress bar description of a sampler.

    Example: `"Drawing 100 samples for each of 5 observations [slice_np, 20 chains,
    4 workers]"`.

    Args:
        num_samples: Number of samples per observation.
        method: Name of the sampling method, e.g. `"rejection"` or `"slice_np"`.
        num_xos: Number of observations. Shown only if larger than one.
        num_chains: Number of MCMC chains per observation.
        num_workers: Number of parallel workers.
        num_steps: Number of time steps of the sampler, e.g. of a diffusion.

    Returns:
        The description, with the counts that are given in brackets after the method.
    """
    desc = f"Drawing {_count(num_samples, 'sample')}"
    if num_xos > 1:
        desc += f" for each of {num_xos} observations"
    details = [method]
    for count, noun in (
        (num_chains, "chain"),
        (num_workers, "worker"),
        (num_steps, "step"),
    ):
        if count is not None:
            details.append(_count(count, noun))
    return f"{desc} [{', '.join(details)}]"


def _count(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
