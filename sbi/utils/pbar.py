# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Thread-local tracking of nested progress bars.

Samplers that draw from a proposal, e.g. `accept_reject_sample` or SIR, show a
progress bar of their own. If the proposal is itself a sampler with a progress bar,
e.g. the diffusion sampler inside `accept_reject_sample`, both bars interleave in the
terminal. To show only the outermost bar, the outer sampler wraps every proposal call
in `nested_pbar_context()`, and every sampler disables its bar if `is_nested()`.
"""

import threading
from contextlib import contextmanager
from typing import Iterator

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
