# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from contextlib import contextmanager
import threading


_pbar_context = threading.local()


@contextmanager
def nested_pbar_context():
    _pbar_context.active = getattr(_pbar_context, "active", 0) + 1
    try:
        yield
    finally:
        _pbar_context.active -= 1


def is_nested():
    return getattr(_pbar_context, "active", 0) > 0
