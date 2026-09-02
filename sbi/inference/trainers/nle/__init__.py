# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings as _warnings

from sbi.inference.trainers.nle.mnle import MNLE  # noqa: F401
from sbi.inference.trainers.nle.nle_a import NLE_A  # noqa: F401

NLE = NLE_A

_DEPRECATED_ALIASES = {
    "SNLE": "NLE_A",
    "SNLE_A": "NLE_A",
}


def __getattr__(name: str):
    if name in _DEPRECATED_ALIASES:
        canonical = _DEPRECATED_ALIASES[name]
        _warnings.warn(
            f"`{__name__}.{name}` is deprecated since sbi v0.27.0 and will be "
            f"removed in v0.28.0. Use `{__name__}.{canonical}` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return globals()[canonical]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
