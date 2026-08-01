# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# Underscored: no `__all__` here, so a bare name would leak into `import *`.
import warnings as _warnings
from typing import TYPE_CHECKING as _TYPE_CHECKING

from sbi.inference.trainers.npe.mnpe import MNPE  # noqa: F401
from sbi.inference.trainers.npe.npe_a import NPE_A  # noqa: F401
from sbi.inference.trainers.npe.npe_b import NPE_B  # noqa: F401
from sbi.inference.trainers.npe.npe_base import PosteriorEstimatorTrainer  # noqa: F401
from sbi.inference.trainers.npe.npe_c import NPE_C  # noqa: F401
from sbi.inference.trainers.npe.npe_pfn import NPE_PFN  # noqa: F401

NPE = NPE_C

_DEPRECATED_ALIASES = {
    "SNPE_A": "NPE_A",
    "SNPE_B": "NPE_B",
    "SNPE_C": "NPE_C",
    "SNPE": "NPE_C",
}


if _TYPE_CHECKING:
    SNPE_A = NPE_A
    SNPE_B = NPE_B
    SNPE = SNPE_C = NPE_C


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
