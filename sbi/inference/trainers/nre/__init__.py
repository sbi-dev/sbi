# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# Underscored: no `__all__` here, so a bare name would leak into `import *`.
import warnings as _warnings

from sbi.inference.trainers.nre.bnre import BNRE
from sbi.inference.trainers.nre.nre_a import NRE_A
from sbi.inference.trainers.nre.nre_b import NRE_B
from sbi.inference.trainers.nre.nre_c import NRE_C

NRE = NRE_B

_DEPRECATED_ALIASES = {
    "AALR": "NRE_A",
    "SNRE_A": "NRE_A",
    "SRE": "NRE_B",
    "SNRE_B": "NRE_B",
    "CNRE": "NRE_C",
    "SNRE_C": "NRE_C",
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
