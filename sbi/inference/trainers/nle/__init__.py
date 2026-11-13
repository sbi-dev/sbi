# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import TYPE_CHECKING

from sbi.inference.trainers.nle.mnle import MNLE  # noqa: F401
from sbi.inference.trainers.nle.nle_a import NLE_A  # noqa: F401
from sbi.utils.removed_names import removed_names_getattr

NLE = NLE_A

if not TYPE_CHECKING:
    __getattr__ = removed_names_getattr(__name__)
