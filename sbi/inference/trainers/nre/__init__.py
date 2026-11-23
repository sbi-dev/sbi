# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import TYPE_CHECKING

from sbi.inference.trainers.nre.bnre import BNRE
from sbi.inference.trainers.nre.nre_a import NRE_A
from sbi.inference.trainers.nre.nre_b import NRE_B
from sbi.inference.trainers.nre.nre_c import NRE_C
from sbi.utils.removed_names import removed_names_getattr

NRE = NRE_B

if not TYPE_CHECKING:
    __getattr__ = removed_names_getattr(__name__)
