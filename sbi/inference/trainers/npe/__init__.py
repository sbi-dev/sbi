# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import TYPE_CHECKING

from sbi.inference.trainers.npe.mnpe import MNPE  # noqa: F401
from sbi.inference.trainers.npe.npe_a import NPE_A  # noqa: F401
from sbi.inference.trainers.npe.npe_b import NPE_B  # noqa: F401
from sbi.inference.trainers.npe.npe_base import PosteriorEstimatorTrainer  # noqa: F401
from sbi.inference.trainers.npe.npe_c import NPE_C  # noqa: F401
from sbi.inference.trainers.npe.npe_pfn import NPE_PFN  # noqa: F401
from sbi.utils.removed_names import removed_names_getattr

NPE = NPE_C

if not TYPE_CHECKING:  # Hidden so that type checkers still flag unknown names.
    __getattr__ = removed_names_getattr(__name__)
