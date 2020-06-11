from typing import List, Type  # noqa: F401
from sbi.inference.base import NeuralInference, infer  # noqa: F401
from sbi.user_input.user_input_checks import prepare_for_sbi

from sbi.inference.snle.snle_a import SNLE_A


# Unimplemented: don't export
# from sbi.inference.snpe.snpe_a import SNPE_A
from sbi.inference.snpe.snpe_b import SNPE_B
from sbi.inference.snpe.snpe_c import SNPE_C  # noqa: F401

from sbi.inference.snre import SNRE, SNRE_A, SNRE_B  # noqa: F401


SNL = SNLE = SNLE_A
_snle_family = ["SNL"]


SNPE = APT = SNPE_C
_snpe_family = ["SNPE_C", "SNPE", "APT"]


SRE = SNRE_B
AALR = SNRE_A
_snre_family = ["SNRE_A", "AALR", "SNRE_B", "SNRE", "SRE"]


__all__ = _snpe_family + _snre_family + _snle_family
