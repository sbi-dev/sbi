# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import TYPE_CHECKING

from sbi.inference.potentials.likelihood_based_potential import (
    likelihood_estimator_based_potential,
)
from sbi.inference.potentials.posterior_based_potential import (
    posterior_estimator_based_potential,
)
from sbi.inference.potentials.ratio_based_potential import (
    ratio_estimator_based_potential,
)
from sbi.inference.potentials.vector_field_potential import (
    vector_field_estimator_based_potential,
)
from sbi.utils.removed_names import removed_names_getattr

__all__ = [
    "likelihood_estimator_based_potential",
    "posterior_estimator_based_potential",
    "ratio_estimator_based_potential",
    "vector_field_estimator_based_potential",
]

if not TYPE_CHECKING:
    __getattr__ = removed_names_getattr(__name__)
