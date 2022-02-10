from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from sbi.inference.abc.mcabc import MCABC
from sbi.inference.abc.smcabc import SMCABC
from sbi.inference.base import (  # noqa: F401
    NeuralInference,
    check_if_proposal_has_default_x,
    infer,
    simulate_for_sbi,
)
from sbi.inference.snle.mnle import MNLE
from sbi.inference.snle.snle_a import SNLE_A
from sbi.inference.snpe.snpe_a import SNPE_A
from sbi.inference.snpe.snpe_b import SNPE_B
from sbi.inference.snpe.snpe_c import SNPE_C  # noqa: F401
from sbi.inference.snre import SNRE, SNRE_A, SNRE_B  # noqa: F401
from sbi.utils.user_input_checks import prepare_for_sbi

SNL = SNLE = SNLE_A
_snle_family = ["SNL"]


SNPE = APT = SNPE_C
_snpe_family = ["SNPE_A", "SNPE_C", "SNPE", "APT"]


SRE = SNRE_B
AALR = SNRE_A
_snre_family = ["SNRE_A", "AALR", "SNRE_B", "SNRE", "SRE"]

ABC = MCABC
SMC = SMCABC
_abc_family = ["ABC", "MCABC", "SMC", "SMCABC"]


__all__ = _snpe_family + _snre_family + _snle_family + _abc_family

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.inference.potentials.likelihood_based_potential import (
    likelihood_estimator_based_potential,
    mixed_likelihood_estimator_based_potential,
)
from sbi.inference.potentials.posterior_based_potential import (
    posterior_estimator_based_potential,
)
from sbi.inference.potentials.ratio_based_potential import (
    ratio_estimator_based_potential,
)
