from sbi.inference.abc import MCABC, SMCABC
from sbi.inference.trainers.base import (
    NeuralInference,  # noqa: F401
    check_if_proposal_has_default_x,
    infer,
    simulate_for_sbi,
)
from sbi.inference.trainers.fmpe import FMPE
from sbi.inference.trainers.npse.npse import NPSE
from sbi.inference.trainers.snle import MNLE, SNLE_A
from sbi.inference.trainers.snpe import SNPE_A, SNPE_B, SNPE_C  # noqa: F401
from sbi.inference.trainers.snre import BNRE, SNRE, SNRE_A, SNRE_B, SNRE_C  # noqa: F401

SNL = SNLE = SNLE_A
_snle_family = ["SNL"]


SNPE = APT = SNPE_C
_snpe_family = ["SNPE_A", "SNPE_C", "SNPE", "APT"]


SRE = SNRE_B
AALR = SNRE_A
CNRE = NREC = SNRE_C
_snre_family = ["SNRE_A", "AALR", "SNRE_B", "SNRE", "SRE", "SNRE_C", "CNRE", "NREC"]

ABC = MCABC
SMC = SMCABC
_abc_family = ["ABC", "MCABC", "SMC", "SMCABC"]


__all__ = _snpe_family + _snre_family + _snle_family + _abc_family

from sbi.inference.posteriors import (
    DirectPosterior,
    EnsemblePosterior,
    ImportanceSamplingPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
)
from sbi.inference.potentials import (
    likelihood_estimator_based_potential,
    mixed_likelihood_estimator_based_potential,
    posterior_estimator_based_potential,
    ratio_estimator_based_potential,
)
