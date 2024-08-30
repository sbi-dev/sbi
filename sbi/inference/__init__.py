from sbi.inference.abc import MCABC, SMCABC
from sbi.inference.trainers.base import (
    NeuralInference,  # noqa: F401
    check_if_proposal_has_default_x,
    infer,
)
from sbi.inference.trainers.fmpe import FMPE
from sbi.inference.trainers.nle import MNLE, NLE_A
from sbi.inference.trainers.npe import NPE_A, NPE_B, NPE_C  # noqa: F401
from sbi.inference.trainers.npse import NPSE
from sbi.inference.trainers.nre import BNRE, NRE_A, NRE_B, NRE_C  # noqa: F401

SNL = SNLE = SNLE_A = NLE = NLE_A
_nle_family = ["NLE"]


SNPE_A = NPE_A
SNPE_B = NPE_B
SNPE = APT = SNPE_C = NPE = NPE_C
_npe_family = ["NPE_A", "NPE_C"]


SRE = SNRE = SNRE_B = NRE = NRE_B
AALR = SNRE_A = NRE_A
CNRE = SNRE_C = NRE_C
_nre_family = ["NRE_A", "NRE_B", "NRE_C", "BNRE"]

ABC = MCABC
SMC = SMCABC
_abc_family = ["ABC", "MCABC", "SMC", "SMCABC"]


__all__ = _npe_family + _nre_family + _nle_family + _abc_family + ["FMPE", "NPSE"]

from sbi.inference.posteriors import (
    DirectPosterior,
    EnsemblePosterior,
    ImportanceSamplingPosterior,
    MCMCPosterior,
    RejectionPosterior,
    ScorePosterior,
    VIPosterior,
)
from sbi.inference.potentials import (
    likelihood_estimator_based_potential,
    mixed_likelihood_estimator_based_potential,
    posterior_estimator_based_potential,
    ratio_estimator_based_potential,
)
from sbi.utils.simulation_utils import simulate_for_sbi
