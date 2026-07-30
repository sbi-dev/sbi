# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings

from sbi.inference.abc import MCABC, SMCABC
from sbi.inference.trainers.base import (
    NeuralInference,  # noqa: F401
    check_if_proposal_has_default_x,
    infer,
)
from sbi.inference.trainers.marginal import MarginalTrainer
from sbi.inference.trainers.nle import MNLE, NLE_A
from sbi.inference.trainers.npe import MNPE, NPE_A, NPE_B, NPE_C, NPE_PFN  # noqa: F401
from sbi.inference.trainers.nre import BNRE, NRE_A, NRE_B, NRE_C  # noqa: F401
from sbi.inference.trainers.vfpe import FMPE, NPSE

NLE = NLE_A
_nle_family = ["NLE_A", "MNLE"]

NPE = NPE_C
_npe_family = ["NPE_A", "NPE_B", "NPE_C", "NPE_PFN", "MNPE"]

NRE = NRE_B
_nre_family = ["NRE_A", "NRE_B", "NRE_C", "BNRE"]

_abc_family = ["MCABC", "SMCABC"]

_DEPRECATED_ALIASES = {
    "SNL": "NLE_A",
    "SNLE": "NLE_A",
    "SNLE_A": "NLE_A",
    "SNPE_A": "NPE_A",
    "SNPE_B": "NPE_B",
    "SNPE": "NPE_C",
    "SNPE_C": "NPE_C",
    "APT": "NPE_C",
    "SRE": "NRE_B",
    "SNRE": "NRE_B",
    "SNRE_B": "NRE_B",
    "AALR": "NRE_A",
    "SNRE_A": "NRE_A",
    "CNRE": "NRE_C",
    "SNRE_C": "NRE_C",
    "ABC": "MCABC",
    "SMC": "SMCABC",
}


def __getattr__(name: str):
    if name in _DEPRECATED_ALIASES:
        canonical = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"`sbi.inference.{name}` is deprecated since sbi v0.27.0 and will be "
            f"removed in v0.28.0. Use `sbi.inference.{canonical}` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return globals()[canonical]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from sbi.inference.posteriors import (
    DirectPosterior,
    EnsemblePosterior,
    ImportanceSamplingPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
    VectorFieldPosterior,
)
from sbi.inference.potentials import (
    likelihood_estimator_based_potential,
    mixed_likelihood_estimator_based_potential,
    posterior_estimator_based_potential,
    ratio_estimator_based_potential,
    vector_field_estimator_based_potential,
)
from sbi.utils.simulation_utils import simulate_for_sbi

__all__ = (
    _npe_family
    + _nre_family
    + _nle_family
    + _abc_family
    + [
        "FMPE",
        "MarginalTrainer",
        "NPSE",
        "DirectPosterior",
        "EnsemblePosterior",
        "ImportanceSamplingPosterior",
        "MCMCPosterior",
        "RejectionPosterior",
        "VIPosterior",
        "VectorFieldPosterior",
        "simulate_for_sbi",
        "likelihood_estimator_based_potential",
        "mixed_likelihood_estimator_based_potential",
        "posterior_estimator_based_potential",
        "ratio_estimator_based_potential",
        "vector_field_estimator_based_potential",
    ]
)
