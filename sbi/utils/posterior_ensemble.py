# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import warnings

from sbi.inference.posteriors.ensemble_posterior import (
    EnsemblePosterior,
    EnsemblePotential,
)

warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)

warnings.warn(
    """
    NeuralPosteriorEnsemble was renamed EnsemblePosterior and moved to 
    sbi.inference.posteriors.ensemble_posterior along with EnsemblePotential.
    sbi.utils.posterior_ensemble will be deprecated in v0.24.
    """,
    DeprecationWarning,
)


def NeuralPosteriorEnsemble(*args, **kwargs):
    warnings.warn(
        "NeuralPosteriorEnsemble was renamed EnsemblePosterior and moved to sbi.inference.posteriors.ensemble_posterior. sbi.utils.posterior_ensemble will be deprecated in v0.24.",
        DeprecationWarning,
    )
    return EnsemblePosterior(*args, **kwargs)


def EnsemblePotential(*args, **kwargs):
    warnings.warn(
        "EnsemblePotential was moved to sbi.inference.posteriors.ensemble_posterior. sbi.utils.posterior_ensemble will be deprecated in v0.24.",
        DeprecationWarning,
    )
    return EnsemblePotential(*args, **kwargs)
