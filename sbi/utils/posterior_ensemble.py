# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import warnings
from typing import List, Optional, Union

from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior
from sbi.inference.posteriors.ensemble_posterior import (
    EnsemblePotential as EnsemblePotentialMoved,
)
from sbi.sbi_types import TorchTransform

warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)

warnings.warn(
    """
    NeuralPosteriorEnsemble was renamed EnsemblePosterior and moved to
    sbi.inference.posteriors.ensemble_posterior along with EnsemblePotential.
    sbi.utils.posterior_ensemble will be deprecated in v0.24.
    """,
    DeprecationWarning,
    stacklevel=2,
)


def NeuralPosteriorEnsemble(
    posteriors: List,
    weights: Optional[Union[List[float], Tensor]] = None,
    theta_transform: Optional[TorchTransform] = None,
):
    warnings.warn(
        "NeuralPosteriorEnsemble was renamed EnsemblePosterior and moved to \
        sbi.inference.posteriors.ensemble_posterior. sbi.utils.posterior_ensemble \
        will be deprecated in v0.24.",
        DeprecationWarning,
        stacklevel=2,
    )
    return EnsemblePosterior(posteriors, weights, theta_transform)


def EnsemblePotential(
    potential_fns: List,
    weights: Tensor,
    prior: Distribution,
    x_o: Optional[Tensor],
    device: str = "cpu",
):
    warnings.warn(
        "EnsemblePotential was moved to sbi.inference.posteriors.ensemble_posterior. \
        sbi.utils.posterior_ensemble will be deprecated in v0.24.",
        DeprecationWarning,
        stacklevel=2,
    )
    return EnsemblePotentialMoved(potential_fns, weights, prior, x_o, device)
