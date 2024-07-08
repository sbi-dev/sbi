### Posterior object using vector field estimators
from typing import Optional

import torch
from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.neural_nets.vf_estimators import VectorFieldEstimator


class VectorFieldPosterior(NeuralPosterior):
    def __init__(
        self,
        vf_estimator: VectorFieldEstimator,
        prior: Distribution,
        max_sampling_batch_size: int = 10_000,
        device: Optional[str] = None,
        x_shape: Optional[torch.Size] = None,
        enable_transform: bool = True,
    ):
        raise NotImplementedError

    def sample():
        raise NotImplementedError
