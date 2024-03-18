from typing import Tuple

import torch
from torch import Tensor, nn

from sbi.neural_nets.vf_estimators import VectorFieldEstimator
from sbi.types import Shape

class ScoreEstimator(VectorFieldEstimator):
    r"""Score estimator for denoising diffusion probabilistic models (and similar).

    Neural net objects already have a .log_prob() and .sample() method, so here we just
    wrap them and add the .loss() method.
    """

    def __init__(self, net: nn.Module, condition_shape: torch.Size) -> None:
        super().__init__(net, condition_shape)

    def forward():
        raise NotImplementedError
    
    def loss(self, input: Tensor, condition: Tensor) -> Tensor:        
        raise NotImplementedError
