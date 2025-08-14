# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from abc import ABC, abstractmethod

from torch import Tensor, nn


class VectorFieldNet(nn.Module, ABC):
    """Abstract base class for vector field estimation networks.

    Used by both flow matching and score matching approaches.
    """

    @abstractmethod
    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass to compute the vector field.

        Args:
            theta: Parameters
            x: Conditioning information
            t: Time parameter (scalar or batched)

        Returns:
            Vector field evaluation at the provided points
        """
        pass
