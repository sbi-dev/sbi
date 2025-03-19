
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from sbi.neural_nets.estimators.base import ConditionalEstimator


class ConditionalVectorFieldEstimator(ConditionalEstimator):
    r"""Base class for vector field (e.g., score and ODE flow) estimators.

    The vector field estimator class is a wrapper around neural networks that allows to
    evaluate the `vector_field`, and provide the `loss` of $\theta,x$ pairs. Here
    $\theta$ would be the `input` and $x$ would be the `condition`.

    Note:
        We assume that the input to the density estimator is a tensor of shape
        (sample_dim, batch_dim, *input_shape), where input_shape is the dimensionality
        of the input. The condition is a tensor of shape (batch_dim, *condition_shape),
        where condition_shape is the shape of the condition tensor.

    """

    def __init__(
        self, net: nn.Module, input_shape: torch.Size, condition_shape: torch.Size
    ) -> None:
        r"""Base class for vector field estimators.

        Args:
            net: Neural network.
            condition_shape: Shape of the condition. If not provided, it will assume a
                            1D input.
        """
        super().__init__(input_shape, condition_shape)
        self.net = net

    @abstractmethod
    def forward(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        """Forward pass of the score estimator.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """

    @abstractmethod
    def ode_fn(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        """ODE flow function of the vector field estimator.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.
            t: Time.
        
        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """

    def score(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        """Score function of the vector field estimator.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.
            t: Time.

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        raise NotImplementedError("This method should be implemented by sub-classes.")
