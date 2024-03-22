from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class VectorFieldEstimator(nn.Module):
    r"""Base class for vector field (e.g., score and ODE flow) estimators.

    The density estimator class is a wrapper around neural networks that
    allows to evaluate the `vector_field`, and provide the `loss` of $\theta,x$
    pairs. Here $\theta$ would be the `input` and $x$ would be the `condition`.

    Note:
        We assume that the input to the density estimator is a tensor of shape
        (batch_size, input_size), where input_size is the dimensionality of the input.
        The condition is a tensor of shape (batch_size, *condition_shape), where
        condition_shape is the shape of the condition tensor.

    """
    def __init__(self, net: nn.Module, condition_shape: torch.Size) -> None:
        r"""Base class for vector field estimators.

        Args:
            net: Neural network.
            condition_shape: Shape of the condition. If not provided, it will assume a
                            1D input.
        """
        super().__init__()
        self.net = net
        self._condition_shape = condition_shape

    def forward(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on of shape (batch_size, input_size).
            condition: Conditions of shape (batch_size, *condition_shape).

        Returns:
            Loss of shape (batch_size,)
        """

        raise NotImplementedError


    def _check_condition_shape(self, condition: Tensor):
        r"""This method checks whether the condition has the correct shape.

        Args:
            condition: Conditions of shape (*batch_shape, *condition_shape).

        Raises:
            ValueError: If the condition has a dimensionality that does not match
                        the expected input dimensionality.
            ValueError: If the shape of the condition does not match the expected
                        input dimensionality.
        """
        if len(condition.shape) < len(self._condition_shape):
            raise ValueError(
                f"Dimensionality of condition is to small and does not match the\
                expected input dimensionality {len(self._condition_shape)}, as provided\
                by condition_shape."
            )
        else:
            condition_shape = condition.shape[-len(self._condition_shape) :]
            if tuple(condition_shape) != tuple(self._condition_shape):
                raise ValueError(
                    f"Shape of condition {tuple(condition_shape)} does not match the \
                    expected input dimensionality {tuple(self._condition_shape)}, as \
                    provided by condition_shape. Please reshape it accordingly."
                )
