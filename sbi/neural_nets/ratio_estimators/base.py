import torch
from torch import Tensor, nn


class RatioEstimator(nn.Module):
    r"""Base class for ratio estimators.

    The ratio estimator class is a wrapper around neural networks that
    allows to evaluate the classifier logits  of $\theta,x$ pairs. Here $\theta$ would
    be the `input` and $x$ would be the `condition`.

    Note:
        We assume that the input to the ratio estimator is a tensor of shape
        (batch_size, input_size), where input_size is the dimensionality of the input.
        The condition is a tensor of shape (batch_size, *condition_shape), where
        condition_shape is the shape of the condition tensor.

    """

    def __init__(self, net: nn.Module, condition_shape: torch.Size) -> None:
        r"""Base class for ratio estimators.

        Args:
            net: Neural network.
            condition_shape: Shape of the condition. If not provided, it will assume a
                            1D input.
        """
        super().__init__()
        self.net = net
        self._condition_shape = condition_shape

    def forward(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the logits of the batched (input,condition) pairs.

        Args:
            input: Inputs to evaluate the log probability on of shape
                    (*batch_shape, input_size).
            condition: Conditions of shape (*batch_shape, *condition_shape).

        Raises:
            RuntimeError: If batch_shapes don't match.

        Returns:
            Sample-wise logits.

        Note:
            This function should support PyTorch's automatic broadcasting. This means
            the function should behave as follows for different input and condition
            shapes:
            - (input_size,) + (batch_size,*condition_shape) -> (batch_size,)
            - (batch_size, input_size) + (*condition_shape) -> (batch_size,)
            - (batch_size, input_size) + (batch_size, *condition_shape) -> (batch_size,)
            - (batch_size1, input_size) + (batch_size2, *condition_shape)
                                                  -> RuntimeError i.e. not broadcastable
        """

        raise NotImplementedError

    def loss(self, input: Tensor, condition: Tensor, labels, **kwargs) -> Tensor:
        r"""Return the loss for training the ratio estimator.

        Args:
            input: Inputs to evaluate the loss on of shape (batch_size, input_size).
            condition: Conditions of shape (batch_size, *condition_shape).
            labels: Labels of shape (batch_size,).

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
