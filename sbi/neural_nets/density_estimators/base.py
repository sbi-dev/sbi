from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class DensityEstimator(nn.Module):
    r"""Base class for density estimators.

    The density estimator class is a wrapper around neural networks that
    allows to evaluate the `log_prob`, `sample`, and provide the `loss` of $theta,x$
    pairs.

    Note:
        We assume that the input to the density estimator is a tensor of shape
        (batch_size, d), where d is the dimensionality of the input. The condition
        is a tensor of shape (batch_size, *condition_shape), where condition_shape
        is the shape of the condition tensor.

    """

    def __init__(self, net: nn.Module, condition_shape: torch.Size) -> None:
        r"""Base class for density estimators.

        Args:
            net: Neural network.
            condition_shape: Shape of the input. If not provided, it will assume a 1D
                             input.
        """
        super().__init__()
        self.net = net
        self._condition_shape = condition_shape

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the log probabilities of the inputs given a condition or multiple
        i.e. batched conditions.

        Note:
            This function should support PyTorch's automatic broadcasting. This means
            the function should behave as follows for different input and condition
            shapes:
            - (d,) + (b,*condition_shape) -> (b,)
            - (b, d) + (*condition_shape) -> (b,)
            - (b, d) + (b, *condition_shape) -> (b,)
            - (b1, d) + (b2, *condition_shape) -> RuntimeError i.e. not broadcastable
            - (b1,1, d) + (b2, *condition_shape) -> (b1,b2)
            - (b1, d) + (b2,1, *condition_shape) -> (b2,b1)

        Args:
            input: Inputs to evaluate the log probability on.
            condition: Conditions.

        Returns:
            Sample-wise log probabilities.
        """

        raise NotImplementedError

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on of shape (batch_size, d).
            condition: Conditions of shape (batch_size, *condition_shape).

        Returns:
            Loss.
        """

        raise NotImplementedError

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        r"""Return samples from the density estimator.

        Note:
            This function should support batched conditions and should admit the
            following behavior for different condition shapes:
            - (*condition_shape) -> (*sample_shape,d)
            - (*batch_shapes, *condition_shape) -> (*batch_shapes, *sample_shape, d)

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions.

        Returns:
            Samples.
        """

        raise NotImplementedError

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        r"""Return samples and their density from the density estimator.

        Note:
            For some density estimators, computing log_probs for samples is
            more efficient than computing them separately. This method should
            then be overwritten to provide a more efficient implementation.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions.
        Returns:
            Samples and associated log probabilities.
        """

        x = self.sample(sample_shape, condition, **kwargs)
        log_prob = self.log_prob(x, condition, **kwargs)
        return x, log_prob

    def _check_condition_shape(self, condition: Tensor):
        r"""This method checks whether the condition has the correct shape.

        Args:
            condition: Conditions.

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
                by x_shape."
            )
        else:
            condition_shape = condition.shape[-len(self._condition_shape) :]
            if tuple(condition_shape) != tuple(self._condition_shape):
                raise ValueError(
                    f"Shape of condition {tuple(condition_shape)} does not match the \
                    expected input dimensionality {tuple(self._condition_shape)}, as \
                    provided by x_shape. Please reshape it accordingly."
                )
