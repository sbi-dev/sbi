from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class DensityEstimator(nn.Module):
    r"""Base class for density estimators.

    The density estimator class is a wrapper around neural networks that
    allows to evaluate the `log_prob`, `sample`, and provide the `loss` of $\theta,x$
    pairs. Here $\theta$ would be the `input` and $x$ would be the `condition`.

    Note:
        We assume that the input to the density estimator is a tensor of shape
        (batch_size, input_size), where input_size is the dimensionality of the input.
        The condition is a tensor of shape (batch_size, *condition_shape), where
        condition_shape is the shape of the condition tensor.

    """

    def __init__(self, net: nn.Module, condition_shape: torch.Size) -> None:
        r"""Base class for density estimators.

        Args:
            net: Neural network.
            condition_shape: Shape of the condition. If not provided, it will assume a
                            1D input.
        """
        super().__init__()
        self.net = net
        self._condition_shape = condition_shape

    @property
    def embedding_net(self) -> Optional[nn.Module]:
        r"""Return the embedding network if it exists."""
        return None

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the log probabilities of the inputs given a condition or multiple
        i.e. batched conditions.

        Args:
            input: Inputs to evaluate the log probability on of shape
                    (*batch_shape1, input_size).
            condition: Conditions of shape (*batch_shape2, *condition_shape).

        Raises:
            RuntimeError: If batch_shape1 and batch_shape2 are not broadcastable.

        Returns:
            Sample-wise log probabilities.

        Note:
            This function should support PyTorch's automatic broadcasting. This means
            the function should behave as follows for different input and condition
            shapes:
            - (input_size,) + (batch_size,*condition_shape) -> (batch_size,)
            - (batch_size, input_size) + (*condition_shape) -> (batch_size,)
            - (batch_size, input_size) + (batch_size, *condition_shape) -> (batch_size,)
            - (batch_size1, input_size) + (batch_size2, *condition_shape)
                                                  -> RuntimeError i.e. not broadcastable
            - (batch_size1,1, input_size) + (batch_size2, *condition_shape)
                                                  -> (batch_size1,batch_size2)
            - (batch_size1, input_size) + (batch_size2,1, *condition_shape)
                                                  -> (batch_size2,batch_size1)
        """

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

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape (*batch_shape, *condition_shape).

        Returns:
            Samples of shape (*batch_shape, *sample_shape, input_size).

        Note:
            This function should support batched conditions and should admit the
            following behavior for different condition shapes:
            - (*condition_shape) -> (*sample_shape, input_size)
            - (*batch_shape, *condition_shape)
                                        -> (*batch_shape, *sample_shape, input_size)
        """

        raise NotImplementedError

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        r"""Return samples and their density from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape (*batch_shape, *condition_shape).

        Returns:
            Samples and associated log probabilities.


        Note:
            For some density estimators, computing log_probs for samples is
            more efficient than computing them separately. This method should
            then be overwritten to provide a more efficient implementation.
        """

        samples = self.sample(sample_shape, condition, **kwargs)
        log_probs = self.log_prob(samples, condition, **kwargs)
        return samples, log_probs

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
