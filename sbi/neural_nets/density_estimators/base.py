from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class DensityEstimator(nn.Module):
    r"""Base class for density estimators.

    The density estimator class is a wrapper around neural networks that
    allows to evaluate the `log_prob`, `sample`, and provide the `loss` of $theta,x$
    pairs.
    """

    def __init__(self, net: nn.Module, x_shape: torch.Size) -> None:
        r"""Base class for density estimators.

        Args:
            net: Neural network.
            x_shape: Shape of the input. If not provided, it will assume a 1D input.
        """
        super().__init__()
        self.net = net
        self._x_shape = x_shape

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the batched log probabilities of the inputs given the conditions.

        Args:
            input: Inputs to evaluate the log probability of. Must have batch dimension.
            x: Conditions. Must have batch dimension.

        Returns:
            Sample-wise log probabilities.
        """

        raise NotImplementedError

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on.
            condition: Conditions.

        Returns:
            Loss.
        """

        raise NotImplementedError

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        r"""Return samples from the density estimator.

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

    def _check_for_invalid_condition_shape(self, condition: Tensor):
        r"""This method checks whether the condition has the correct shape.

        Args:
            condition (Tensor): Given condition.

        Raises:
            ValueError: If the condition has a dimensionality that does not match the expected input dimensionality.
            ValueError: If the shape of the condition does not match the expected input dimensionality.
        """
        if len(condition.shape) < len(self._x_shape):
            raise ValueError(
                f"Dimensionality of condition is to small and does not match the expected input dimensionality {len(self._x_shape)}, as provided by x_shape."
            )
        else:
            x_shape = condition.shape[-len(self._x_shape) :]
            if tuple(x_shape) != tuple(self._x_shape):
                raise ValueError(
                    f"Shape of condition {tuple(x_shape)} does not match the expected input dimensionality {tuple(self._x_shape)}, as provided by x_shape. Please reshape it accordingly."
                )
