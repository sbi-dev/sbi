from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class RatioEstimator(nn.Module, ABC):
    r"""Base class for ratio estimators.

    The ratio estimator class is a wrapper around neural networks that enables
    evaluation of `unnormalized_log_ratio` and the `loss` for `theta`, `x`
    pairs. It also provides a method for combining (embedded) `theta` and `x`
    into a single tensor.

    Note:
        We assume that the input to the ratio estimator is a tensor of shape
        (sample, batch, event) where sample are iid draws, batch are
        independent (not necessarily identically distributed) draws, and
        event can be totally dependent.
        See: https://bochang.me/blog/posts/pytorch-distributions/
    """

    def __init__(
        self,
        net: nn.Module,
        theta_shape: torch.Size,
        x_shape: torch.Size,
        embedding_net_theta: nn.Module,
        embedding_net_x: nn.Module,
    ) -> None:
        r"""Base class for ratio estimators.

        Args:
            net: neural network taking in combined (embedded) `theta` and `x`
            theta_shape
            x_shape
            embedding_net_theta
            embedding_net_x
        """
        super().__init__()
        self.net = net
        self._theta_shape = theta_shape
        self._x_shape = x_shape
        self.embedding_net_theta = embedding_net_theta
        self.embedding_net_x = embedding_net_x

    @abstractmethod
    def combine_embedded_theta_and_x(
        embedded_theta: Tensor, embedded_x: Tensor
    ) -> Tensor:
        return None

    @abstractmethod
    def unnormalized_log_ratio(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        r"""Return the unnormalized log ratios of the thetas given an x, or multiple
        (batched) xs.

        Args:
            theta: parameters of shape (sample, batch, theta_shape).
            x: data of shape (sample, batch, x_shape).

        Returns:
            Sample-wise unnormalized log ratios.
        """

        raise NotImplementedError

    @abstractmethod
    def loss(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the ratio estimator.

        Args:
            theta: parameters of shape (sample, batch, theta_shape).
            x: data of shape (sample, batch, x_shape).

        Returns:
            Loss of shape (batch_size,)
        """

        raise NotImplementedError
