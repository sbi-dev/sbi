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
        embedding_net_theta: nn.Module,
        embedding_net_x: nn.Module,
    ) -> None:
        r"""Base class for ratio estimators.

        Args:
            net: neural network taking in combined (embedded) `theta` and `x`
            embedding_net_theta
            embedding_net_x
        """
        super().__init__()
        self.net = net
        self.embedding_net_theta = embedding_net_theta
        self.embedding_net_x = embedding_net_x

    @abstractmethod
    def combine_embedded_theta_and_x(
        self, embedded_theta: Tensor, embedded_x: Tensor
    ) -> Tensor:
        r"""Combine embedded theta and embedded x sensibly for the data type.

        Args:
            embedded_theta: theta after embedding
            embedded_x: x after embedding

        Returns:
            Single object containing both embedded_theta and embedded_x
        """
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
            Just like log_prob, the last dimension should be squeezed.
        """

        raise NotImplementedError

    def forward(self, *args, **kwargs) -> Tensor:
        r"""Wraps `unnormalized_log_ratio`"""
        return self.unnormalized_log_ratio(*args, **kwargs)


class TensorRatioEstimator(RatioEstimator):
    def __init__(
        self,
        net: nn.Module,
        embedding_net_theta: nn.Module = nn.Identity(),
        embedding_net_x: nn.Module = nn.Identity(),
    ) -> None:
        r"""Base class for ratio estimators.

        Args:
            net: neural network taking in combined (embedded) `theta` and `x`
            embedding_net_theta
            embedding_net_x
        """
        super().__init__(
            net=net,
            embedding_net_theta=embedding_net_theta,
            embedding_net_x=embedding_net_x,
        )

    @staticmethod
    def combine_embedded_theta_and_x(
        embedded_theta: Tensor, embedded_x: Tensor, dim: int = -1
    ) -> Tensor:
        """Concatenate embedded theta and embedded x"""
        return torch.cat([embedded_theta, embedded_x], dim=dim)

    def unnormalized_log_ratio(self, theta: Tensor, x: Tensor) -> Tensor:
        z = self.combine_embedded_theta_and_x(
            self.embedding_net_theta(theta), self.embedding_net_x(x)
        )
        return self.net(z).squeeze(-1)
