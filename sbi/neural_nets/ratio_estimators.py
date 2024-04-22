from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class RatioEstimator(nn.Module, ABC):
    r"""Base class for ratio estimators.

    The ratio estimator class is a wrapper around neural networks that enables
    evaluation of `unnormalized_log_ratio` for `theta`, `x` pairs. It also
    provides a method for combining `theta` and `x` into a single tensor.
    """

    def __init__(self) -> None:
        r"""Base class for ratio estimators."""
        super().__init__()

    @abstractmethod
    def combine_theta_and_x(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""Combine theta and x sensibly for the data type.

        Args:
            theta
            x

        Returns:
            Single object containing both theta and x
        """
        pass

    @abstractmethod
    def unnormalized_log_ratio(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        r"""Return the unnormalized log ratios of the thetas given an x, or multiple
        (batched) xs.

        Args:
            theta
            x

        Returns:
            Sample-wise unnormalized log ratios. Just like log_prob, the last dimension
            should be squeezed.
        """
        pass

    def forward(self, *args, **kwargs) -> Tensor:
        r"""Wraps `unnormalized_log_ratio`"""
        return self.unnormalized_log_ratio(*args, **kwargs)


# TODO: following class does not yet actually use the shapes it claims to use!
class TensorRatioEstimator(RatioEstimator):
    """Base class for ratio estimators that take tensors as input.

    Note:
        We assume that the input to the ratio estimator is a tensor of shape
        (sample_dim, batch_dim, *event_shape) where sample_dim are iid draws,
        batch_dim are independent (not necessarily identically distributed)
        draws, and event_shape can be totally dependent.
        See: https://bochang.me/blog/posts/pytorch-distributions/
    """

    def __init__(
        self,
        net: nn.Module,
        theta_shape: torch.Size | tuple[int, ...],
        x_shape: torch.Size | tuple[int, ...],
        embedding_net_theta: nn.Module = nn.Identity(),
        embedding_net_x: nn.Module = nn.Identity(),
    ) -> None:
        r"""Wrapper class for ratio estimators concatenating theta and x embeddings.

        Args:
            net: neural network taking in combined, embedded `theta` and `x`
            theta_shape
            x_shape
            embedding_net_theta
            embedding_net_x
        """
        super().__init__()
        self.net = net
        self.theta_shape = theta_shape
        self.x_shape = x_shape
        self.embedding_net_theta = embedding_net_theta
        self.embedding_net_x = embedding_net_x

    @staticmethod
    def _check_shape(y: Tensor, shape: torch.Size | tuple[int, ...]) -> None:
        r"""This method checks whether y has the correct shape.

        Args:
            y: Tensor of shape (*batch_shape, *y_shape).

        Returns:
            False:
                1. If the y has a dimensionality that does not match
                the expected input dimensionality.
                2. If the shape of the y does not match the expected
                input dimensionality.
            True: otherwise
        """
        if len(y.shape) < len(shape):
            raise ValueError(
                f"Dimensionality of tensor is to small and does not match the\
                expected input dimensionality {len(shape)}, as provided\
                by tensor_shape."
            )
        else:
            tensor_shape = y.shape[-len(shape) :]
            if tuple(tensor_shape) != tuple(shape):
                raise ValueError(
                    f"Shape of tensor {tuple(tensor_shape)} does not match the \
                    expected input dimensionality {tuple(shape)}, as \
                    provided by tensor_shape. Please reshape it accordingly."
                )

    def _check_x_shape(self, x: Tensor) -> None:
        r"""This method checks whether x has the correct shape.

        Args:
            x: Tensor of shape (*batch_shape, *x_shape).

        Raises:
            ValueError: If the x has a dimensionality that does not match
                        the expected input dimensionality.
            ValueError: If the shape of the x does not match the expected
                        input dimensionality.
        """
        return self._check_shape(x, self.x_shape)

    def _check_theta_shape(self, theta: Tensor) -> None:
        r"""This method checks whether theta has the correct shape.

        Args:
            theta: Tensor of shape (*batch_shape, *theta_shape).

        Raises:
            ValueError: If the theta has a dimensionality that does not match
                        the expected input dimensionality.
            ValueError: If the shape of the theta does not match the expected
                        input dimensionality.
        """
        return self._check_shape(theta, self.theta_shape)

    def combine_theta_and_x(self, theta: Tensor, x: Tensor, dim: int = -1) -> Tensor:
        """After embedding them, concatenate embedded_theta and embedded_x

        Args:
            theta: parameters of shape (sample_dim, batch_dim, *theta_shape).
            x: data of shape (batch_dim, *x_shape).

        Returns:
            combined: shape (sample_dim, batch_dim, combined_event_dim)
        """
        self._check_theta_shape(theta)
        self._check_x_shape(x)
        embedded_theta = self.embedding_net_theta(theta)
        embedded_x = self.embedding_net_x(x)
        return torch.cat([embedded_theta, embedded_x], dim=dim)

    def unnormalized_log_ratio(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""Return the unnormalized log ratios of the thetas given an x, or multiple
        (batched) xs.

        Args:
            theta: parameters of shape (sample_dim, batch_dim, *theta_shape).
            x: data of shape (batch_dim, *x_shape).

        Returns:
            Sample-wise unnormalized log ratios with shape (sample_dim, batch_dim).
            Just like log_prob, the last dimension should be squeezed.
        """

        z = self.combine_theta_and_x(theta, x)
        return self.net(z).squeeze(-1)
