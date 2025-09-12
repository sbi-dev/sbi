# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import torch
from torch import Tensor, nn

from sbi.neural_nets.estimators.base import ConditionalEstimator


class RatioEstimator(ConditionalEstimator):
    r"""Base class for ratio estimators.

    The ratio estimator class is a wrapper around neural networks that enables
    evaluation of `unnormalized_log_ratio` for `theta`, `x` pairs. It also
    provides a method for combining `theta` and `x` into a single tensor.

    Note:
        We assume that the input to the ratio estimator is a tensor of shape
        `(sample_dim, batch_dim, *event_shape)` where `sample_dim` are iid draws,
        `batch_dim` are independent (not necessarily identically distributed)
        draws, and `event_shape` can be totally dependent.

        It is also possible that `sample_dim` and `batch_dim` are not present.
        Note: `dim` implies 1 dim, `shape` could be more than 1 dim.

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
        r"""Class for ratio estimators that concatenate theta and x embeddings.

        Args:
            net: neural network taking in combined, embedded `theta` and `x`
            theta_shape: event_shape for theta
            x_shape: event_shape for x
            embedding_net_theta
            embedding_net_x
        """

        super().__init__(input_shape=theta_shape, condition_shape=x_shape)
        self.net = net
        self.theta_shape = self.input_shape
        self.x_shape = self.condition_shape
        self.embedding_net_theta = embedding_net_theta
        self.embedding_net_x = embedding_net_x

    def _check_x_shape_suffix(self, x: Tensor) -> None:
        r"""This method checks whether x has the correct shape.

        Args:
            x: Tensor of shape (*batch_shape, *x_shape).

        Raises:
            ValueError: If the x has a dimensionality that does not match
                        the expected input dimensionality.
            ValueError: If the shape of the x does not match the expected
                        input dimensionality.
        """
        return self._check_condition_shape(x)

    def _check_theta_shape_suffix(self, theta: Tensor) -> None:
        r"""This method checks whether theta has the correct shape.

        Args:
            theta: Tensor of shape (*batch_shape, *theta_shape).

        Raises:
            ValueError: If the theta has a dimensionality that does not match
                        the expected input dimensionality.
            ValueError: If the shape of the theta does not match the expected
                        input dimensionality.
        """
        return self._check_input_shape(theta)

    def _get_shape_prefix(
        self, theta: Tensor, x: Tensor
    ) -> torch.Size | tuple[int, ...]:
        r"""This method checks whether theta and x agree on the prefix of their shape.

        Args:
            theta: Tensor of shape (*batch_shape, *theta_shape).
            x: Tensor of shape (*batch_shape, *x_shape).

        Raises:
            ValueError: If the `batch_shape`s do not agree.
        """
        theta_prefix = theta.shape[: -len(self.theta_shape)]
        x_prefix = x.shape[: -len(self.x_shape)]
        if theta_prefix != x_prefix:
            raise ValueError(
                f"{tuple(theta_prefix)=} != {tuple(x_prefix)=}. \
                             Make them agree, since we do not broadcast for you."
            )
        else:
            return theta_prefix

    def combine_theta_and_x(self, theta: Tensor, x: Tensor) -> Tensor:
        """After embedding them, concatenate embedded_theta and embedded_x

        Args:
            theta: parameters of shape (sample_dim, batch_dim, *theta_shape).
            x: data of shape (batch_dim, *x_shape).

        Returns:
            combined: shape (sample_dim, batch_dim, combined_event_dim)
        """
        dim = -1
        self._check_theta_shape_suffix(theta)
        self._check_x_shape_suffix(x)
        prefix_shape = self._get_shape_prefix(theta, x)
        embedded_theta = self.embedding_net_theta(theta.reshape(-1, *self.theta_shape))
        embedded_x = self.embedding_net_x(x.reshape(-1, *self.x_shape))
        return torch.cat([embedded_theta, embedded_x], dim=dim).reshape(
            *prefix_shape, dim
        )

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

    def forward(self, *args, **kwargs) -> Tensor:
        r"""Wraps `unnormalized_log_ratio`"""
        return self.unnormalized_log_ratio(*args, **kwargs)

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError()
