# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class ConditionalEstimator(nn.Module, ABC):
    r"""Base class for conditional estimators that estimate properties of
    distributions conditional on an input.

    For example, this can be:
    - Conditional density estimator of the posterior $p(\theta|x)$.
    - Conditional density estimator of the likelihood $p(x|\theta)$.
    - Conditional vector field estimator e.g. $\nabla_\theta \log p(\theta|x)$.

    Subclasses of ConditionalEstimator should implement the ``loss(input, condition)``
    method to be compatible with sbi's training procedures.
    """

    def __init__(self, input_shape: Tuple, condition_shape: Tuple) -> None:
        r"""Construct a conditional estimator given shapes.

        Args:
            input_shape: Event shape of the input at which the density is being
                evaluated (and which is also the event_shape of samples).
            condition_shape: Shape of the condition.
        """
        super().__init__()
        self._input_shape = torch.Size(input_shape)
        self._condition_shape = torch.Size(condition_shape)

    @property
    def input_shape(self) -> torch.Size:
        r"""Return the input shape."""
        return self._input_shape

    @property
    def condition_shape(self) -> torch.Size:
        r"""Return the condition shape."""
        return self._condition_shape

    @abstractmethod
    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the estimator.

        Args:
            input: Inputs to evaluate the loss on of shape
                `(batch_dim, *input_event_shape)`.
            condition: Conditions of shape `(batch_dim, *event_shape_condition)`.

        Returns:
            Loss of shape (batch_dim,)
        """
        pass

    def _check_condition_shape(self, condition: Tensor):
        r"""This method checks whether the condition has the correct shape.

        Args:
            condition: Conditions of shape `(batch_dim, *event_shape_condition)`.

        Raises:
            ValueError: If the condition has a dimensionality that does not match
                        the expected input dimensionality.
            ValueError: If the shape of the condition does not match the expected
                        input dimensionality.
        """
        exp_condition_shape = self.condition_shape
        if len(condition.shape) < len(exp_condition_shape):
            raise ValueError(
                "Dimensionality of condition is too small and does not match the "
                f"expected dimensionality {len(exp_condition_shape)}. It should "
                f"be compatible with condition_shape {exp_condition_shape}."
            )
        else:
            condition_shape = condition.shape[-len(self.condition_shape) :]
            if condition_shape != exp_condition_shape:
                raise ValueError(
                    f"Shape of condition {condition_shape} does not match the "
                    f"expected input dimensionality {exp_condition_shape}, as "
                    "provided by condition_shape. Please reshape it accordingly."
                )

    def _check_input_shape(self, input: Tensor):
        r"""This method checks whether the input has the correct shape.

        Args:
            input: Inputs to evaluate the log probability on of shape
                    `(sample_dim_input, batch_dim_input, *event_shape_input)`.

        Raises:
            ValueError: If the input has a dimensionality that does not match
                        the expected input dimensionality.
            ValueError: If the shape of the input does not match the expected
                        input dimensionality.
        """
        input_shape = input.shape
        exp_input_shape = self.input_shape
        if len(input_shape) < len(exp_input_shape):
            raise ValueError(
                "Dimensionality of input is too small and does not match the "
                f"expected dimensionality {len(exp_input_shape)}. It should "
                f"be compatible with the provided input_shape {exp_input_shape}."
            )
        else:
            input_shape = input.shape[-len(self.input_shape) :]
            if input_shape != exp_input_shape:
                raise ValueError(
                    f"Shape of input {input_shape} does not match the "
                    f"expected input dimensionality {exp_input_shape}, as "
                    "provided by input_shape. Please reshape it accordingly."
                )


class ConditionalDensityEstimator(ConditionalEstimator):
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

    def __init__(
        self, net: nn.Module, input_shape: torch.Size, condition_shape: torch.Size
    ) -> None:
        r"""Base class for density estimators.

        Args:
            net: Neural network or any parameterized model that is used to estimate the
                probability density of the input given a condition.
            input_shape: Event shape of the input at which the density is being
                evaluated (and which is also the event_shape of samples).
            condition_shape: Shape of the condition.
        """
        super().__init__(input_shape, condition_shape)
        self.net = net

    @property
    def embedding_net(self) -> Optional[nn.Module]:
        r"""Return the embedding network if it exists."""
        return None

    @abstractmethod
    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the log probabilities of the inputs given a condition or multiple
        i.e. batched conditions.

        Args:
            input: Inputs to evaluate the log probability on of shape
                    `(sample_dim_input, batch_dim_input, *event_shape_input)`.
            condition: Conditions of shape
                `(batch_dim_condition, *event_shape_condition)`.

        Raises:
            RuntimeError: If batch_dim_input and batch_dim_condition do not match.

        Returns:
            Sample-wise log probabilities.
        """

        pass

    @abstractmethod
    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on of shape
                `(batch_dim, *input_event_shape)`.
            condition: Conditions of shape `(batch_dim, *event_shape_condition)`.

        Returns:
            Loss of shape (batch_dim,)
        """

        pass

    @abstractmethod
    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape `(batch_dim, *event_shape_condition)`.

        Returns:
            Samples of shape (*sample_shape, batch_dim, *event_shape_input).
        """

        pass

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        r"""Return samples and their density from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape `(batch_dim, *event_shape_condition)`.

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


class ConditionalVectorFieldEstimator(ConditionalEstimator):
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
