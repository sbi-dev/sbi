from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor, nn


class Estimator(nn.Module, ABC):
    """Base class for SBI estimators.

    This is a base class for all SBI estimators, e.g., density estimators (MDNs, flows),
    vector field estimators (score matching, flow matching), ratio estimators, or custom
    estimator.

    The only requirement for any child class of Estimator is to adhere to our shape
    convention: We assume that all incoming tensors, e.g., the input and the condition
    tensor have a shape composed of three parts:
        - iid_shape (sample_shape in PyTorch, independent and identically distributed)
        - batch_shape (independent but not identically distributed)
        - event_shape (can be dependent, e.g., an image)
    similar to the convention in PyTorch (see, e.g.,
    https://bochang.me/blog/posts/pytorch-distributions/)

    All methods implemented in an Estimator can assume to get this shape and to handle
    it for their needs, e.g., for computing a loss, log_prob, or for sampling. Arranging
    the input to those shapes is handled upstream in the Inference, Posterior, or
    Potential classes.
    """


class DensityEstimator(Estimator, ABC):
    r"""Base class for density estimators.

    The density estimator defines an interfact for neural networks that can perform
    conditional density estimation, e.g., mixture density networks, or conditional
    normalizing flows. All DensityEstimators have to implement `log_prob` and a `sample`
    method, that are conditional on a `condition` tensor. The loss is implemented as the
    negative log probability of the input given the condition.

    Note:
        We assume that the input to the density estimator is a tensor of shape
        (iid_size, batch_size, *event_shape), where input_size is the dimensionality
        (i.e., event shape) of the input. The condition is a tensor of shape
        (iid_size, batch_size, *event_shape), where condition_shape is the shape
        (event shape) of the condition tensor.

    """

    @abstractmethod
    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the log probabilities of the inputs given a condition or multiple
        i.e. batched conditions.

        Args:
            input: Inputs to evaluate the log probability on. Of shape
                 `(iid_dim, batch_dim, *event_shape)`.
            condition: Conditions of shape `(iid_dim, batch_dim, *event_shape)`.

        Raises:
            AssertionError: If `input_batch_dim != condition_batch_dim`.

        Returns:
            Sample-wise log probabilities, shape `(input_iid_dim, input_batch_dim)`.
        """
        ...

    @abstractmethod
    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
             sample_shape: Shape of the samples to return.
             condition: Conditions of shape `(iid_dim, batch_dim, *event_shape)`.
        Returns:
             Samples of shape `(*sample_shape, condition_batch_dim)`.
        """

        raise NotImplementedError

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
             input: Inputs to evaluate the loss on of shape
                 `(iid_dim, batch_dim, *event_shape)`.
             condition: Conditions of shape `(iid_dim, batch_dim, *event_dim)`.
        Returns:
             Negative log_probability of shape `(input_iid_dim, condition_batch_dim)`.
        """
        return -self.log_prob(input, condition, **kwargs)

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        r"""Return samples and their density from the density estimator.

        Args:
             sample_shape: Shape of the samples to return.
             condition: Conditions of shape (iid_dim, batch_dim, *event_shape).

        Returns:
             Samples of shape `(*sample_shape, condition_batch_dim, *input_event_shape)`
             and associated log probs of shape `(*sample_shape, condition_batch_dim)`.

        Note:
            For some density estimators, computing log_probs for samples is
            more efficient than computing them separately. This method should
            then be overwritten to provide a more efficient implementation.
        """

        samples = self.sample(sample_shape, condition, **kwargs)
        log_probs = self.log_prob(samples, condition, **kwargs)
        return samples, log_probs
