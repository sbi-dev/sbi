from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class RatioEstimator(nn.Module):
    r"""Base class for ratio estimators.

    The ratio estimator class is a wrapper around neural networks that enables
    evaluation of `unnormalized_log_ratio` and the `loss` for $\theta,x$ pairs.
    It is also possible to access embedding networks for theta and x as well as
    concatenate the output of those networks into a single tensor.

    Note:
        We assume that the input to the ratio estimator is a tensor of shape
        (batch_size, embed_theta_size + embed_x_size) where embed_theta_size
        and embed_x_size are the dim of the theta and x embedding respectively.

        Both embed_*_size values are infered through use of embedding networks.

        Those embedding networks map from a tensor of shape
        (batch_size, *theta_shape) or (batch_size, *x_shape) to
        (batch_size, embed_theta_size) or (batch_size, embed_x_size) respectively.
    """

    def __init__(
        self, net: nn.Module, theta_shape: torch.Size, x_shape: torch.Size
    ) -> None:
        r"""Base class for ratio estimators.

        Args:
            net: Neural network.
            theta_shape
            x_shape
        """
        super().__init__()
        self.net = net
        self._theta_shape = theta_shape
        self._x_shape = x_shape

    @property
    def embedding_net_theta(self) -> Optional[nn.Module]:
        r"""Return the embedding network for theta, if it exists."""
        return None

    @property
    def embedding_net_x(self) -> Optional[nn.Module]:
        r"""Return the embedding network for x, if it exists."""
        return None

    def unnormalized_log_ratio(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        r"""Return the unnormalized log ratios of the thetas given an x, or multiple
        (batched) xs.

        Args:
            theta: parameters of shape (*batch_shape1, theta_shape).
            x: data of shape (*batch_shape2, *x_shape).

        Raises:
            RuntimeError: If batch_shape1 and batch_shape2 are not broadcastable.

        Returns:
            Sample-wise unnormalized log ratios.

        Note:
            This function should support PyTorch's automatic broadcasting for both
            theta and x. This means the function should behave as follows for
            different theta and x shapes:
            - (*theta_shape,) + (batch_size, *x_shape) -> (batch_size,)
            - (batch_size, *theta_shape) + (*x_shape) -> (batch_size,)
            - (batch_size, *theta_shape) + (batch_size, *x_shape) -> (batch_size,)
            - (batch_size1, *theta_shape) + (batch_size2, *x_shape)
                                                  -> RuntimeError i.e. not broadcastable
            - (batch_size1,1, *theta_shape) + (batch_size2, *x_shape)
                                                  -> (batch_size1, batch_size2)
            - (batch_size1, *theta_shape) + (batch_size2, 1, *x_shape)
                                                  -> (batch_size2, batch_size1)
        """

        raise NotImplementedError

    # TODO below

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the ratio estimator.

        Args:
            input: Inputs to evaluate the loss on of shape (batch_size, input_size).
            condition: Conditions of shape (batch_size, *condition_shape).

        Returns:
            Loss of shape (batch_size,)
        """

        raise NotImplementedError

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        r"""Return samples from the ratio estimator.

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
        r"""Return samples and their ratio from the ratio estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape (*batch_shape, *condition_shape).

        Returns:
            Samples and associated log probabilities.


        Note:
            For some ratio estimators, computing log_probs for samples is
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
