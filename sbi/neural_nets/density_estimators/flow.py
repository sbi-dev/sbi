from typing import Tuple

import torch
from pyknos.nflows import flows
from torch import Tensor

from sbi.neural_nets.density_estimators.base import DensityEstimator
from sbi.types import Shape


class NFlowsFlow(DensityEstimator):
    r"""`nflows`- based normalizing flow density estimator.

    Flow type objects already have a .log_prob() and .sample() method, so here we just
    wrap them and add the .loss() method.
    """

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
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
        self._check_condition_shape(condition)
        condition_dims = len(self._condition_shape)

        # PyTorch's automatic broadcasting
        batch_shape_in = input.shape[:-1]
        batch_shape_cond = condition.shape[:-condition_dims]
        batch_shape = torch.broadcast_shapes(batch_shape_in, batch_shape_cond)
        # Expand the input and condition to the same batch shape
        input = input.expand(batch_shape + (input.shape[-1],))
        condition = condition.expand(batch_shape + self._condition_shape)
        # Flatten required by nflows, but now both have the same batch shape
        input = input.reshape(-1, input.shape[-1])
        condition = condition.reshape(-1, *self._condition_shape)

        log_probs = self.net.log_prob(input, context=condition)
        log_probs = log_probs.reshape(batch_shape)
        return log_probs

    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on. Must be batched.
            condition: Conditions. Must be batched.

        Returns:
            Negative log-probability.
        """

        return -self.log_prob(input, condition)

    def sample(self, sample_shape: Shape, condition: Tensor) -> Tensor:
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
        self._check_condition_shape(condition)

        num_samples = torch.Size(sample_shape).numel()
        condition_dims = len(self._condition_shape)

        if len(condition.shape) == condition_dims:
            # nflows.sample() expects conditions to be batched.
            condition = condition.unsqueeze(0)
            samples = self.net.sample(num_samples, context=condition).reshape(
                (*sample_shape, -1)
            )
        else:
            # For batched conditions, we need to reshape the conditions and the samples
            batch_dims = condition.shape[:-condition_dims]
            condition = condition.reshape(-1, *self._condition_shape)
            samples = self.net.sample(num_samples, context=condition).reshape(
                (*batch_dims, *sample_shape, -1)
            )

        return samples

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        r"""Return samples and their density from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions.

        Returns:
            Tuple[Tensor, Tensor]: samples and log_probs.
        """
        self._check_condition_shape(condition)

        num_samples = torch.Size(sample_shape).numel()
        condition_dims = len(self._condition_shape)

        if len(condition.shape) == condition_dims:
            # nflows.sample() expects conditions to be batched.
            condition = condition.unsqueeze(0)
            samples, log_probs = self.net.sample_and_log_prob(
                num_samples, context=condition
            )
            samples = samples.reshape((*sample_shape, -1))
            log_probs = log_probs.reshape((*sample_shape,))
        else:
            # For batched conditions, we need to reshape the conditions and the samples
            batch_shape = condition.shape[:-condition_dims]
            condition = condition.reshape(-1, *self._condition_shape)
            samples, log_probs = self.net.sample_and_log_prob(
                num_samples, context=condition
            )
            samples = samples.reshape((*batch_shape, *sample_shape, -1))
            log_probs = log_probs.reshape((*batch_shape, *sample_shape))

        return samples, log_probs
