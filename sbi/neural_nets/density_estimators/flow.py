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
        r"""Return the batched log probabilities of the inputs given the conditions.

        Args:
            input: Inputs to evaluate the log probability of. Must have batch dimension.
            condition: Conditions. Must have batch dimension.

        Returns:
            Sample-wise log probabilities.
        """
        self._check_for_invalid_condition_shape(condition)
        x_dim = len(self._x_shape)

        batch_shape_in = input.shape[:-1]
        batch_shape_cond = condition.shape[:-x_dim]
        batch_shape = torch.broadcast_shapes(batch_shape_in, batch_shape_cond)
        # Expand the input and condition to the same batch shape
        input = input.expand(batch_shape + (input.shape[-1],))
        condition = condition.expand(batch_shape + self._x_shape)
        # Flatten required by nflows, but now both have the same batch shape
        input = input.reshape(-1, input.shape[-1])
        condition = condition.reshape(-1, *self._x_shape)

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

        Args:
            sample_shape: Batch dimensions of the samples to return
            condition: Condition.

        Returns:
            Samples.
        """
        self._check_for_invalid_condition_shape(condition)

        num_samples = torch.Size(sample_shape).numel()
        x_dim = len(self._x_shape)

        if len(condition.shape) == x_dim:
            # nflows.sample() expects conditions to be batched.
            condition = condition.unsqueeze(0)
            samples = self.net.sample(num_samples, context=condition).reshape(
                (*sample_shape, -1)
            )
        else:
            # For batched conditions, we need to reshape the conditions and the samples
            batch_dims = condition.shape[:-x_dim]
            condition = condition.reshape(-1, *self._x_shape)
            samples = self.net.sample(num_samples, context=condition).reshape(
                (*batch_dims, *sample_shape, -1)
            )

        return samples

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        r"""Return samples and their density from the density estimator.

        Args:
            sample_shape (torch.Size): Shape of the samples to return.
            condition (Tensor): Conditions.

        Returns:
            Tuple[Tensor, Tensor]: samples and log_probs.
        """
        self._check_for_invalid_condition_shape(condition)

        num_samples = torch.Size(sample_shape).numel()
        x_dim = len(self._x_shape)

        if len(condition.shape) == x_dim:
            # nflows.sample() expects conditions to be batched.
            condition = condition.unsqueeze(0)
            samples, log_probs = self.net.sample_and_log_prob(
                num_samples, context=condition
            )
            samples = samples.reshape((*sample_shape, -1))
            log_probs = log_probs.reshape((*sample_shape,))
        else:
            # For batched conditions, we need to reshape the conditions and the samples
            batch_dims = condition.shape[:-x_dim]
            condition = condition.reshape(-1, *self._x_shape)
            samples, log_probs = self.net.sample_and_log_prob(
                num_samples, context=condition
            )
            samples = samples.reshape((*batch_dims, *sample_shape, -1))
            log_probs = log_probs.reshape((*batch_dims, *sample_shape))

        return samples, log_probs
