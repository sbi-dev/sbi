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

    def __init__(self, net: flows.Flow):

        super().__init__(net)

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the batched log probabilities of the inputs given the conditions.

        Args:
            input: Inputs to evaluate the log probability of. Must have batch dimension.
            condition: Conditions. Must have batch dimension.

        Returns:
            Sample-wise log probabilities.
        """
        return self.net.log_prob(input, context=condition)

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

        num_samples = torch.Size(sample_shape).numel()

        if len(condition.shape) == 1:
            # nflows.sample() expects conditions to be batched.
            condition = condition.unsqueeze(0)
            samples = self.net.sample(num_samples, context=condition).reshape(
                (*sample_shape, -1)
            )
        else:
            # For batched conditions, we need to reshape the conditions and the samples
            batch_dims = condition.shape[:-1]
            condition = condition.reshape(-1, condition.shape[-1])
            samples = self.net.sample(num_samples, context=condition).reshape(
                (*batch_dims, *sample_shape, -1)
            )
        return samples

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Return samples and their density from the density estimator.

        Args:
            sample_shape (torch.Size): Shape of the samples to return.
            condition (Tensor): Conditions.

        Returns:
            Tuple[Tensor, Tensor]: samples and log_probs.
        """

        num_samples = torch.Size(sample_shape).numel()

        if len(condition.shape) == 1:
            # nflows.sample() expects conditions to be batched.
            condition = condition.unsqueeze(0)
            samples, log_probs = self.net.sample_and_log_prob(
                num_samples, context=condition
            )
            samples = samples.reshape((*sample_shape, -1))
            log_probs = log_probs.reshape((*sample_shape,))
        else:
            # For batched conditions, we need to reshape the conditions and the samples
            batch_dims = condition.shape[:-1]
            condition = condition.reshape(-1, condition.shape[-1])
            samples, log_probs = self.net.sample_and_log_prob(
                num_samples, context=condition
            )
            samples = samples.reshape((*batch_dims, *sample_shape, -1))
            log_probs = log_probs.reshape((*batch_dims, *sample_shape))

        return samples, log_probs
