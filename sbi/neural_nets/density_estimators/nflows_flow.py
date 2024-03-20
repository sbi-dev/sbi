from typing import Tuple

import torch
from pyknos.nflows.flows import Flow
from torch import Tensor, nn

from sbi.neural_nets.density_estimators.base import DensityEstimator
from sbi.sbi_types import Shape


class NFlowsFlow(DensityEstimator):
    r"""`nflows`- based normalizing flow density estimator.

    Flow type objects already have a .log_prob() and .sample() method, so here we just
    wrap them and add the .loss() method.
    """

    def __init__(self, net: Flow, condition_shape: torch.Size) -> None:
        super().__init__(net, condition_shape)
        # TODO: Remove as soon as DensityEstimator becomes abstract
        self.net: Flow

    @property
    def embedding_net(self) -> nn.Module:
        r"""Return the embedding network."""
        return self.net._embedding_net

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
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
        input_iid_dim = input.shape[0]
        input_batch_dim = input.shape[1]
        condition_batch_dim = condition.shape[1]
        input_event_dims = len(input.shape[2:])
        
        assert condition_batch_dim == input_batch_dim, (
            f"Batch shape of condition {condition_batch_dim} and input "
            f"{input_batch_dim} do not match."
        )

        # Nflows needs to have a single batch dimension for condition and input. 
        input = input.reshape((input_batch_dim * input_iid_dim, -1))
        
        # Repeat the condition to match `input_batch_dim * input_iid_dim`.
        ones_for_event_dims = (1,) * input_event_dims  # Tuple of 1s, e.g. (1, 1, 1)
        condition = condition.repeat(1, input_iid_dim, *ones_for_event_dims)
        # The `.net` expects `batch, iid, event`, not `iid, batch, event`.
        condition = condition.transpose(1, 0)
        # If no iid samples then squeeze the iid dimension.
        condition = torch.squeeze(condition, dim=1)

        log_probs = self.net.log_prob(input, context=condition)
        return log_probs.reshape((input_iid_dim, input_batch_dim))


    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on of shape 
                `(iid_dim, batch_dim, *event_shape)`.
            condition: Conditions of shape `(iid_dim, batch_dim, *event_dim)`.

        Returns:
            Negative log_probability of shape `(input_iid_dim, condition_batch_dim)`.
        """

        return -self.log_prob(input, condition)

    def sample(self, sample_shape: Shape, condition: Tensor) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape `(iid_dim, batch_dim, *event_shape)`.

        Returns:
            Samples of shape `(*sample_shape, condition_batch_dim)`.
        """
        condition_batch_dim = condition.shape[1]
        num_samples = torch.Size(sample_shape).numel()

        # The `.net` expects `batch, iid, event`, not `iid, batch, event`.
        condition = condition.transpose(1, 0)
        condition = torch.squeeze(condition, dim=1)
        samples = self.net.sample(num_samples, context=condition)

        return samples.reshape((
            *sample_shape,
            condition_batch_dim,
            -1,
        ))

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
        """
        condition_batch_dim = condition.shape[1]
        num_samples = torch.Size(sample_shape).numel()

        # The `.net` expects `batch, iid, event`, not `iid, batch, event`.
        condition = condition.transpose(1, 0)

        samples, log_probs = self.net.sample_and_log_prob(
            num_samples, context=condition
        )
        samples = samples.reshape((*sample_shape, condition_batch_dim, -1))
        log_probs = log_probs.reshape((*sample_shape, -1))
        return samples, log_probs
