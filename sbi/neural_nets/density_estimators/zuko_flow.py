from typing import Tuple

import torch
from torch import Tensor, nn
from zuko.flows import Flow

from sbi.neural_nets.density_estimators.base import DensityEstimator
from sbi.sbi_types import Shape


class ZukoFlow(DensityEstimator):
    r"""`zuko`- based normalizing flow density estimator.

    Flow type objects already have a .log_prob() and .sample() method, so here we just
    wrap them and add the .loss() method.
    """

    def __init__(
        self, net: Flow, embedding_net: nn.Module, condition_shape: torch.Size
    ):
        r"""Initialize the density estimator.

        Args:
            flow: Flow object.
            condition_shape: Shape of the condition.
        """

        # assert len(condition_shape) == 1, "Zuko Flows require 1D conditions."
        super().__init__(net=net, condition_shape=condition_shape)
        self._embedding_net = embedding_net

    @property
    def embedding_net(self) -> nn.Module:
        r"""Return the embedding network."""
        return self._embedding_net

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
        input_batch_dim = input.shape[1]
        condition_batch_dim = condition.shape[1]

        assert condition_batch_dim == input_batch_dim, (
            f"Batch shape of condition {condition_batch_dim} and input "
            f"{input_batch_dim} do not match."
        )

        condition = condition.transpose(1, 0)
        condition = torch.squeeze(condition, dim=1)
        emb_cond = self._embedding_net(condition)

        distributions = self.net(emb_cond)
        log_probs = distributions.log_prob(input)

        return log_probs

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
        condition = condition.transpose(1, 0)
        condition = torch.squeeze(condition, dim=1)

        emb_cond = self._embedding_net(condition)
        dists = self.net(emb_cond)
        return dists.sample(sample_shape)

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
        condition = condition.transpose(1, 0)
        condition = torch.squeeze(condition, dim=1)
        emb_cond = self._embedding_net(condition)
        dists = self.net(emb_cond)
        samples, log_probs = dists.rsample_and_log_prob(sample_shape)
        return samples, log_probs
