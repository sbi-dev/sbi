# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Tuple

import torch
from torch import Tensor, nn
from zuko.flows.core import Flow

from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.sbi_types import Shape


class ZukoFlow(ConditionalDensityEstimator):
    r"""`zuko`- based normalizing flow density estimator.

    Flow type objects already have a .log_prob() and .sample() method, so here we just
    wrap them and add the .loss() method.
    """

    def __init__(
        self,
        net: Flow,
        embedding_net: nn.Module,
        input_shape: torch.Size,
        condition_shape: torch.Size,
    ):
        r"""Initialize the density estimator.

        Args:
            flow: Flow object.
            input_shape: Event shape of the input at which the density is being
                evaluated (and which is also the event_shape of samples).
            condition_shape: Event shape of the condition.
        """

        # assert len(condition_shape) == 1, "Zuko Flows require 1D conditions."
        super().__init__(
            net=net, input_shape=input_shape, condition_shape=condition_shape
        )
        self._embedding_net = embedding_net

    @property
    def embedding_net(self) -> nn.Module:
        r"""Return the embedding network."""
        return self._embedding_net

    def inverse_transform(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the inverse flow-transform of the inputs given a condition.

        The inverse transform is the transformation that maps the inputs back to the
        base distribution (noise) space.

        Args:
            input: Inputs to evaluate the inverse transform on of shape
                    (*batch_shape1, input_size).
            condition: Conditions of shape (*batch_shape2, *condition_shape).

        Raises:
            RuntimeError: If batch_shape1 and batch_shape2 are not broadcastable.

        Returns:
            noise: Transformed inputs.

        Note:
            This function should support PyTorch's automatic broadcasting. This means
            the function should behave as follows for different input and condition
            shapes:
            - (input_size,) + (batch_size,*condition_shape) -> (batch_size,)
            - (batch_size, input_size) + (*condition_shape) -> (batch_size,)
            - (batch_size, input_size) + (batch_size, *condition_shape) -> (batch_size,)
            - (batch_size1, input_size) + (batch_size2, *condition_shape)
                                                  -> RuntimeError i.e. not broadcastable
            - (batch_size1,1, input_size) + (batch_size2, *condition_shape)
                                                  -> (batch_size1,batch_size2)
            - (batch_size1, input_size) + (batch_size2,1, *condition_shape)
                                                  -> (batch_size2,batch_size1)
        """

        self._check_condition_shape(condition)
        condition_dims = len(self.condition_shape)

        # PyTorch's automatic broadcasting
        batch_shape_in = input.shape[:-1]
        batch_shape_cond = condition.shape[:-condition_dims]
        batch_shape = torch.broadcast_shapes(batch_shape_in, batch_shape_cond)
        # Expand the input and condition to the same batch shape
        input = input.expand(batch_shape + (input.shape[-1],))
        emb_cond = self._embedding_net(condition)
        emb_cond = emb_cond.expand(batch_shape + (emb_cond.shape[-1],))

        dists = self.net(emb_cond)
        noise = dists.transform(input)

        return noise

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the log probabilities of the inputs given a condition or multiple
        i.e. batched conditions.

        Args:
            input: Inputs to evaluate the log probability on. Of shape
                `(sample_dim, batch_dim, *event_shape)`.
            # TODO: the docstring is not correct here. in the code it seems we
            do not have a sample_dim for the condition.
            condition: Conditions of shape `(sample_dim, batch_dim, *event_shape)`.

        Raises:
            AssertionError: If `input_batch_dim != condition_batch_dim`.

        Returns:
            Sample-wise log probabilities, shape `(input_sample_dim, input_batch_dim)`.
        """
        input_batch_dim = input.shape[1]
        condition_batch_dim = condition.shape[0]

        assert condition_batch_dim == input_batch_dim, (
            f"Batch shape of condition {condition_batch_dim} and input "
            f"{input_batch_dim} do not match."
        )

        emb_cond = self._embedding_net(condition)
        distributions = self.net(emb_cond)
        log_probs = distributions.log_prob(input)

        return log_probs

    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        r"""Return the negative log-probability for training the density estimator.

        Args:
            input: Inputs of shape `(batch_dim, *input_event_shape)`.
            condition: Conditions of shape `(batch_dim, *condition_event_shape)`.

        Returns:
            Negative log-probability of shape `(batch_dim,)`.
        """

        return -self.log_prob(input.unsqueeze(0), condition)[0]

    def sample(self, sample_shape: Shape, condition: Tensor) -> Tensor:
        r"""Return samples from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape `(sample_dim, batch_dim, *event_shape)`.

        Returns:
            Samples of shape `(*sample_shape, condition_batch_dim)`.
        """
        emb_cond = self._embedding_net(condition)
        dists = self.net(emb_cond)

        samples = dists.sample(sample_shape)

        return samples

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        r"""Return samples and their density from the density estimator.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape (sample_dim, batch_dim, *event_shape).

        Returns:
            Samples of shape `(*sample_shape, condition_batch_dim, *input_event_shape)`
            and associated log probs of shape `(*sample_shape, condition_batch_dim)`.
        """
        emb_cond = self._embedding_net(condition)
        dists = self.net(emb_cond)

        samples, log_probs = dists.rsample_and_log_prob(sample_shape)
        return samples, log_probs
