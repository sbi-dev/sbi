# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import zuko
from torch import Tensor
from torch.distributions import Transform
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.flows.core import Flow
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast

from sbi.neural_nets.density_estimators.base import ConditionalDensityEstimator
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


class FlowMatchingEstimator(ConditionalDensityEstimator):
    def __init__(
        self,
        net: nn.Module,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net_input: nn.Module,
        embedding_net_condition: nn.Module,
        zscore_transform_input: Optional[Transform] = None,
        num_freqs: int = 3,
        noise_scale: float = 1e-3,
    ) -> None:
        """Creates a vector field estimator for Flow Matching.

        Args:
            net: Neural network that estimates the vector field.
            input_shape: Shape of the input.
            condition_shape: Shape of the condition.
            noise_scale: Scale of the noise added to the vector field.
            embedding_net_input: Embedding network for the input.
            embedding_net_condition: Embedding network for the condition.
            num_freqs: Number of frequencies to use for the positional time encoding.
        """

        super().__init__(
            net=net, input_shape=input_shape, condition_shape=condition_shape
        )

        self.noise_scale = noise_scale
        # Identity transform for z-scoring the input
        if zscore_transform_input is None:
            zscore_transform_input = zuko.transforms.IdentityTransform()
        self.zscore_transform_input: Transform = zscore_transform_input
        self._embedding_net = embedding_net_input
        self._embedding_net_condition = embedding_net_condition

        self.register_buffer("freqs", torch.arange(1, num_freqs + 1) * math.pi)
        self.register_buffer('zeros', torch.zeros(input_shape))
        self.register_buffer('ones', torch.ones(input_shape))

    @property
    def embedding_net(self):
        return self._embedding_net

    def forward(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        # positional encoding of time steps
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        # embed the input and condition
        embedded_condition = self._embedding_net(condition)
        zscored_input = self.zscore_transform_input(input)

        # broadcast to match shapes of theta, x, and t
        # TODO: Does this comply with SBI shape conventions?
        zscored_input, embedded_condition, t = broadcast(
            zscored_input,  # type: ignore
            embedded_condition,
            t,
            ignore=1,
        )

        # return the estimated vector field
        return self.net(torch.cat((zscored_input, embedded_condition, t), dim=-1))

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        """Return the loss for training the density estimator. More precisely,
        we compute the conditional flow matching loss with naive optimal
        trajectories as described in the original paper.

        Args:
            theta: Parameters.
            x: Observed data.
        """
        # randomly sample the time steps to compare the vector field at
        # different time steps
        t = torch.rand(input.shape[:-1], device=input.device, dtype=input.dtype)
        t_ = t[..., None]

        # sample from probability path at time t
        # TODO: Change to notation from Lipman et al. or Tong et al.
        epsilon = torch.randn_like(input)
        theta_prime = (1 - t_) * input + (t_ + self.noise_scale) * epsilon

        # compute vector field at the sampled time steps
        vector_field = epsilon - input

        # compute the mean squared error between the vector fields
        return torch.mean(
            (self.forward(theta_prime, condition, t) - vector_field) ** 2, dim=-1
        )

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        # TODO: comply with new shape conventions.
        input_batch_dim = input.shape[1]
        condition_batch_dim = condition.shape[0]

        assert condition_batch_dim == input_batch_dim, (
            f"Batch shape of condition {condition_batch_dim} and input "
            f"{input_batch_dim} do not match."
        )

        # the flow will apply and take into account input zscoring.
        log_probs = self.flow(condition=condition).log_prob(input)
        return log_probs

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        # TODO: comply with new shape conventions.

        # the flow will take care of inverse z-scoring.
        samples = self.flow(condition=condition).sample(sample_shape)
        return samples

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        samples, log_probs = self.flow(condition).rsample_and_log_prob(sample_shape)

        return samples, log_probs

    def flow(self, condition: Tensor) -> NormalizingFlow:
        """Return the normalizing flow.

        Here, the actual continuous normalizing flow is created via zuko. It
        gives access to the log_prob() and sample() methods and internally calls
        ODE routines to compute them.
        """

        # TODO: zscore before or after CNF transform?
        transform = zuko.transforms.ComposedTransform(
            FreeFormJacobianTransform(
                f=lambda t, input: self.forward(input, condition, t),
                t0=condition.new_tensor(0.0),
                t1=condition.new_tensor(1.0),
                phi=(condition, *self.net.parameters()),
            ),
            self.zscore_transform_input,
        )

        return NormalizingFlow(
            transform=transform,
            base=DiagNormal(self.zeros, self.ones).expand(condition.shape[:-1]),
        )
