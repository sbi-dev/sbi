import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import zuko
from torch import Tensor
from torch.distributions import Transform
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast

from sbi.neural_nets.estimators.base import ConditionalDensityEstimator


# abstract class to ensure forward signature for flow matching networks
class VectorFieldNet(nn.Module, ABC):
    @abstractmethod
    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor: ...


class FlowMatchingEstimator(ConditionalDensityEstimator):
    def __init__(
        self,
        net: VectorFieldNet,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net: nn.Module,
        zscore_transform_input: Optional[Transform] = None,
        num_freqs: int = 3,
        noise_scale: float = 1e-3,
        **kwargs,
    ) -> None:
        """Creates a vector field estimator for Flow Matching.

        Args:
            net: Neural network that estimates the vector field.
            input_shape: Shape of the input, e.g., the parameters.
            condition_shape: Shape of the condition, e.g., the data.
            noise_scale: Scale of the noise added to the vector field.
            embedding_net: Embedding network for the condition.
            zscore_transform_input: Transform to z-score the input.
            num_freqs: Number of frequencies to use for the positional time encoding.
            noise_scale: Scale of the noise added to the vector field.
        """

        super().__init__(
            net=net, input_shape=input_shape, condition_shape=condition_shape
        )

        self.noise_scale = noise_scale
        # Identity transform for z-scoring the input
        if zscore_transform_input is None:
            zscore_transform_input = zuko.transforms.IdentityTransform()
        self.zscore_transform_input: Transform = zscore_transform_input
        self._embedding_net = embedding_net

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
        zscored_input, embedded_condition, t = broadcast(
            zscored_input,  # type: ignore
            embedded_condition,
            t,
            ignore=1,
        )

        # return the estimated vector field
        return self.net(theta=zscored_input, x=embedded_condition, t=t)

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

        Args:
            condition: Condition for the normalizing flow.

        Returns:
            NormalizingFlow: flow with log_prob and sample methods via probability ODEs.
        """

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
