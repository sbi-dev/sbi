import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from torch import Tensor
from zuko.utils import broadcast

from sbi.neural_nets.estimators.vector_field_estimator import ConditionalVectorFieldEstimator


# abstract class to ensure forward signature for flow matching networks
class VectorFieldNet(nn.Module, ABC):
    @abstractmethod
    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor: ...


class FlowMatchingEstimator(ConditionalVectorFieldEstimator):
    r"""
    
    Rectified flow matching estimator class that estimates the conditional vector field, 
    i.e. the expectation E[x_t - x_0 | x_0 = x].

    """
    
    SCORE_DEFINED: bool = True  # Whether the score is defined for this estimator. Required for gradient-based methods.
    SDE_DEFINED: bool = True  # Whether the SDE functions - score, drift and diffusion - are defined for this estimator. 
    MARGINALS_DEFINED: bool = True  # Whether the marginals are defined for this estimator. Required for iid methods.
    
    def __init__(
        self,
        net: VectorFieldNet,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net: Optional[nn.Module] = None,
        num_freqs: int = 3,
        noise_scale: float = 1e-3,
        zscore_transform_input=None,  # This is ignored and will be removed.
        **kwargs,
    ) -> None:
        """Creates a vector field estimator for Flow Matching.

        Args:
            net: Neural network that estimates the vector field.
            input_shape: Shape of the input, e.g., the parameters.
            condition_shape: Shape of the condition, e.g., the data.
            noise_scale: Scale of the noise added to the vector field.
            embedding_net: Embedding network for the condition.
            num_freqs: Number of frequencies to use for the positional time encoding.
            noise_scale: Scale of the noise added to the vector field.
        """

        super().__init__(
            net=net, input_shape=input_shape, condition_shape=condition_shape
        )

        self.noise_scale = noise_scale
        # Identity transform for z-scoring the input
        self._embedding_net = embedding_net if embedding_net is not None else nn.Identity()

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

        # broadcast to match shapes of theta, x, and t
        input, embedded_condition, t = broadcast(
            input,  # type: ignore
            embedded_condition,
            t,
            ignore=1,
        )

        # return the estimated vector field
        return self.net(theta=input, x=embedded_condition, t=t)

    def loss(self, input: Tensor, condition: Tensor, times: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Return the loss for training the density estimator. More precisely,
        we compute the conditional flow matching loss with naive optimal
        trajectories as described in the original paper.

        Args:
            theta: Parameters.
            x: Observed data.
            times: Time steps to compute the loss at. Optional, will sample from [0, 1] if not provided.
        
        Returns:
            Loss value.
        """
        # randomly sample the time steps to compare the vector field at
        # different time steps
        if times is None:
            times = torch.rand(input.shape[:-1], device=input.device, dtype=input.dtype)
        t_ = times[..., None]

        # sample from probability path at time t
        # TODO: Change to notation from Lipman et al. or Tong et al.
        epsilon = torch.randn_like(input)
        theta_prime = (1 - t_) * input + (t_ + self.noise_scale) * epsilon

        # compute vector field at the sampled time steps
        vector_field = epsilon - input

        # compute the mean squared error between the vector fields
        return torch.mean(
            (self.forward(theta_prime, condition, times) - vector_field) ** 2, dim=-1
        )

    def ode_fn(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        """
        ODE flow function of the vector field estimator.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.
            t: Time.

        Returns:
            Estimated vector field.
        """
        return self.forward(input, condition, t)

    def score(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        """Score function of the vector field estimator.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.
            t: Time.

        Returns:
            Score function of the vector field estimator.
        """
        v = self.forward(input, condition, t)
        return (-(1 - t) * v - input) / torch.maximum(t, torch.tensor(1e-6))
    

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Drift function for the flow matching estimator.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        return - input / torch.maximum(1 - times, torch.tensor(1e-6))

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Diffusion function for the flow matching estimator.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Diffusion function at a given time.
        """
        return torch.sqrt(2 * times /  torch.maximum(1 - times, torch.tensor(1e-6)))
    
    def mean_t_fn(self, times: Tensor) -> Tensor:
        r"""Linear coefficient of the perturbation kernel mean for the flow matching estimator.

        The general form of the perturbation kernel for rectified flows is:
            mean_t(t) = (1 - t) * mean_0 + t * mean_base
        
        So far, the implementation of iid methods assumes that the mean_base is 0.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Mean function at a given time.
        """
        mean_t = 1 - times
        for _ in range(len(self.input_shape)):
            mean_t = mean_t.unsqueeze(-1)
        return mean_t

    def std_fn(self, times: Tensor) -> Tensor:
        """Standard deviation of the perturbation kernel for the flow matching estimator.

        The general form of the perturbation kernel for rectified flows is:
            std_t(t) = t * std_base

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Standard deviation at a given time.
        """
        std_t = times * self.std_base
        for _ in range(len(self.input_shape)):
            std_t = std_t.unsqueeze(-1)
        return std_t
  