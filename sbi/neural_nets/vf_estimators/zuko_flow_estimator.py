import math

import torch
import torch.nn as nn
from zuko.nn import MLP
from zuko.utils import broadcast

from sbi.neural_nets.vf_estimators.base import VectorFieldEstimator


class ZukoFlowMatchingEstimator(VectorFieldEstimator):
    def __init__(
        self,
        theta_shape: torch.Size,
        condition_shape: torch.Size,
        net: nn.Module = None,
        frequency: int = 3,
        eta: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        """Creates a vector field estimator for Flow Matching.

        Args:
            theta_shape: Shape of the parameters.
            condition_shape: Shape of observed data.
            net: Regression network to estimate v at time t which accepts
            input shape (theta_shape + condition_shape + 2 * freq). Defaults to None.
            frequency: Frequency of the embedding. Defaults to 3.
            eta: Minimal variance of the conditional probability path. Defaults to 1e-3.
        """
        # todo: add embedding net
        # instantiate the regression network
        if not net:
            net = MLP(
                in_features=theta_shape + condition_shape + 2 * frequency,
                out_features=theta_shape,
                hidden_features=[64] * 5,
                activation=nn.ELU,
            )
        elif isinstance(net, nn.Module):
            pass
        else:
            raise ValueError("net must be an instance of torch.nn.Module")

        super().__init__(net=net, condition_shape=condition_shape)
        self.device = device
        self.theta_shape = theta_shape
        self.frequency = torch.arange(1, frequency + 1, device=self.device) * math.pi
        self.eta = eta

    def forward(
        self, theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        # positional encoding of time steps
        t = self.frequency * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        # broadcast to match shapes of theta, x, and t
        theta, x, t = broadcast(theta, x, t, ignore=1)

        # return the estimated vector field
        return self.net(torch.cat((theta, x, t), dim=-1))

    def loss(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # randomly sample the time steps to compare the vector field at different time steps
        t = torch.rand(theta.shape[:-1], device=theta.device, dtype=theta.dtype)
        t_ = t[..., None]

        # sample from probability path at time t
        epsilon = torch.randn_like(theta)
        theta_prime = (1 - t_) * theta + (t_ + self.eta) * epsilon

        # compute vector field at the sampled time steps
        vector_field = epsilon - theta

        # compute the mean squared error between the vector fields
        return (self.net(theta_prime, x, t) - vector_field).pow(2).mean()
