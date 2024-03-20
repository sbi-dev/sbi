import math
import torch
import torch.nn as nn

from sbi.neural_nets.density_estimators.base import DensityEstimator
from torch.distributions import Distribution
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.nn import MLP
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast


class ZukoFlowMatchingEstimator(DensityEstimator):
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

    def embedding_net(self):
        raise NotImplementedError(
            "The Flow Matching estimator does not support an embedding net for now."
        )

    def log_prob(self, input: torch.Tensor, condition: torch.Tensor, **kwargs) -> torch.Tensor:

        # check for possible batch dimension in the condition
        if len(condition.shape) > len(self._condition_shape):
            raise ValueError(
                "The condition has a batch dimension, which is currently not supported."
            )

        dist = self.flow(x_o=condition)
        log_prob = dist.log_prob(input)

        return log_prob

    def loss(self, theta: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
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

    def sample(self, sample_shape: torch.Size, condition: torch.Tensor, **kwargs) -> torch.Tensor:
        # check for possible batch dimension in the condition
        if len(condition.shape) > len(self._condition_shape):
            raise ValueError(
                "The condition has a batch dimension, which is currently not supported."
            )

        dist = self.flow(x_o=condition)
        samples = dist.sample(sample_shape)
        return samples
    
    def flow(self, x_o: torch.Tensor) -> Distribution:
        return NormalizingFlow(
            transform=FreeFormJacobianTransform(
                f=lambda t, theta: self.vf_estimator(theta, x_o, t),
                t0=x_o.new_tensor(0.0),
                t1=x_o.new_tensor(1.0),
                phi=(x_o, *self.vf_estimator.net.parameters()),
            ),
            base=DiagNormal(
                torch.zeros(self.theta_dim), torch.ones(self.theta_dim)
            ).expand(x_o.shape[:-1]),
        )
