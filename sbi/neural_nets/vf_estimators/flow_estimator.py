import warnings
import torch
from torch import Tensor
from torch import nn

from sbi.neural_nets.vf_estimators.base import VectorFieldEstimator
from typing import Callable, Optional

class FlowMatchingEstimator(VectorFieldEstimator):
    r"""Flow Matching is an alternative to score matching that instead aim to learn a
    probability flow that maps a source distribution to a target distribution. Although
    conceptually similar to score matching, it does target a different vector field and 
    has different loss functions.
    
    Literature:
    - https://arxiv.org/pdf/2303.08797.pdf
    - https://arxiv.org/pdf/2210.02747.pdf
    """

    def __init__(self,
                 net: nn.Module,
                 condition_shape: torch.Size,
                 weight_fn: Callable = lambda t: 1.,
                 source_distribution: torch.distributions.Distribution = torch.distributions.Normal(0, 1)
    ) -> None:
        r"""A vector field estimator that learns a flow matching vector field, generally
        defined as a vector field that maps a source distribution to a target 
        distribution.
        

        Args:
            net: Base neural network, should take input, condition, and time (in [0,1]).
            condition_shape: Shape of the condition.
            mean_fn: Mean function of the flow matching vector field, specifying the mean
                     of the target distribution at a given time. Does specify the mean of
                     the the Gaussian conditional probability flow.
                     This function should satisfy the following boundary conditions:
                     - mean_fn(xs_source, xs_target, 0) = xs_source
                     - mean_fn(xs_source, xs_target, 1) = xs_target
            std_fn: Standard deviation function of the flow matching vector field,
                    specifying the standard deviation of the Gaussian conditional
                    probability flow.
                    This function should satisfy the following boundary conditions:
                    - std_fn(xs_source, xs_target, 0) = 0
                    - std_fn(xs_source, xs_target, 1) = 0
            mean_fn_grad: Optional method . Defaults to None.
            std_fn_grad: _description_. Defaults to None.
            weight_fn _description_. Defaults to lambdat:1..
            source_distribution: _description_. Defaults to torch.distributions.Normal(0, 1).
        """
        super().__init__(net, condition_shape)
        self.weight_fn = weight_fn
        self.source_distribution = source_distribution

        
    def mean_fn(self, xs_source: Tensor, xs_target: Tensor, times: Tensor):
        r"""Mean function of the flow matching vector field, specifying the mean
        of the target distribution at a given time. Does specify the mean of the 
        Gaussian conditional probability flow.
        
        This function should satisfy the following boundary conditions:
        - mean_fn(xs_source, xs_target, 0) = xs_source
        - mean_fn(xs_source, xs_target, 1) = xs_target

        Args:
            xs_source: Samples from the source distribution at time 0.
            xs_target: Samples from the target distribution at time 1.
            times: Time points in [0,1].

        Raises:
            NotImplementedError: This method should be implemented by the user.
        """
        raise NotImplementedError()
    
    def std_fn(self, xs_source: Tensor, xs_target: Tensor, times: Tensor) -> Tensor:
        r"""Standard deviation function of the flow matching vector field, specifying 
        the standard deviation of the Gaussian conditional probability flow.
        
        This function should satisfy the following boundary conditions:
            - std_fn(xs_source, xs_target, 0) ~ 0 (Should be small)
            - std_fn(xs_source, xs_target, 1) ~ 0 (Should be small)

        Args:
            xs_source: Samples from the source distribution at time 0.
            xs_target: Samples from the target distribution at time 1.
            times: Time ponts in [0,1].

        Raises:
            NotImplementedError: This method should be implemented by the user.
        """
        raise NotImplementedError()
    
    def mean_fn_grad(self, xs_source: Tensor, xs_target: Tensor, times: Tensor) -> Tensor:
        r""" Time derivative of the mean function. Required for the loss function.
        
        Note: This method can be implemented by the user, but is not required (autograd).

        Args:
            xs_source: Samples from the source distribution at time 0.
            xs_target: Samples from the target distribution at time 1.
            times: Time points in [0,1].

        Returns:
            Tensor: Mean function gradient.
        """
        with torch.enable_grad():
            times = times.clone().requires_grad_(True)
            m_t = self.mean_fn(xs_source, xs_target, times)
            grad = torch.autograd.grad(m_t, times, grad_outputs=torch.ones_like(m_t), create_graph=True)[0]
        return grad
    
    def std_fn_grad(self, xs_source: Tensor, xs_target: Tensor, times: Tensor) -> Tensor:
        r""" Time derivative of the std function. Required for the loss function.
        
        Note: This method can be implemented by the user, but is not required (autograd).

        Args:
            xs_source: Samples from the source distribution at time 0.
            xs_target: Samples from the target distribution at time 1.
            times: Time points in [0,1].

        Returns:
            Tensor: Std function gradient.
        """
        with torch.enable_grad():
            times = times.clone().requires_grad_(True)
            s_t = self.std_fn(xs_source, xs_target, times)
            grad = torch.autograd.grad(s_t, times, grad_outputs=torch.ones_like(s_t), create_graph=True)[0]
        return grad

    def forward(self, input: Tensor, condition: Tensor, time: Tensor):
        return self.net([input, condition, time])


    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        
        times = torch.rand((input.shape[0],) + (1,)*(len(input.shape)-1))
        eps = torch.randn_like(input)
        xs_target = input
        xs_source = self.source_distribution.sample(input.shape)

        m_t = self.mean_fn(xs_source, xs_target, times)
        s_t = self.std_fn(xs_source, xs_target, times)
        xs_t = m_t + s_t * eps

        u_t = self.std_fn_grad(xs_source, xs_target, times) * eps + self.mean_fn_grad(xs_source, xs_target, times)
        v_t = self.net(xs_t, condition, times)


        loss = torch.sum(self.weight_fn(times)*(u_t - v_t).pow(2), dim=-1)

        return loss
    
class OTFlowMatchingEstimator(FlowMatchingEstimator):
    """_summary_

    Args:
        FlowMatchingEstimator (_type_): _description_
    """
    
    def __init__(self, net: nn.Module, condition_shape: torch.Size, sigma_min:float=1e-3):
        super().__init__(net, condition_shape)
        self.sigma_min = sigma_min

    def mean_fn(self, xs_source: Tensor, xs_target: Tensor, times: Tensor):
        return  times*xs_target
    
    def std_fn(self, xs_source: Tensor, xs_target: Tensor, times: Tensor):
        return  (1 - (1-self.sigma_min)*times)
    
    
    
    
