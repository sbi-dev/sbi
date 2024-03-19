import torch
from torch import nn

from sbi.neural_nets.vf_estimators.base import VectorFieldEstimator
from typing import Callable, Optional

class FlowMachtingEstimator(VectorFieldEstimator):

    def __init__(self, net: nn.Module,
                 condition_shape: torch.Size,
                 mean_fn: Callable,
                 std_fn: Callable,
                 mean_fn_grad: Optional[Callable] = None,
                 std_fn_grad: Optional[Callable] = None,
                 weight_fn: Callable = lambda t: 1.,
                 source_distribution: torch.distributions.Distribution = torch.distributions.Normal(0, 1)
    ) -> None:
        super().__init__(net, condition_shape)
        self.weight_fn = weight_fn
        self.source_distribution = source_distribution
        self._set_mean_and_std_fn(mean_fn, std_fn, mean_fn_grad, std_fn_grad, weight_fn)

    def _set_mean_and_std_fn(self, mean_fn, std_fn, mean_fn_grad, std_fn_grad, weight_fn):
        self.mean_fn = mean_fn
        self.std_fn = std_fn
        if mean_fn_grad is None:
            def mean_fn_grad(xs_source, xs_target, times):
                with torch.enable_grad():
                    times = times.clone().requires_grad_(True)
                    m_t = self.mean_fn(xs_source, xs_target, times)
                    grad = torch.autograd.grad(m_t, times, grad_outputs=torch.ones_like(m_t), create_graph=True)[0]
                return grad

        if std_fn_grad is None:
            def std_fn_grad(xs_source, xs_target, times):
                with torch.enable_grad():
                    times = times.clone().requires_grad_(True)
                    s_t = self.std_fn(xs_source, xs_target, times)
                    grad = torch.autograd.grad(s_t, times, grad_outputs=torch.ones_like(s_t), create_graph=True)[0]
                return grad

        self.mean_fn_grad = mean_fn_grad
        self.std_fn_grad = std_fn_grad


    def forward(self, input, condition, time):
        return self.net(input, condition, time)

    def loss(self, input, condition):
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
    
class OTFlowMatchingEstimator(VectorFieldEstimator):
    def __init__(self, net: nn.Module, condition_shape: torch.Size) -> None:
        
        def mean_fn(xs_source, xs_target, times):
            return (1 - times) * xs_source + times * xs_target
        
        def std_fn(xs_source, xs_target, times):
            return torch.tensor([1e-5])
    
        super().__init__(net, condition_shape, mean_fn, std_fn)
