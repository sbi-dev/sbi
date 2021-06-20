from .mixture_of_flows import build_mixture
from .flows import build_flow
from .divergence_optimizers import (
    ElboOptimizer,
    IWElboOptimizer,
    RenjeyDivergenceOptimizer,
    TailAdaptivefDivergenceOptimizer,
    ForwardKLOptimizer,
)

import torch

# Some of the main arguments for
KWARGS_Q = ["flow", "num_components", "rsample", "permute", "batch_norm"]


def build_q(
    event_shape: torch.Size,
    support: torch.distributions.constraints.Constraint,
    flow: str = "spline_autoregressive",
    num_components: int = 1,
    **kwargs,
):
    """This method builds an normalizing flow or a mixture of normalizing flows.
    Args:
        event_shape: Event shape
        support: The support of the distribtuion
        flow: The type of flow
        num_components: Number of mixture components, default is one.
    
    Returns:
        [type]: [description]
    
    """
    if num_components > 1:
        return build_mixture(
            event_shape, support, num_components=num_components, type=flow, **kwargs
        )
    else:
        return build_flow(event_shape, support, type=flow, **kwargs)


def build_optimizer(posterior, loss, **kwargs):
    """ This methods builds an optimizer"""
    if loss.lower() == "elbo":
        optimizer = ElboOptimizer(posterior, **kwargs)
    elif loss.lower() == "iwelbo":
        optimizer = IWElboOptimizer(posterior, **kwargs)
    elif loss.lower() == "renjey_divergence":
        optimizer = RenjeyDivergenceOptimizer(posterior, **kwargs)
    elif loss.lower() == "tail_adaptive_fdivergence":
        optimizer = TailAdaptivefDivergenceOptimizer(posterior, **kwargs)
    elif loss.lower() == "forward_kl":
        optimizer = ForwardKLOptimizer(posterior, **kwargs)
    else:
        raise NotImplementedError("Unknown loss...")
    return optimizer
