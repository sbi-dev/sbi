from typing import Callable, List, Optional

import torch
from torch import Tensor
from torch.distributions import Distribution

_SAMPLING_METHOD = {}


def register_sampling_method(
    cls: Optional[Callable] = None,
    name: Optional[str] = None,
) -> Callable:
    """Registers a sampling method which can be used to debias the variational posterior.

    Args:
        cls: The method to add.
        name: The name of the method


    """

    def _register(cls: Callable):
        if name is None:
            cls_name = cls.__name__
        else:
            cls_name = name
        if cls_name in _SAMPLING_METHOD:
            raise ValueError(f"The sampling method {cls_name} is already registered")
        else:
            _SAMPLING_METHOD[cls_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_sampling_method(name: str) -> Callable:
    """Returns the sampling_function of the specified method

    Args:
        name: The name of the sampling method.

    Returns:
        Callable: Returns the sampling method.

    """
    return _SAMPLING_METHOD[name]


def get_default_sampling_methods() -> List[str]:
    """Returns the list of all registered sampling methods.

    Returns:
        List[str]: Registered sampling methods.

    """
    return list(_SAMPLING_METHOD.keys())


@register_sampling_method(name="naive")
def naive_sampling(
    num_samples: int, potential_fn: Callable, proposal: Distribution, **kwargs
) -> Tensor:
    """Basic sampling method, which just samples from the proposal i.e. the variational
    posterior.

    Args:
        num_samples: Number of samples to draw.
        potential_fn: Potential function, this may be used to debias the proposal.
        proposal: Proposal distribution to propose samples.

    Returns:
        Tensor: Samples of shape (num_samples, event_shape)

    """
    return proposal.sample(torch.Size((num_samples,)))


@register_sampling_method(
    name="sir",
)
def importance_resampling(
    num_samples: int,
    potential_fn: Callable,
    proposal: Distribution,
    K: int = 32,
    num_samples_batch: int = 10000,
    **kwargs,
) -> Tensor:
    """Perform sequential importance resampling (SIR).

    Args:
        num_samples: Number of samples to draw.
        potential_fn: Potential function, this may be used to debias the proposal.
        proposal: Proposal distribution to propose samples.
        K: Number of proposed samples form which only one is selected based on its
            importance weight.
        num_samples_batch: Number of samples processed in parallel. For large K you may
            want to reduce this, depending on your memory capabilities.

    Returns:
        Tensor: Samples of shape (num_samples, event_shape)

    """
    final_samples = []
    num_samples_batch = min(num_samples, num_samples_batch)
    iters = int(num_samples / num_samples_batch)
    for _ in range(iters):
        batch_size = min(num_samples_batch, num_samples - len(final_samples))
        with torch.no_grad():
            thetas = proposal.sample(torch.Size((batch_size * K,)))
            logp = potential_fn(thetas)
            logq = proposal.log_prob(thetas)
            weights = (logp - logq).reshape(batch_size, K).softmax(-1).cumsum(-1)
            u = torch.rand(batch_size, 1, device=thetas.device)
            mask = torch.cumsum(weights >= u, -1) == 1
            samples = thetas.reshape(batch_size, K, -1)[mask]
            final_samples.append(samples)
    return torch.vstack(final_samples)
