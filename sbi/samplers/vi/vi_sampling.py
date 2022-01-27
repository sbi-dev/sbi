import torch
from torch import Tensor
from torch.distributions import Distribution
import numpy as np

from typing import Optional, List, Callable

import warnings

from .vi_quality_controll import gpdfit, GerneralizedParetto
from sbi.inference.potentials.base_potential import BasePotential

_SAMPLING_METHOD = {}
_SAMPLING_PARAMETERS_DOC = {}


def register_sampling_method(
    cls: Optional[object] = None,
    name: Optional[str] = None,
    doc: Optional[str] = "",
):
    """Registers a sampling method which can be used to debias the variational posterior.

    Args:
        cls: The method to add.
        name: The name of the method
        doc: A short description, which will be shown to the user.


    """

    def _register(cls):
        if name is None:
            cls_name = cls.__name__
        else:
            cls_name = name
        if cls_name in _SAMPLING_METHOD:
            raise ValueError(f"The sampling method {cls_name} is already registered")
        else:
            _SAMPLING_METHOD[cls_name] = cls
            _SAMPLING_PARAMETERS_DOC[cls_name] = doc
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


def get_sampling_method_parameters_doc(name: str) -> str:
    """Returns a description of a sampling method method

    Args:
        name: The name of the sampling method.

    Returns:
        str: Short description of the method.

    """
    return _SAMPLING_PARAMETERS_DOC[name]


def get_default_sampling_methods() -> List[str]:
    """Returns the list of all registered sampling methods!

    Returns:
        List[str]: Registered sampling methods.

    """
    return list(_SAMPLING_METHOD.keys())


def paretto_smoothed_weights(weights: Tensor) -> Tensor:
    """This will model M largest importance weights as a Generalized Paretto
    distribution, and will replace them with respective quantiles. All other weights are
    unchanged. The method is asymptotically unbiased, but also can be used for
    diagnostics as the parameter k can be related to the minimum sample size for
    accurate estimation which is only finite for k < 1.

    Args:
        weights: A tensor of importance weights.

    Returns:
        Tensor: A tensor of importance paretto smoothed importance weights.

    References:
        _Pareto Smoothed Importance Sampling_, Aki Vehtari, Daniel Simpson, Andrew
            Gelman, Yuling Yao, Jonah Gabry, 2015, https://arxiv.org/abs/1507.02646.

    """
    with torch.no_grad():
        N = len(weights)
        M = int(min(N / 5, 3 * np.sqrt(N)))
        vals, index = weights.sort()
        largest_weigths = vals[-M:]
        mu = vals[-M]
        k, sigma = gpdfit(largest_weigths)
        if k > 1.0:
            warnings.warn("The estimator has infinite variance")
        p = GerneralizedParetto(mu, sigma, k)
        new_weights = p.inv_cdf((torch.arange(1, M + 1) - 0.5) / M)
        weights[index[-M:]] = torch.min(new_weights, largest_weigths[-1])
    return weights


def clamp_weights(weights: Tensor) -> Tensor:
    """This will clamp the M largest importance weights to a constant value i.e.
    sqrt(N)*mean(w).

    Args:
        weights: A tensor of importance weights.

    Returns:
        Tensor: A tensor of importance clamped importance weights.

    References:
        _Truncated Importance Sampling_, Edward L Ionides, 2012, https://www.tandfonline.com/doi/abs/10.1198/106186008X320456.

    """
    with torch.no_grad():
        weight_mean = weights.mean()
        weights = weights.clamp(max=np.sqrt(len(weights) * weight_mean))
    return weights


@register_sampling_method(name="naive", doc="naive: Just samples from q")
def naive_sampling(
    num_samples: int, potential_fn: BasePotential, proposal: Distribution, **kwargs
) -> Tensor:
    """Basic sampling method, which just samples from the proposal i.e. the variational posterior.

    Args:
        num_samples: Number of samples to draw.
        potential_fn: Potential function, this may be used to debias the proposal.
        proposal: Proposal distribution to propose samples.

    Returns:
        Tensor: Samples of shape (num_samples, event_shape)

    """
    return proposal.sample((num_samples,))


@register_sampling_method(
    name="sir",
    doc="sir: Performs sampling importance resampling. \n'K': Number of importance samples \n'num_samples_batch': How many samples are drawn in parallel (For large K you may have to decrease this due to memory limitation)",
)
def importance_resampling(
    num_samples: int,
    potential_fn: BasePotential,
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
            thetas = proposal.sample((batch_size * K,))
            logp = potential_fn(thetas)
            logq = proposal.log_prob(thetas)
            weights = (logp - logq).reshape(batch_size, K).softmax(-1).cumsum(-1)
            u = torch.rand(batch_size, 1)
            mask = torch.cumsum(weights >= u, -1) == 1
            samples = thetas.reshape(batch_size, K, -1)[mask]
            final_samples.append(samples)
    return torch.vstack(final_samples)
