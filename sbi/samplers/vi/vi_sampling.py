import torch
import numpy as np

from typing import Optional

import warnings

from .vi_quality_controll import gpdfit, GerneralizedParetto

_SAMPLING_METHOD = {}


def register_sampling_method(
    cls: Optional[object] = None,
    name: Optional[str] = None,
):
    def _register(cls):
        if name is None:
            cls_name = cls.__name__
        else:
            cls_name = name
        if cls_name in _SAMPLING_METHOD:
            raise ValueError(f"The sampling {cls_name} is already registered")
        else:
            _SAMPLING_METHOD[cls_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_sampling_method(name: str):
    return _SAMPLING_METHOD[name]


def paretto_smoothed_weights(weights):
    """This models the largest importance weights as paretto distributed and smooth
    them as their quantiles as proposed in https://arxiv.org/pdf/1507.02646.pdf .

    They show that this rule is asymtotically unbiased"""
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


def clamp_weights(weights):
    """This clamps the largest weights to reduce variance."""
    with torch.no_grad():
        weight_mean = weights.mean()
        weights = weights.clamp(max=np.sqrt(len(weights) * weight_mean))
    return weights


@register_sampling_method(name="naive")
def naive_sampling(num_samples, potential_fn, proposal, **kwargs):
    return proposal.sample((num_samples,))


@register_sampling_method(name="sir")
def importance_resampling(
    num_samples, potential_fn, proposal, K=32, num_samples_batch=10000, **kwargs
):
    """This will sample based on the importance weights. This can correct for partially
    collapsed modes in the variational approximation."""
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
