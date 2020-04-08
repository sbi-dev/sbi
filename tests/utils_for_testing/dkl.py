from typing import Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.sbi_posterior import Posterior


def dkl_via_monte_carlo(
    p: Union[Posterior, Distribution],
    q: Union[Posterior, Distribution],
    num_samples: int = 1000,
) -> Tensor:
    """
    Returns Monte-Carlo estimate of the Kullback-Leibler divergence of distributions p,
    q.

    Unlike torch.distributions.kl.kl_divergence(p, q), this function does not require p
    and q to be torch.Distribution objects, but just to provide sample() and log_prob()
    methods. 

    For added flexibility, we squeeze the output of log_prob() and hence can handle
    outputs such as torch.tensor([[p_1], [p_2], [p_3]]), instead of just
    torch.tensor([p_1, p_2, p_3]) (like torch.distributions.kl.kl_divergence(p, q)),
    with p_n being probabilities.

    Computes D = \int p(x) * log(p(x)/q(x)) dx \approx 1/N * log(p(x)/q(x)) 
    
    Args: 
        p, q: distribution-like objects with sample() and log_prob() methods 
        num_samples: number of samples that the Monte-Carlo estimate is based on

    """

    cumulative_log_ratio = torch.tensor([0.0])
    for _ in range(num_samples):
        target_sample = p.sample()
        # squeeze to make the shapes match. The output from log_prob() is either
        # torch.tensor([[p_1], [p_2], [p_3]]) or torch.tensor([p_1, p_2, p_3]), so we
        # squeeze to make both of them torch.tensor([p_1, p_2, p_3])
        cumulative_log_ratio += torch.squeeze(
            p.log_prob(target_sample)
        ) - torch.squeeze(q.log_prob(target_sample))

    dkl = cumulative_log_ratio / num_samples

    return dkl
