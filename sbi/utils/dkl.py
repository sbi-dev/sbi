import torch
import torch.distributions
from sbi.inference.posteriors.sbi_posterior import Posterior
from typing import Union


def dkl_via_monte_carlo(
    p: Union[Posterior, torch.distributions.Distribution],
    q: Union[Posterior, torch.distributions.Distribution],
    num_samples: int = 1000,
) -> torch.Tensor:
    """
    Returns the Monte-Carlo estimate of the Kullback-Leibler divergence of two
     distributions p and q.

    Unlike torch.distributions.kl.kl_divergence(p, q), this function does not require p
     and q to be torch.Distribution objects, but instead they only need sample() and
     log_prob() methods. In addition, it squeezes the log_prob() outputs, which makes it
     more flexible in the output format of the log_prob() function, e.g. it can handle
     outputs such as torch.tensor([[p_1], [p_2], [p_3]]), instead of only
     torch.tensor([p_1, p_2, p_3]) (like torch.distributions.kl.kl_divergence(p, q)),
     with p_n being probabilities.

    Computes D = \int p(x) * log(p(x)/q(x)) dx \approx 1/N * log(p(x)/q(x))
    Args:
        p: distribution that has sample() and log_prob() methods
        q: distribution that has sample() and log_prob() methods
        num_samples: number of samples that the Monte-Carlo mean is based on

    Returns: Kullback-Leibler divergence from p to q
    """

    summed_log_ratio = torch.tensor([0.0])
    for _ in range(num_samples):
        target_sample = p.sample()
        # squeeze to make the shapes match. The output from log_prob() is either
        # torch.tensor([[p_1], [p_2], [p_3]]) or torch.tensor([p_1, p_2, p_3]), so we
        # squeeze to make both of them torch.tensor([p_1, p_2, p_3])
        summed_log_ratio += torch.squeeze(p.log_prob(target_sample)) - torch.squeeze(
            q.log_prob(target_sample)
        )
    dkl = summed_log_ratio / num_samples
    return dkl
