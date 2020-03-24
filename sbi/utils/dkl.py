import torch


def dkl_monte_carlo_estimate(p, q, num_samples: int = 1000) -> torch.Tensor:
    """
    Computes the Monte-Carlo estimate of the Kullback-Leibler divergence of two distributions p and q.

    Unlike torch.distributions.kl.kl_divergence(p, q), this function does not require p and q to be
    torch.Distribution objects, but instead they only need sample() and log_prob() methods

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
        summed_log_ratio += p.log_prob(target_sample) - q.log_prob(target_sample)
    dkl = summed_log_ratio / num_samples
    return dkl
