from typing_extensions import final
import torch
import numpy as np

from torch.distributions import constraints
from torch.distributions import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform, PowerTransform
from torch.distributions.utils import broadcast_all

import warnings


def gpdfit(x, sorted=True, eps=1e-8, return_quadrature=False):
    """ Pytorch version of gpdfit according to https://github.com/avehtari/PSIS/blob/master/py/psis.py """
    if not sorted:
        x, _ = x.sort()
    N = len(x)
    PRIOR = 3
    M = 30 + int(np.sqrt(N))

    bs = torch.arange(1, M + 1)
    bs = 1 - np.sqrt(M / (bs - 0.5))
    bs /= PRIOR * x[int(N / 4 + 0.5) - 1]
    bs += 1 / x[-1]

    ks = -bs
    temp = ks[:, None] * x
    ks = torch.log1p(temp).mean(axis=1)
    L = N * (torch.log(-bs / ks) - ks - 1)

    temp = torch.exp(L - L[:, None])
    w = 1 / torch.sum(temp, axis=1)

    dii = w >= 10 * eps
    if not torch.all(dii):
        w = w[dii]
        bs = bs[dii]
    w /= w.sum()

    # posterior mean for b
    b = torch.sum(bs * w)
    # Estimate for k
    temp = (-b) * x
    temp = torch.log1p(temp)
    k = torch.mean(temp)
    if return_quadrature:
        temp = -x
        temp = bs[:, None] * temp
        temp = torch.log1p(temp)
        ks = torch.mean(temp, axis=1)

    # estimate for sigma
    sigma = -k / b * N / (N - 0)
    # weakly informative prior for k
    a = 10
    k = k * N / (N + a) + a * 0.5 / (N + a)
    if return_quadrature:
        ks *= N / (N + a)
        ks += a * 0.5 / (N + a)

    if return_quadrature:
        return k, sigma, ks, w
    else:
        return k, sigma


class GerneralizedParetto(TransformedDistribution):
    r"""
    Samples from a generalized Pareto distribution.
    """
    arg_constraints = {
        "mu": constraints.real,
        "scale": constraints.positive,
        "k": constraints.real,
    }

    def __init__(self, mu, scale, k, validate_args=None):
        self.mu, self.scale, self.k = broadcast_all(mu, scale, k)
        base_dist = Uniform(0, 1, validate_args=validate_args)
        if k == 0:
            transforms = [
                ExpTransform().inv,
                AffineTransform(loc=self.mu, scale=self.scale),
            ]
        else:
            transforms = [
                PowerTransform(-self.k),
                AffineTransform(loc=-1.0, scale=1.0),
                AffineTransform(loc=self.mu, scale=self.scale / self.k),
            ]
        super(GerneralizedParetto, self).__init__(
            base_dist, transforms, validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GerneralizedParetto, _instance)
        new.mu = self.mu.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.k = self.k.expand(batch_shape)
        return super(GerneralizedParetto, self).expand(batch_shape, _instance=new)

    @property
    def mean(self):
        if self.k >= 1:
            warnings.warn("Mean does not exist, clamped k")
        k = self.k.clamp(max=0.99999)
        return self.mu + self.scale / (1 - k)

    @property
    def variance(self):
        if self.k >= 0.5:
            warnings.warn("Variance does not exist, clamped k")
        k = self.k.clamp(max=0.49999)
        return self.scale ** 2 / (1 - k) ** 2 * (1 - 2 * k)

    def cdf(self, x):
        # Why is this necessary???
        return 1 - super().cdf(x)

    def inv_cdf(self, x):
        y = 1 - x
        for t in self.transforms:
            y = t(y)
        return y

    def support(self):
        return constraints.greater_than(self.mu)


def paretto_smoothed_weights(weights):
    """ This models the largest importance weights as paretto distributed and smooth
    them as their quantiles as proposed in https://arxiv.org/pdf/1507.02646.pdf .
    
    They show that this rule is asymtotically unbiased """
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
    """ This clamps the largest weights to reduce variance. """
    with torch.no_grad():
        weight_mean = weights.mean()
        weights = weights.clamp(max=np.sqrt(len(weights) * weight_mean))
    return weights


def importance_resampling(
    num_samples, potential_fn, proposal, K=32, num_samples_batch=10000
):
    """ This will sample based on the importance weights. This can correct for partially
   collapsed modes in the variational approximation. """
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


def independent_mh(num_samples, potential_fn, proposal, T=20, num_samples_batch=10000):
    """ Independent metropolis hasting algorithm, which uses as proposal the variational
    posterior approximation"""
    final_samples = []
    num_samples_batch = min(num_samples_batch, num_samples)
    samples = proposal.sample((num_samples_batch,))
    sampling_steps = int(T / (int(num_samples / num_samples_batch)))
    for t in range(T):
        potential = potential_fn(samples) - proposal.log_prob(samples)
        new_samples = proposal.sample((num_samples_batch,))
        new_potential = potential_fn(new_samples) - proposal.log_prob(new_samples)
        acceptance_probability = torch.exp(new_potential - potential)
        u = torch.rand(num_samples_batch)
        mask = u < acceptance_probability
        samples[mask] = new_samples[mask]
        if ((t + 1) % sampling_steps) == 0:
            final_samples.append(samples)
    return torch.vstack(final_samples)


def random_direction_slice_sampler(
    num_samples, potential_fn, proposal, T=20, num_samples_batch=1000, steps=0.01
):
    """ Random direction slice sampler """
    final_samples = []
    num_samples_batch = min(num_samples_batch, num_samples)
    samples = proposal.sample((num_samples_batch,))
    sampling_steps = int(T / (int(num_samples / num_samples_batch)))

    for t in range(T):
        potential_function = potential_fn(samples)
        y = torch.rand(num_samples_batch) * potential_function.exp()

        random_directions = torch.randn(samples.shape)
        random_directions /= torch.linalg.norm(random_directions, dim=-1).unsqueeze(-1)
        lb = torch.zeros((num_samples_batch,))
        ub = torch.zeros((num_samples_batch,))
        mask_lb = y < potential_function.exp()
        mask_ub = mask_lb.clone()

        while mask_lb.any() and mask_ub.any():
            lb[mask_lb] -= steps
            ub[mask_ub] += steps
            proposed_samples_lb = (
                samples[mask_lb, :]
                + lb[mask_lb].unsqueeze(-1) * random_directions[mask_lb, :]
            )
            proposed_samples_ub = (
                samples[mask_ub, :]
                + ub[mask_ub].unsqueeze(-1) * random_directions[mask_ub, :]
            )

            potential_function_lb = potential_fn(proposed_samples_lb)
            potential_function_ub = potential_fn(proposed_samples_ub)

            mask_lb[mask_lb.clone()] = y[mask_lb] < potential_function_lb.exp()
            mask_ub[mask_ub.clone()] = y[mask_ub] < potential_function_ub.exp()
        x = torch.rand(num_samples_batch) * (ub - lb) + lb
        samples = samples + x.unsqueeze(-1) * random_directions
        if ((t + 1) % sampling_steps) == 0:
            final_samples.append(samples)
    return torch.vstack(final_samples)

