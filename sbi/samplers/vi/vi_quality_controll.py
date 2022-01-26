import torch
import numpy as np

from typing import Optional, Tuple, Callable

from torch.distributions import constraints
from torch.distributions import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform, PowerTransform
from torch.distributions.utils import broadcast_all

import warnings


_QUALITY_METRIC = {}
_METRIC_MESSAGE = {}


def register_quality_metric(
    cls: Optional[object] = None,
    name: Optional[str] = None,
    msg: Optional[str] = None,
):
    """This method will register a given metric for cheap quality evaluation of
    variational posteriors!

    Args:
        cls: Function to compute the metric.
        name: Associated name.
        msg: Short description on how to interpret the metric.

    """

    def _register(cls):
        if name is None:
            cls_name = cls.__name__
        else:
            cls_name = name
        if cls_name in _QUALITY_METRIC:
            raise ValueError(f"The sampling {cls_name} is already registered")
        else:
            _QUALITY_METRIC[cls_name] = cls
            _METRIC_MESSAGE[cls_name] = msg
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_quality_metric(name: str) -> Tuple[Callable, str]:
    """Returns the quality, metric as well as a short description."""
    return _QUALITY_METRIC[name], _METRIC_MESSAGE[name]


def gpdfit(x, sorted=True, eps=1e-8, return_quadrature=False):
    """Pytorch version of gpdfit according to
    https://github.com/avehtari/PSIS/blob/master/py/psis.py. This function will compute
    an maximum likelihood (regularized, thus more an MAP) for a Generalized Paretto Distribution."""
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


# TODO REMOVE ? THIS is used by parettor smoothed importance sampling (not realy used)
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
        # It's strange that this is necessary
        return 1 - super().cdf(x)

    def icdf(self, x):
        return super().icdf(1 - x)

    def inv_cdf(self, x):
        y = 1 - x
        for t in self.transforms:
            y = t(y)
        return y

    def support(self):
        return constraints.greater_than(self.mu)


def basic_checks(posterior, N=int(1e4)):
    prior = posterior._prior
    prior_samples = prior.sample((N,))
    samples = posterior.sample((N,))
    assert (torch.isfinite(samples)).all(), "Some of the samples are not finite"
    assert (
        prior.support.check(samples)
    ).all(), "Some of the samples are not within the prior support!"
    assert (
        torch.isfinite(posterior.log_prob(samples))
    ).all(), "The log probability is not finite for some samples"
    assert (
        torch.isfinite(posterior.log_prob(prior_samples))
    ).all(), "The log probability is not finite for some samples"


def psis_diagnostics(potential_function, q, proposal=None, N=int(5e4)):
    """This will evaluate the posteriors quality, best on importance weights.
    See https://arxiv.org/pdf/1802.02538.pdf for details"""
    M = int(min(N / 5, 3 * np.sqrt(N)))
    with torch.no_grad():
        if proposal is None:
            samples = q.sample((N,))
        else:
            samples = proposal.sample((N,))
        log_q = q.log_prob(samples)
        log_potential = potential_function(samples)
        logweights = log_potential - log_q
        logweights = logweights[torch.isfinite(logweights)]
        logweights_max = logweights.max()
        weights = torch.exp(logweights - logweights_max)  # Thus will only affect scale
        vals, _ = weights.sort()
        largest_weigths = vals[-M:]
        k, _ = gpdfit(largest_weigths)
    return k


def proportional_to_joint_diagnostics(potential_function, q, proposal=None, N=int(1e4)):
    """If we plot logp(x, theta) and logq(theta) side by side they should admit a
    linear relationship!
    """

    with torch.no_grad():
        if proposal is None:
            samples = q.sample((N,))
        else:
            samples = proposal.sample((N,))
        log_q = q.log_prob(samples)
        log_potential = potential_function(samples)

        X = log_q.exp().unsqueeze(-1)
        Y = log_potential.exp().unsqueeze(-1)
        w = torch.linalg.solve(X.T @ X, X.T @ Y)  # Linear regression

        residuals = Y - w * X
        var_res = torch.sum(residuals ** 2)
        var_tot = torch.sum((Y - Y.mean()) ** 2)
        r2 = 1 - var_res / var_tot  # R2 statistic to evaluate fit
    return r2


@register_quality_metric(
    name="psis",
    msg="A good variational posterior will have a score smaller than 0.7. Bad approximations typically admit a score greater than 1. This metric is a measure of 'constantness' for importance weights. \n NOTE: This metric is not sensitive for mode collapse",
)
def psis_q(posterior):
    basic_checks(posterior)
    return psis_diagnostics(posterior.potential_fn, posterior.q)


@register_quality_metric(
    name="psis_prior",
    msg="A good variational posterior will have a score smaller than 0.7. Bad approximations typically admit a score greater than 1. This metric is a measure of 'constantness' for importance weights. NOTE: This may take a while",
)
def psis_prior(posterior):
    basic_checks(posterior)
    return psis_diagnostics(
        posterior.potential_fn, posterior.q, proposal=posterior.prior, N=int(1e7)
    )


@register_quality_metric(
    name="proportionality",
    msg="A good variational posterior will have a score greater near 1. Bad approximations typically admit a score smaller than zero. This metric is a measure of proportionality of p and q. \n NOTE: This metric is not sensitive for mode collapse",
)
def proportionality(posterior):
    basic_checks(posterior)
    proportional_to_joint_diagnostics(posterior.potential_function, posterior.q)
