from pkg_resources import Distribution
import torch
import numpy as np

from typing import Optional, Tuple, Callable

from torch.distributions import constraints
from torch.distributions import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform, PowerTransform
from torch.distributions.utils import broadcast_all

import warnings

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.base_potential import BasePotential


_QUALITY_METRIC = {}
_METRIC_MESSAGE = {}


def register_quality_metric(
    cls: Optional[object] = None,
    name: Optional[str] = None,
    msg: Optional[str] = None,
) -> None:
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
            raise ValueError(f"The metric {cls_name} is already registered")
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


def gpdfit(
    x, sorted: bool = True, eps: float = 1e-8, return_quadrature: bool = False
) -> Tuple:
    """Maximum aposteriori estimate of a Generalized Paretto distribution.

    Pytorch version of gpdfit according to
    https://github.com/avehtari/PSIS/blob/master/py/psis.py. This function will compute
    an MAP (more stable than the MLE estimator).


    Args:
        x: Tensor of floats, the data which is used to fit the GPD.
        sorted: If x is already sorted
        eps: Numerical stability jitter
        return_quadrature: Weather to return individual results.
    Returns:
        Tuple: Parameters of the Generalized Paretto Distribution.

    """
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


# NOTE We may remove this if not required
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


def basic_checks(posterior: NeuralPosterior, N: int = int(1e4)):
    """Makes some basic checks to ensure the distribution is well defined.

    Args:
        posterior: Variational posterior object to check.
        N: Number of samples that are checked.


    """
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


def psis_diagnostics(
    potential_function: BasePotential,
    q: Distribution,
    proposal: Optional[Distribution] = None,
    N: int = int(5e4),
) -> float:
    """This will evaluate the posteriors quality by investingating its importance
    weights. If q is a perfect posterior approximation then q(theta) ~
    potential_function(theta) thus logw = log potential_function(theta) - log q must be
    constant. This function will fit a Generalized Paretto distribution to the tails of
    w. The shape parameter k serves as metric as detailed in [1].



    Args:
        potential_function: Potential function of target.
        q: Variational distribution, should be proportional to the potential_function
        proposal: Proposal for samples. Typically this is q.
        N: Number of samples involved in the test.

    Returns:
        float: Quality metric

    Reference:
        [1] _Yes, but Did It Work?: Evaluating Variational Inference_, Yuling Yao, Aki
        Vehtari, Daniel Simpson, Andrew Gelman, 2018, https://arxiv.org/abs/1802.02538

    """
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


def proportional_to_joint_diagnostics(
    potential_function: BasePotential,
    q: Distribution,
    proposal: Optional[Distribution] = None,
    N=int(5e4),
) -> float:
    """This will evaluate the posteriors quality by investingating its importance
    weights. If q is a perfect posterior approximation then q(theta) ~
    potential_function(theta). This we should be able to fit a line to (q(theta),
    potential_function(theta)), whereas the slope will be the normalizing constant. The
    quality of a linear fit is thus a direct metric for the quality of q. We use R2
    statistic.

    Args:
        potential_function: Potential function of target.
        q: Variational distribution, should be proportional to the potential_function
        proposal: Proposal for samples. Typically this is q.
        N: Number of samples involved in the test.

    Returns:
        float: Quality metric

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
    msg="\t Good: Smaller than 0.5  Bad: Larger than 1.0 \t NOTE: Less sensitive to mode collapse.",
)
def psis_q(posterior):
    basic_checks(posterior)
    return psis_diagnostics(posterior.potential_fn, posterior.q)


@register_quality_metric(
    name="psis_prior",
    msg="\t Good: Smaller than 0.5  Bad: Larger than 2.0 \t NOTE: More sensitive to mode collapse.",
)
def psis_prior(posterior):
    basic_checks(posterior)
    return psis_diagnostics(
        posterior.potential_fn, posterior.q, proposal=posterior.prior, N=int(1e7)
    )


@register_quality_metric(
    name="proportionality",
    msg="\t Good: Larger than 0.5, best is 1.0  Bad: Smaller than 0.5 \t NOTE: Less sensitive to mode collapse.",
)
def proportionality(posterior):
    basic_checks(posterior)
    proportional_to_joint_diagnostics(posterior.potential_function, posterior.q)
