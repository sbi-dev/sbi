# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior

# Posteriors that can only evaluate the density up to a normalizing constant
# announce it with this warning when `log_prob()` is called. Keying on the
# warning rather than on a list of classes keeps the check next to the behaviour
# it describes, and it propagates through wrappers such as `EnsemblePosterior`,
# whose normalization depends on the components it holds.
UNNORMALIZED_LOG_PROB_WARNING = ".*log-probability is unnormalized.*"


def _log_prob_normalized(
    dist: Union[NeuralPosterior, Distribution],
    theta: Tensor,
    x_o: Optional[Tensor],
    name: str,
) -> Tensor:
    """Evaluate a log-density, refusing objects defined only up to a constant.

    Args:
        dist: Distribution or posterior to evaluate.
        theta: Parameters at which to evaluate.
        x_o: Observation to condition posteriors on. Ignored for torch
            distributions; if None, a posterior's `default_x` is used.
        name: Argument name used in the error message.

    Returns:
        Log-probabilities, flattened to shape `(len(theta),)`.

    Raises:
        NotImplementedError: If `dist` reports an unnormalized log-density.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=UNNORMALIZED_LOG_PROB_WARNING)
        try:
            log_prob = (
                dist.log_prob(theta)
                if isinstance(dist, Distribution)
                else dist.log_prob(theta, x=x_o)
            )
        except Warning as unnormalized:
            raise NotImplementedError(
                f"`{name}` is a `{type(dist).__name__}`, whose `log_prob()` is "
                "only defined up to a normalizing constant. The constants do not "
                "cancel in a KL divergence, so it cannot be estimated here. Use a "
                "sample-based divergence such as `sbi.utils.metrics.c2st` instead."
            ) from unnormalized
    return log_prob.reshape(-1)


def kl_divergence_mc(
    p: Union[NeuralPosterior, Distribution],
    q: Union[NeuralPosterior, Distribution],
    x_o: Optional[Tensor] = None,
    num_samples: int = 1000,
    p_samples: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Monte Carlo estimate of $D_{KL}(p \| q)$ using samples drawn from `p`.

    Computes $\frac{1}{N} \sum_i \log p(\theta_i) - \log q(\theta_i)$ with
    $\theta_i \sim p$. Both `p` and `q` must expose a *normalized* `log_prob()`,
    and `p` must expose `sample()`; posteriors that are only defined up to a
    normalizing constant raise a `NotImplementedError`.

    The estimator is unbiased but its variance grows with the divergence, so the
    returned standard error should be checked before interpreting small
    differences.

    Args:
        p: Distribution to sample from and evaluate in the numerator.
        q: Distribution to evaluate in the denominator.
        x_o: Observation to condition posteriors on. If None, a posterior's
            `default_x` is used. Ignored for torch distributions.
        num_samples: Number of Monte Carlo samples. Ignored if `p_samples` is
            given.
        p_samples: Pre-drawn samples from `p`. Reusing one sample set across
            several calls gives a paired (lower-variance) comparison and avoids
            repeated sampling. Note that the normalization of `p` can only be
            checked once it is evaluated, so when this is None the samples are
            drawn before that check; pass `p_samples` to avoid paying for
            sampling that is then discarded.

    Returns:
        Tuple of the KL estimate and its standard error, both scalar tensors.

    Raises:
        NotImplementedError: If `p` or `q` has no normalized `log_prob()`.
        ValueError: If any sample falls outside the support of `q`, which makes
            the divergence infinite.
    """
    if p_samples is None:
        p_samples = (
            p.sample((num_samples,))
            if isinstance(p, Distribution)
            else p.sample((num_samples,), x=x_o, show_progress_bars=False)
        )

    log_ratio = _log_prob_normalized(p, p_samples, x_o, "p") - _log_prob_normalized(
        q, p_samples, x_o, "q"
    )

    num_nonfinite = int((~torch.isfinite(log_ratio)).sum())
    if num_nonfinite > 0:
        raise ValueError(
            f"{num_nonfinite}/{len(log_ratio)} samples from `p` have non-finite "
            "log-ratios, so they most likely fall outside the support of `q` "
            "and the KL divergence is infinite. This typically happens when `q` "
            "is truncated or has bounded support that `p` exceeds."
        )

    estimate = log_ratio.mean()
    standard_error = log_ratio.std() / math.sqrt(log_ratio.numel())
    return estimate, standard_error

