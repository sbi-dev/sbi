# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.inference import DirectPosterior
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior

# Objects whose `log_prob()` is a normalized density. Subclasses of `DirectPosterior`
# (e.g. `NPE_A_Posterior`) are included. `VectorFieldPosterior` is not: with a bounded
# prior, its `log_prob()` is not corrected for the mass outside the prior.
_NORMALIZED = (Distribution, DirectPosterior, VIPosterior)


def kl_divergence_mc(
    p: Union[NeuralPosterior, Distribution],
    q: Union[NeuralPosterior, Distribution],
    x: Optional[Tensor] = None,
    num_samples: int = 1000,
    p_samples: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Monte Carlo estimate of $D_{KL}(p \| q)$ from samples of `p`.

    Computes $\frac{1}{N} \sum_i \log p(\theta_i) - \log q(\theta_i)$ with
    $\theta_i \sim p$. Both `p` and `q` must have a normalized `log_prob()`. Torch
    distributions, `DirectPosterior` and `VIPosterior` are accepted; other posteriors
    are refused.

    In sequential inference, `kl_divergence_mc(posterior, proposal)` shows how much
    a round changed the estimate. See the how-to guide on sequential methods.

    Args:
        p: Distribution or posterior to sample from.
        q: Distribution or posterior to compare against.
        x: Observation to condition posteriors on. If None, their `default_x` is used.
            Ignored for torch distributions.
        num_samples: Number of samples drawn from `p`. Ignored if `p_samples` is
            given.
        p_samples: Samples from `p`. Reusing the same samples for several
            divergences gives a paired comparison.

    Returns:
        Estimate and its standard error, both scalar tensors. The standard error
        includes only the sampling noise. For a `DirectPosterior` with a bounded
        prior, `log_prob()` also contains an estimated leakage correction, whose
        noise is not included.

    Raises:
        NotImplementedError: If `p` or `q` has no normalized `log_prob()`.
        ValueError: If a torch distribution has a batch shape, if fewer than two
            samples are used, or if a sample of `p` is outside the support of `q`,
            which makes the divergence infinite.
    """
    for name, dist in (("p", p), ("q", q)):
        if not isinstance(dist, _NORMALIZED):
            raise NotImplementedError(
                f"`{name}` is a `{type(dist).__name__}`, whose `log_prob()` is not "
                "guaranteed to be a normalized density. Use a sample-based metric "
                "such as `sbi.utils.metrics.c2st` instead."
            )
        if isinstance(dist, Distribution) and dist.batch_shape:
            raise ValueError(
                f"`{name}` has batch shape {tuple(dist.batch_shape)}. Wrap it in "
                "`torch.distributions.Independent` so that it is one joint "
                "distribution."
            )

    if p_samples is None:
        p_samples = (
            p.sample((num_samples,))
            if isinstance(p, Distribution)
            else p.sample((num_samples,), x=x, show_progress_bars=False)
        )
    if len(p_samples) < 2:
        raise ValueError("At least two samples are needed for the standard error.")

    log_ratio = _log_prob(p, p_samples, x) - _log_prob(q, p_samples, x)

    num_nonfinite = int((~torch.isfinite(log_ratio)).sum())
    if num_nonfinite > 0:
        raise ValueError(
            f"{num_nonfinite}/{len(log_ratio)} samples of `p` have a non-finite "
            "log-ratio. Most likely they are outside the support of `q`, so the KL "
            "divergence is infinite."
        )

    return log_ratio.mean(), log_ratio.std() / math.sqrt(len(log_ratio))


def _log_prob(
    dist: Union[NeuralPosterior, Distribution], theta: Tensor, x: Optional[Tensor]
) -> Tensor:
    if isinstance(dist, Distribution):
        return dist.log_prob(theta).reshape(-1)
    return dist.log_prob(theta, x=x).reshape(-1)
