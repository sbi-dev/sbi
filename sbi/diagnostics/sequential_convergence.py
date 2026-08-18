# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

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
        except UserWarning as unnormalized:
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
            "log-ratios, i.e. they fall outside the support of `q`. The KL "
            "divergence is infinite. This typically happens when `q` is "
            "truncated or has bounded support that `p` exceeds."
        )

    estimate = log_ratio.mean()
    standard_error = log_ratio.std() / torch.sqrt(
        torch.tensor(float(log_ratio.numel()))
    )
    return estimate, standard_error


class SequentialConvergenceTracker:
    r"""Tracks how a posterior estimate evolves across rounds of sequential inference.

    Call `update()` once per round with the round's posterior. Each call records
    three quantities at the observation `x_o`:

    - **compression**, $D_{KL}(q_r \| \pi)$: how far the current estimate has
      travelled from the prior.
    - **increment**, $D_{KL}(q_r \| q_{r-1})$: how much the last round moved the
      estimate. Undefined in round 0.
    - **ratio**, increment / compression: the fraction of the distance from the
      prior contributed by the last round. Being dimensionless, it is comparable
      across problems, whereas an absolute threshold on the compression is not:
      the compression is bounded above by $D_{KL}(p(\theta \mid x_o) \| \pi)$,
      the information the observation carries, which is a fixed property of the
      prior, simulator and observation that no number of rounds can raise.

    If the compression is not significantly greater than zero, the estimate is
    still indistinguishable from the prior; `update()` then flags the round as
    `uninformative` and reports the ratio as NaN.

    **This is a diagnostic, not a stopping rule.** All three quantities measure
    the self-consistency of the sequence of estimates, not the distance to the
    true posterior. A sequence that stabilizes on a wrong answer -- for instance
    when the first round concentrates in the wrong region and later rounds
    reinforce it -- produces the same numbers as one that has converged. To
    check correctness, use `sbi.diagnostics.run_sbc`, `run_tarp` or `LC2ST`.

    Example:
        ```python
        tracker = SequentialConvergenceTracker(prior, x_o)
        proposal = prior
        for _ in range(num_rounds):
            theta = proposal.sample((num_sims,))
            x = simulator(theta)
            inference.append_simulations(theta, x, proposal=proposal).train()
            posterior = inference.build_posterior().set_default_x(x_o)
            print(tracker.update(posterior))
            proposal = posterior
        ```

    Attributes:
        history: List of the dictionaries returned by `update()`, one per round.
    """

    def __init__(
        self,
        prior: Distribution,
        x_o: Tensor,
        num_samples: int = 1000,
        significance_z: float = 2.0,
    ) -> None:
        """
        Args:
            prior: The prior the sequential scheme started from.
            x_o: The observation the posterior is conditioned on.
            num_samples: Number of Monte Carlo samples drawn per round. Drawn
                once and reused for both divergences.
            significance_z: How many standard errors the compression must exceed
                zero by before a round counts as informative.
        """
        self.prior = prior
        self.x_o = x_o
        self.num_samples = num_samples
        self.significance_z = significance_z

        self.history: List[Dict[str, Any]] = []
        self._previous: Optional[Union[NeuralPosterior, Distribution]] = None

    def update(self, posterior: Union[NeuralPosterior, Distribution]) -> Dict[str, Any]:
        """Record the diagnostics for one round.

        The posterior is stored by reference for the next round's comparison.
        `build_posterior()` already returns an estimator that is decoupled from
        the trainer's network, so retraining does not alter it.

        Args:
            posterior: The posterior estimate of the current round, conditioned
                on `x_o` (or with `x_o` set as its `default_x`).

        Returns:
            Dictionary with keys `round`, `compression`, `compression_sem`,
            `increment`, `increment_sem`, `ratio` and `uninformative`. The
            increment entries are None in round 0.

        Raises:
            NotImplementedError: If `posterior` has no normalized `log_prob()`.
        """
        # Probe once on a cheap prior sample, so that an unsupported posterior
        # fails before any expensive sampling (e.g. MCMC chains) happens.
        _log_prob_normalized(posterior, self.prior.sample((1,)), self.x_o, "posterior")

        # One sample set for both divergences: paired, and one `sample()` call.
        samples = (
            posterior.sample((self.num_samples,))
            if isinstance(posterior, Distribution)
            else posterior.sample(
                (self.num_samples,), x=self.x_o, show_progress_bars=False
            )
        )

        compression, compression_sem = kl_divergence_mc(
            posterior, self.prior, x_o=self.x_o, p_samples=samples
        )
        uninformative = bool(compression <= self.significance_z * compression_sem)

        increment: Optional[float] = None
        increment_sem: Optional[float] = None
        ratio: Optional[float] = None
        if self._previous is not None:
            increment_t, increment_sem_t = kl_divergence_mc(
                posterior, self._previous, x_o=self.x_o, p_samples=samples
            )
            increment = float(increment_t)
            increment_sem = float(increment_sem_t)
            # The ratio is meaningless if the denominator is within noise of 0.
            ratio = float("nan") if uninformative else increment / float(compression)

        record: Dict[str, Any] = {
            "round": len(self.history),
            "compression": float(compression),
            "compression_sem": float(compression_sem),
            "increment": increment,
            "increment_sem": increment_sem,
            "ratio": ratio,
            "uninformative": uninformative,
        }
        self.history.append(record)
        self._previous = posterior
        return record

    @property
    def compressions(self) -> List[float]:
        """Compression per round, in round order."""
        return [record["compression"] for record in self.history]

    @property
    def increments(self) -> List[Optional[float]]:
        """Increment per round, in round order. None for round 0."""
        return [record["increment"] for record in self.history]

    @property
    def ratios(self) -> List[Optional[float]]:
        """Ratio of increment to compression per round. None for round 0."""
        return [record["ratio"] for record in self.history]
