# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable

import torch
import torch.distributions.transforms as torch_tf
from torch import Tensor

from sbi.samplers.importance.sir import sampling_importance_resampling


class IterateParameters:
    """Iterates through parameters by rows"""

    def __init__(self, parameters: torch.Tensor, **kwargs):
        self.iter = self._make_iterator(parameters)

    @staticmethod
    def _make_iterator(t):
        for i in range(t.shape[0]):
            yield t[i, :].reshape(1, -1)

    def __call__(self):
        return next(self.iter)


def proposal_init(
    proposal: Any, transform: torch_tf.Transform, **kwargs: Any
) -> Tensor:
    """Return a sample from the proposal."""
    prior_samples = proposal.sample((1,)).detach()
    transformed_prior_samples = transform(prior_samples)
    return transformed_prior_samples  # type: ignore


def sir_init(
    proposal: Any,
    potential_fn: Callable,
    transform: torch_tf.Transform,
    num_candidate_samples: int = 10_000,
    **kwargs: Any,
) -> Tensor:
    r"""Return a sample obtained by sequential importance reweighting.

    See Rubin 1988, "Using the sir algorithm to simulate posterior distributions."

    Args:
        proposal: Proposal distribution, candidate samples are drawn from it.
        potential_fn: Potential function that the candidate samples are weighted with.
            Note that the function needs to return log probabilities.
        num_candidate_samples: Number of candidate samples per batch.

    Returns:
        A single sample.
    """
    sample = sampling_importance_resampling(
        potential_fn=potential_fn,
        proposal=proposal,
        num_samples=1,
        num_candidate_samples=num_candidate_samples,
        **kwargs,
    )
    return transform(sample)  # type: ignore


def resample_given_potential_fn(
    proposal: Any,
    potential_fn: Callable,
    transform: torch_tf.Transform,
    num_candidate_samples: int = 10_000,
    num_batches: int = 1,
    **kwargs: Any,
) -> Tensor:
    r"""Return a sample via resampling proposal samples with `potential_fn` weights.

    The difference to actually performing SIR is that the weights are given only
    by the `potential_fn`, whereas SIR corrects for the `proposal.log_prob()`.

    Up to `sbi` v0.18.0, this method was the default. As of `sbi` v0.19.0, the default
    is SIR (i.e., with correction).

    Args:
        proposal: Proposal distribution, candidate samples are drawn from it.
        potential_fn: Potential function that the candidate samples are weighted with.
            Note that the function needs to return log probabilities.
        num_batches: Number of batches drawn.
        num_candidate_samples: Number of candidate samples per batch.


    Returns:
        A single sample.
    """

    with torch.set_grad_enabled(False):
        log_weights = []
        init_param_candidates = []
        for _ in range(num_batches):
            batch_draws = proposal.sample((num_candidate_samples,)).detach()
            init_param_candidates.append(batch_draws)
            log_weights.append(potential_fn(batch_draws).detach())
        log_weights = torch.cat(log_weights)
        init_param_candidates = torch.cat(init_param_candidates)

        # Norm weights in log space
        log_weights -= torch.logsumexp(log_weights, dim=0)
        probs = torch.exp(log_weights.view(-1))
        probs[torch.isnan(probs)] = 0.0
        probs[torch.isinf(probs)] = 0.0
        probs /= probs.sum()

        idxs = torch.multinomial(probs, 1, replacement=False)
        # Return transformed sample.
        return transform(init_param_candidates[idxs, :])  # type: ignore
