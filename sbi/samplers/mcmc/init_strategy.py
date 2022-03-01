# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable

import torch
import torch.distributions.transforms as torch_tf
from torch import Tensor


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


def sir(
    proposal: Any,
    potential_fn: Callable,
    transform: torch_tf.Transform,
    sir_num_batches: int = 10,
    sir_batch_size: int = 1000,
    **kwargs: Any,
) -> Tensor:
    r"""Return a sample obtained by sequential importance reweighting.

    See Rubin 1988, "Using the sir algorithm to simulate posterior distributions."

    This function can also do `SIR` on the conditional posterior
    $p(\theta_i|\theta_j, x)$ when a `condition` and `dims_to_sample` are passed.

    Args:
        proposal: Proposal distribution, candidate samples are drawn from it.
        potential_fn: Potential function that the candidate samples are weighted with.
            Note that the function needs to return log probabilities.
        sir_num_batches: Number of candidate batches drawn.
        sir_batch_size: Batch size used for evaluating candidates.

    Returns:
        A single sample.
    """

    with torch.set_grad_enabled(False):
        log_weights = []
        init_param_candidates = []
        for i in range(sir_num_batches):
            batch_draws = proposal.sample((sir_batch_size,)).detach()
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
