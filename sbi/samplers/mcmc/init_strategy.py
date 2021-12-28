# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable

import numpy as np
from pyknos import nflows
import torch
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


def prior_init(prior: Any, transform: nflows.transforms, **kwargs: Any) -> Tensor:
    """Return a sample from the prior."""
    prior_samples = prior.sample((1,)).detach()
    transformed_prior_samples = transform(prior_samples)
    return transformed_prior_samples


def sir(
    prior: Any,
    potential_fn: Callable,
    transform: nflows.transforms,
    sir_num_batches: int = 10,
    sir_batch_size: int = 1000,
    **kwargs: Any,
) -> Tensor:
    r"""Return a sample obtained by sequential importance reweighting.

    This function can also do `SIR` on the conditional posterior
    $p(\theta_i|\theta_j, x)$ when a `condition` and `dims_to_sample` are passed.

    Args:
        prior: Prior distribution, candidate samples are drawn from it.
        potential_fn: Potential function that the candidate samples are weighted with.
            Note that the function needs to return log probabilities.
        sir_num_batches: Number of candidate batches drawn.
        sir_batch_size: Batch size used for evaluating candidates.

    Returns:
        A single sample.
    """
    init_strategy_num_candidates = sir_num_batches * sir_batch_size

    with torch.set_grad_enabled(False):
        log_weights = []
        init_param_candidates = []
        for i in range(sir_num_batches):
            batch_draws = prior.sample((sir_batch_size,)).detach()
            transformed_batch_draws = transform(batch_draws)
            init_param_candidates.append(transformed_batch_draws)
            log_weights.append(potential_fn(transformed_batch_draws.numpy()).detach())
        log_weights = torch.cat(log_weights)
        init_param_candidates = torch.cat(init_param_candidates)

        # Norm weights in log space
        log_weights -= torch.logsumexp(log_weights, dim=0)
        probs = np.exp(log_weights.view(-1).numpy().astype(np.float64))
        probs[np.isnan(probs)] = 0.0
        probs[np.isinf(probs)] = 0.0
        probs /= probs.sum()

        idxs = np.random.choice(
            a=np.arange(init_strategy_num_candidates),
            size=1,
            replace=False,
            p=probs,
        )

        return init_param_candidates[torch.from_numpy(idxs.astype(int)), :]
