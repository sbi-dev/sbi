# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Callable, Optional, Union

import torch
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde


def get_1d_marginal_peaks_from_kde(
    samples: torch.Tensor,
    num_candidates: int = 10_000,
    num_workers: int = 1,
    bw_method: Optional[Union[str, float, Callable]] = "scott",
) -> torch.Tensor:
    """
    Returns estimated peaks for each dimension in samples separately.

    Beware, the method is approximate: it fits a 1D KDE to each column in samples
    evaluates the log-prob of a grid of candidate points (num_candidates) on the KDE
    and then returns the grid value of the max log prob as the peak.

    NOTE: This can be inaccurate especially when the 1D marginals are not unimodal, or
    if they have large variance.

    Args:
        samples: samples for which to estimate the peaks.
        num_candidates: number of grid samples for finding the peak.
        num_workers: number of CPU cores for parallelization, useful with
            high-dimensional samples and when using many candidates.
        bw_method: bandwidth method for KDE, can be 'scott' or 'silverman' for
            heuristics, float, or Callable (see scipy.stats.gaussian_kde).

    Returns:
        peaks for each column in samples.
    """

    assert samples.ndim == 2, "samples must have shape (num_samples, num_dim)."

    num_workers = min(num_workers, samples.shape[1])

    def get_max(s):
        # get KDE
        kde = gaussian_kde(s, bw_method=bw_method)
        # Sample candidates and get log probs
        candidates = torch.linspace(s.min(), s.max(), num_candidates)
        # candidates = kde.resample(num_candidates)
        scores = kde(candidates)
        # return value with max log prob.
        max_idx = torch.argmax(torch.tensor(scores))
        return candidates.flatten()[max_idx]

    # Optimize separately for each marginal.
    peaks = Parallel(n_jobs=num_workers)(delayed(get_max)(s) for s in samples.T)

    return torch.tensor(peaks, dtype=torch.float32)
