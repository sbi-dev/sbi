from typing import Any, Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm

from sbi.samplers.importance.importance_sampling import importance_sample


def sampling_importance_resampling(
    potential_fn: Callable,
    proposal: Any,
    num_samples: int = 1,
    num_candidate_samples: int = 32,
    max_sampling_batch_size: int = 10_000,
    show_progress_bars: bool = False,
    device: str = "cpu",
    **kwargs,
) -> Tensor:
    """Return samples obtained with sampling importance resampling (SIR).

    Args:
        potential_fn: Potential function $log(p(\theta))$ from which to draw samples.
        proposal: Proposal distribution for SIR.
        num_samples: Number of samples to draw.
        num_candidate_samples: Number of proposed samples from which only one is
            selected based on its importance weight.
        max_sampling_batch_size: The batchsize of samples being drawn from the
            proposal at every iteration.
        show_progress_bars: Whether to show a progress bar.
        device: Device on which to sample.

    Returns:
        Tensor: Samples of shape (num_samples, event_shape).
    """

    selected_samples = []

    max_sampling_batch_size = max(
        1, int(max_sampling_batch_size / num_candidate_samples)
    )
    sampling_batch_size = min(num_samples, max_sampling_batch_size)

    num_remaining = num_samples
    pbar = tqdm(
        disable=not show_progress_bars,
        total=num_samples,
        desc=f"Drawing {num_samples} posterior samples",
    )

    while num_remaining > 0:
        batch_size = min(sampling_batch_size, num_remaining)
        with torch.no_grad():
            thetas, log_weights = importance_sample(
                potential_fn=potential_fn,
                proposal=proposal,
                num_samples=batch_size * num_candidate_samples,
            )
            log_weights = log_weights.reshape(batch_size, num_candidate_samples)
            weights = log_weights.softmax(-1).cumsum(-1)
            uniform_decision = torch.rand(batch_size, 1, device=device)
            mask = torch.cumsum(weights >= uniform_decision, -1) == 1
            samples = thetas.reshape(batch_size, num_candidate_samples, -1)[mask]
            selected_samples.append(samples)

        num_remaining -= batch_size
        pbar.update(samples.shape[0])
    pbar.close()

    selected_samples = torch.cat(selected_samples)
    return selected_samples
