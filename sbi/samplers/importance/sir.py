from typing import Any, Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm

from sbi.samplers.importance.importance_sampling import importance_sample


def sampling_importance_resampling(
    potential_fn: Callable,
    proposal: Any,
    num_samples: int = 1,
    oversampling_factor: int = 32,
    max_sampling_batch_size: int = 10_000,
    show_progress_bars: bool = False,
    **kwargs,
) -> Tensor:
    """Perform sampling importance resampling (SIR).

    Args:
        num_samples: Number of samples to draw.
        potential_fn: Potential function, this may be used to debias the proposal.
        proposal: Proposal distribution to propose samples.
        oversampling_factor: Number of proposed samples form which only one is
            selected based on its importance weight.
        num_samples_batch: Number of samples processed in parallel. For large K you may
            want to reduce this, depending on your memory capabilities.

    Returns:
        Tensor: Samples of shape (num_samples, event_shape).
    """

    selected_samples = []
    max_sampling_batch_size = int(max_sampling_batch_size / oversampling_factor)
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
                num_samples=batch_size * oversampling_factor,
            )
            log_weights = log_weights.reshape(batch_size, oversampling_factor)
            weights = log_weights.softmax(-1).cumsum(-1)
            uniform_decision = torch.rand(batch_size, 1, device=thetas.device)
            mask = torch.cumsum(weights >= uniform_decision, -1) == 1
            samples = thetas.reshape(batch_size, oversampling_factor, -1)[mask]
            selected_samples.append(samples)

        num_remaining -= batch_size
        pbar.update(samples.shape[0])
    pbar.close()

    selected_samples = torch.cat(selected_samples)
    return selected_samples
