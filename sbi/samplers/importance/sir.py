# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import time
from typing import Callable, Optional

import torch
from torch import Tensor
from tqdm.auto import tqdm

from sbi.samplers.importance.importance_sampling import importance_sample
from sbi.sbi_types import Proposal
from sbi.utils.pbar import is_nested, nested_pbar_context, sampling_desc


def sampling_importance_resampling(
    potential_fn: Callable,
    proposal: Proposal,
    num_samples: int = 1,
    num_candidate_samples: int = 32,
    max_sampling_batch_size: int = 10_000,
    show_progress_bars: bool = False,
    device: str = "cpu",
    max_retries: int = 100,
    max_sampling_time: Optional[float] = None,
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
        max_retries: Maximum number of consecutive retry attempts without collecting
            any valid samples before raising a RuntimeError. Helps prevent infinite
            loops when candidate samples have non-finite importance weights (-inf).
        max_sampling_time: Optional maximum allowed sampling time (in seconds).
            If this time is exceeded, sampling is aborted and a RuntimeError
            is raised. Default is None.

    Returns:
        Tensor: Samples of shape (num_samples, event_shape).
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be at least 1, got {max_retries}.")
    if max_sampling_time is not None and max_sampling_time <= 0:
        raise ValueError(
            f"max_sampling_time must be positive, got {max_sampling_time}."
        )

    selected_samples = []

    sampling_batch_size = min(num_samples, max_sampling_batch_size)

    num_remaining = num_samples
    pbar = tqdm(
        disable=not show_progress_bars or is_nested(),
        total=num_samples,
        desc=sampling_desc(num_samples, "sir"),
    )

    try:
        start_time = time.time()
        retries = 0
        while num_remaining > 0:
            if (
                max_sampling_time is not None
                and (time.time() - start_time) > max_sampling_time
            ):
                num_collected = num_samples - num_remaining
                raise RuntimeError(
                    f"Sampling importance resampling (SIR) exceeded "
                    f"max_sampling_time of {max_sampling_time}s. "
                    f"Collected {num_collected}/{num_samples} samples. "
                    "This is likely due to extremely low acceptance or heavy leakage. "
                    "Consider increasing `oversampling_factor`, setting a higher "
                    "`max_sampling_time`, or checking proposal coverage."
                )

            batch_size = min(sampling_batch_size, num_remaining)
            with torch.no_grad():
                with nested_pbar_context():
                    thetas, log_weights = importance_sample(
                        potential_fn=potential_fn,
                        proposal=proposal,
                        num_samples=batch_size * num_candidate_samples,
                    )
                log_weights = log_weights.reshape(batch_size, num_candidate_samples)
                valid_groups = torch.isfinite(log_weights).any(dim=-1)

                if valid_groups.any():
                    # Substitute non-finite rows with 0.0 to prevent NaNs in softmax.
                    # Non-finite rows and invalid candidates are masked out below.
                    safe_log_weights = torch.where(
                        valid_groups.unsqueeze(-1),
                        log_weights,
                        torch.zeros_like(log_weights),
                    )
                    weights = safe_log_weights.softmax(-1).cumsum(-1)
                    eps = torch.finfo(log_weights.dtype).tiny
                    uniform_decision = torch.rand(
                        batch_size,
                        1,
                        device=log_weights.device,
                        dtype=log_weights.dtype,
                    ).clamp(min=eps)
                    mask = (
                        (torch.cumsum(weights >= uniform_decision, -1) == 1)
                        & valid_groups.unsqueeze(-1)
                        & torch.isfinite(log_weights)
                    )
                else:
                    mask = torch.zeros_like(log_weights, dtype=torch.bool)

                samples = thetas.reshape(batch_size, num_candidate_samples, -1)[mask]

                if samples.shape[0] == 0:
                    retries += 1
                    if retries >= max_retries:
                        num_collected = num_samples - num_remaining
                        raise RuntimeError(
                            "Sampling importance resampling (SIR) failed to produce "
                            f"valid samples after {max_retries} attempts with no "
                            "accepted candidates. Collected "
                            f"{num_collected}/{num_samples} samples. All candidate "
                            "proposals had non-finite log weights (-inf). This occurs "
                            "when all proposed candidates are rejected by the target "
                            "potential or prior support. Consider increasing "
                            "`oversampling_factor`, checking proposal coverage, or "
                            "verifying that the target region has non-zero probability "
                            "under the proposal."
                        )
                else:
                    retries = 0
                    selected_samples.append(samples)
                    num_remaining -= samples.shape[0]
                    pbar.update(samples.shape[0])
    finally:
        pbar.close()

    selected_samples = torch.cat(selected_samples)
    return selected_samples
