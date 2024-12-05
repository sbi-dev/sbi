import warnings
from typing import Optional

import torch
from joblib import Parallel, delayed
from torch import Tensor
from tqdm import tqdm

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.sbi_types import Shape


def get_posterior_samples_on_batch(
    xs: Tensor,
    posterior: NeuralPosterior,
    sample_shape: Shape,
    num_workers: int = 1,
    show_progress_bar: bool = False,
    use_batched_sampling: bool = True,
    batch_size: Optional[int] = None,
) -> Tensor:
    """Get posterior samples for a batch of xs.

    Args:
        xs: batch of observations.
        posterior: sbi posterior.
        num_samples: number of samples to draw from the posterior for each x.
        num_workers: number of workers to use for parallelization.
        show_progress_bars: whether to show progress bars.
        use_batched_sampling: whether to use batched sampling if possible.
        batch_size: batch size for batched sampling. Useful for batched sampling with
            large batches of xs for avoiding memory overflow.
    Returns:
        posterior_samples: of shape (num_samples, batch_size, dim_parameters).
    """
    num_xs = len(xs)
    if batch_size is None:
        batch_size = num_xs

    if use_batched_sampling:
        try:
            # distribute the batch of xs into smaller batches
            batched_xs = xs.split(batch_size)
            posterior_samples = torch.cat(
                [  # has shape (num_samples, num_xs, dim_parameters)
                    posterior.sample_batched(
                        sample_shape, x=xs_batch, show_progress_bars=show_progress_bar
                    )
                    for xs_batch in batched_xs
                ],
                dim=1,
            )
        except (NotImplementedError, AssertionError):
            warnings.warn(
                "Batched sampling not implemented for this posterior. "
                "Falling back to non-batched sampling.",
                stacklevel=2,
            )
            use_batched_sampling = False

    if not use_batched_sampling:
        # We need a function with extra training step for new x for VIPosterior.
        def sample_fun(
            posterior: NeuralPosterior, sample_shape: Shape, x: Tensor, seed: int = 0
        ) -> Tensor:
            if isinstance(posterior, VIPosterior):
                posterior.set_default_x(x)
                posterior.train()
            torch.manual_seed(seed)
            return posterior.sample(sample_shape, x=x, show_progress_bars=False)

        if isinstance(posterior, (VIPosterior, MCMCPosterior)):
            warnings.warn(
                "Using non-batched sampling. Depending on the number of different xs "
                f"( {num_xs}) and the number of parallel workers {num_workers}, "
                "this might take a lot of time.",
                stacklevel=2,
            )

        # Run in parallel with progress bar.
        seeds = torch.randint(0, 2**32, (num_xs,))
        outputs = list(
            tqdm(
                Parallel(return_as="generator", n_jobs=num_workers)(
                    delayed(sample_fun)(posterior, sample_shape, x=x, seed=s)
                    for x, s in zip(xs, seeds)
                ),
                disable=not show_progress_bar,
                total=len(xs),
                desc=f"Sampling {num_xs} times {sample_shape} posterior samples.",
            )
        )  # (batch_size, num_samples, dim_parameters)
        # Transpose to shape convention: (sample_shape, batch_size, dim_parameters)
        posterior_samples = torch.stack(
            outputs  # type: ignore
        ).permute(1, 0, 2)

    assert posterior_samples.shape[:2] == sample_shape + (
        num_xs,
    ), f"""Expected batched posterior samples of shape {sample_shape + (num_xs,)} got {
        posterior_samples.shape[:2]
    }."""
    return posterior_samples
