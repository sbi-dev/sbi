import torch
from joblib import Parallel, delayed
from torch import Tensor
from tqdm import tqdm

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior


def get_posterior_samples_on_batch(
    xs: Tensor,
    posterior: NeuralPosterior,
    num_samples: int,
    num_workers: int = 1,
    show_progress_bar: bool = False,
) -> Tensor:
    """Get posterior samples for a batch of xs.

    Args:
        xs: batch of observations.
        posterior: sbi posterior.
        num_posterior_samples: number of samples to draw from the posterior in each sbc
            run.
        num_workers: number of workers to use for parallelization.
        show_progress_bars: whether to show progress bars.

    Returns:
        posterior_samples: of shape (num_samples, batch_size, dim_parameters).
    """

    # TODO: Use batched sampling when implemented.
    # try sample_batched except, NotImplementedError.
    batch_size = len(xs)

    # We need a function with extra training step for new x for VIPosterior.
    def sample_fun(posterior: NeuralPosterior, sample_shape, x: Tensor) -> Tensor:
        if isinstance(posterior, VIPosterior):
            posterior.set_default_x(x)
            posterior.train()
        return posterior.sample(sample_shape, x=x, show_progress_bars=False)

    # Run in parallel with progress bar.
    outputs = list(
        tqdm(
            Parallel(return_as="generator", n_jobs=num_workers)(
                delayed(sample_fun)(posterior, (num_samples,), x=x) for x in xs
            ),
            disable=not show_progress_bar,
            total=len(xs),
            desc=f"""Sampling {batch_size} times {num_samples}
                    posterior samples using {num_workers} workers.""",
        )
    )
    # Transpose to sample_batched shape convention:
    posterior_samples = torch.stack(outputs).transpose(0, 1)  # type: ignore

    assert posterior_samples.shape[:2] == (
        num_samples,
        batch_size,
    ), f"""Expected batched posterior samples of shape {(num_samples, batch_size)} got {
        posterior_samples.shape[:2]
    }."""
    return posterior_samples
