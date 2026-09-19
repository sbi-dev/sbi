# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch

from sbi.utils.diagnostics_utils import get_posterior_samples_on_batch


class DummyPosterior:
    """Posterior returning zero samples with the requested shape."""

    parameter_dim = 2

    def sample(self, sample_shape, x, show_progress_bars=False):
        """Return zero samples for one observation.

        Args:
            sample_shape: Requested sample dimensions.
            x: Conditioning observation.
            show_progress_bars: Whether to show sampling progress.

        Returns:
            Zero samples with the requested sample dimensions.
        """
        return torch.zeros((*sample_shape, self.parameter_dim))

    def sample_batched(self, sample_shape, x, show_progress_bars=False):
        """Return zero samples for each observation in a batch.

        Args:
            sample_shape: Requested sample dimensions.
            x: Batch of conditioning observations.
            show_progress_bars: Whether to show sampling progress.

        Returns:
            Zero samples with sample dimensions before the batch dimension.
        """
        return torch.zeros((*sample_shape, len(x), self.parameter_dim))


@pytest.mark.parametrize("use_batched_sampling", [True, False])
@pytest.mark.parametrize("sample_shape", [(5,), (2, 3)])
def test_get_posterior_samples_preserves_sample_shape(
    use_batched_sampling, sample_shape
):
    """Return every sample dimension before the observation batch dimension."""
    xs = torch.zeros(4, 1)

    samples = get_posterior_samples_on_batch(
        xs=xs,
        posterior=DummyPosterior(),
        sample_shape=sample_shape,
        num_workers=1,
        use_batched_sampling=use_batched_sampling,
    )

    assert samples.shape == (*sample_shape, len(xs), DummyPosterior.parameter_dim)
