# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch

from sbi.utils.diagnostics_utils import get_posterior_samples_on_batch


class DummyPosterior:
    """Posterior returning zeros that follow the documented shape convention."""

    parameter_dim = 2

    def sample(self, sample_shape, x, show_progress_bars=False):
        return torch.zeros((*sample_shape, self.parameter_dim))

    def sample_batched(self, sample_shape, x, show_progress_bars=False):
        return torch.zeros((*sample_shape, len(x), self.parameter_dim))


@pytest.mark.parametrize("use_batched_sampling", [True, False])
@pytest.mark.parametrize("sample_shape", [(5,), (2, 3)])
def test_get_posterior_samples_preserves_sample_shape(
    use_batched_sampling, sample_shape
):
    """Every sample dimension must come before the observation batch dimension."""
    xs = torch.zeros(4, 1)

    samples = get_posterior_samples_on_batch(
        xs, DummyPosterior(), sample_shape, use_batched_sampling=use_batched_sampling
    )

    assert samples.shape == (*sample_shape, len(xs), DummyPosterior.parameter_dim)
