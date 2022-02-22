import math
from typing import Callable, Optional

import torch

from sbi.types import Shape


# Mocking a neural posterior
class BiasedPosterior:
    def __init__(self, posterior: Callable, shift: float = 5.0):
        """give me a prior/posterior and I'll shift it by a scalar `shift`. All calls to `sample` are wrapped."""

        self.shift = shift
        self.posterior = posterior

    def set_default_x(self, x):

        self.posterior.set_default_x(x)

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[torch.Tensor] = None,
        show_progress_bars: bool = False,
    ):
        return (
            self.posterior.sample(
                sample_shape, x, show_progress_bars=show_progress_bars
            )
            + self.shift
        )


class DispersedPosterior:
    def __init__(self, posterior: Callable, dispersion: float = 1.05):
        """give me a posterior and I'll disperse it. All calls to `sample` are wrapped.
        This class exploits: Var(aX) = a**2 * Var(X) for any random variable X
        while retaining the expectation value E[X] of all samples.

        Parameters:
            posterior: posterior distribution modelled like NeuralPosterior
            dispersion: choose values <1. to make the variance smaller,
                choose values >1. to make the variance larger (distribution more wide)
        """

        self.dispersion = math.sqrt(dispersion)
        self.posterior = posterior

    def set_default_x(self, x):

        self.posterior.set_default_x(x)

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[torch.Tensor] = None,
        show_progress_bars: bool = False,
    ):

        value = self.posterior.sample(
            sample_shape, x, show_progress_bars=show_progress_bars
        )

        # obtain the median of all samples before applying
        # the dispersion to them (use median for more robust estimate)
        median = torch.median(value, dim=0)  # dim 0 is the batch dimension

        # disperse the samples
        dispersed = value * self.dispersion

        # obtain the new median after the dispersion
        median_ = torch.median(dispersed, dim=0)

        # shift to obtain the original expectation values
        # (we only want to disperse the samples, not offset)
        shift = median.values - median_.values

        return dispersed + shift
