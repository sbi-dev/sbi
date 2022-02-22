from typing import Callable, Optional

import torch
from torch import eye, ones
from torch.distributions import MultivariateNormal

from sbi.types import Shape


# Mocking a neural posterior
class BiasedPosterior:
    def __init__(self, posterior: Callable, prior: Callable, shift: float = 5.0):
        """give me a prior/posterior and I'll shift it, this class mimicks the case
        if prior == posterior
        """

        self.shift = shift
        self.prior = prior
        self.posterior = posterior

    def map(self):

        pass

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
    def __init__(
        self, posterior: Callable, prior: Callable, perc_dispersion: float = 0.05
    ):
        """give me a prior/posterior and I'll disperse it, this class mimicks the case
        if prior == posterior
        """

        self.dispersion = perc_dispersion
        self.prior = prior
        num_dim = self.prior.sample((1,)).shape[-1]
        loc_ = ones(num_dim)
        cov_ = eye(num_dim) * self.dispersion
        if self.dispersion > 0.0:
            self.dist = MultivariateNormal(
                loc=loc_, covariance_matrix=cov_, validate_args=False
            )
        else:
            self.dist = MultivariateNormal(
                loc=loc_ + self.dispersion,
                covariance_matrix=eye(num_dim) * self.dispersion * -1.0,
                validate_args=False,
            )
        self.posterior = posterior

    def map(self):

        pass

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
        dispersion = self.dist.sample(sample_shape)
        return value * dispersion
