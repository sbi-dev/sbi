import math
from abc import ABC, abstractmethod
from typing import Callable, Optional, Type

import torch
from torch import Tensor

from sbi.samplers.score.predictors import Predictor

CORRECTORS = {}


def get_corrector(name: str, predictor: Predictor, **kwargs) -> "Corrector":
    """Helper function to get corrector by name.

    Args:
        name: Name of the corrector.
        predictor: Predictor to initialize the corrector.

    Returns:
        Corrector: The corrector.
    """
    return CORRECTORS[name](predictor, **kwargs)


def register_corrector(name: str) -> Callable:
    """Register a corrector.

    Args:
        name (str): Name of the corrector.

    Returns:
        Callable: Decorator for registering the corrector.
    """

    def decorator(corrector: Type[Corrector]) -> Callable:
        assert issubclass(corrector, Corrector), (
            "Corrector must be a subclass of Corrector."
        )
        CORRECTORS[name] = corrector
        return corrector

    return decorator


class Corrector(ABC):
    def __init__(
        self,
        predictor: Predictor,
    ):
        """Base class for correctors.

        Predictor-corrector diffusion-based samplers for score-based sampling where
        originally introduced in [1]. The corrector is used to refine the samples at the
        current time, to match the target marginal distribution.

        Literature:
        - [1] Score-Based Generative Modeling through Stochastic Differential Equations
            (https://arxiv.org/abs/2011.13456)

        Args:
            predictor: The associated predictor.
        """
        self.predictor = predictor
        self.potential_fn = predictor.potential_fn
        self.device = predictor.device

    def __call__(
        self, theta: Tensor, t0: Tensor, t1: Optional[Tensor] = None
    ) -> Tensor:
        """Correct the samples.

        Args:
            theta: The samples to correct.
            t0: The current time.
            t1: The next time. Defaults to None.

        Returns:
            Tensor: The corrected samples.
        """
        return self.correct(theta, t0, t1)

    @abstractmethod
    def correct(self, theta: Tensor, t0: Tensor, t1: Optional[Tensor] = None) -> Tensor:
        """Correct the samples."""
        pass


@register_corrector("langevin")
class LangevinCorrector(Corrector):
    def __init__(
        self,
        predictor: Predictor,
        step_size: float = 1e-4,
        num_steps: int = 5,
    ):
        """Basic Langevin corrector.

        See [1] for more details on unadjusted Langevin dynamics for sampling. This was
        one of the samplers introduced in [2].

        Literature:
        - [1] https://en.wikipedia.org/wiki/Langevin_dynamics
        - [2] Score-Based Generative Modeling through Stochastic Differential Equations
            (https://arxiv.org/abs/2011.13456)

        Args:
            predictor: Associated predictor.
            step_size: Unadjusted Langevin dynamics are only valid for small
                step sizes. Defaults to 1e-4.
            num_steps: Number of steps to correct. Defaults to 5.
        """
        super().__init__(predictor)
        self.step_size = step_size
        self.std = math.sqrt(2 * self.step_size)
        self.num_steps = num_steps

    def correct(self, theta: Tensor, t0: Tensor, t1: Optional[Tensor] = None) -> Tensor:
        """Correct the samples using unadjusted Langevin dynamics.

        Does not explicitly depend on the current time.
        """
        for _ in range(self.num_steps):
            score = self.predictor.potential_fn.gradient(theta, t1)
            eps = self.std * torch.randn_like(theta, device=self.device)
            theta = theta + self.step_size * score + eps

        return theta


@register_corrector("gibbs")
class GibbsCorrector(Corrector):
    def __init__(self, predictor: Predictor, num_steps: int = 5):
        """(Pseudo) Gibbs sampling corrector.

        Iteratively adds back noise according to the correct forward SDE, then removes
        noise using the predictor. Hence, approximatly sampling form the joint
        distribution using Gibbs sampling (if the two conditional distributions are
        compatible).

        Args:
            predictor: Associated predictor.
            num_steps: Number of steps. Defaults to 5.
        """
        super().__init__(predictor)
        self.num_steps = num_steps

    def noise(self, theta: Tensor, t0: Tensor, t1: Tensor) -> Tensor:
        """Add noise according to the correct forward SDE"""
        # Forward sde
        f = self.predictor.drift(theta, t0)
        g = self.predictor.diffusion(theta, t0)
        eps = torch.randn_like(theta, device=self.device)
        dt = t1 - t0
        dt_sqrt = torch.sqrt(dt)
        return theta + f * dt + g * eps * dt_sqrt

    def correct(self, theta: Tensor, t0: Tensor, t1: Tensor) -> Tensor:
        """Correct the samples using Gibbs sampling."""
        for _ in range(self.num_steps):
            theta = self.noise(theta, t0, t1)
            theta = self.predictor(theta, t1, t0)
        return theta
