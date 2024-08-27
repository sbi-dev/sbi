# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from typing import Callable, Type

import torch
from torch import Tensor

from sbi.inference.potentials.score_based_potential import (
    PosteriorScoreBasedPotential,
)

PREDICTORS = {}


def get_predictor(
    name: str, score_based_potential: PosteriorScoreBasedPotential, **kwargs
) -> "Predictor":
    """Helper function to get predictor by name.

    Args:
        name: Name of the predictor.
        score_based_potential: Score-based potential to initialize the predictor.
    """
    return PREDICTORS[name](score_based_potential, **kwargs)


def register_predictor(name: str) -> Callable:
    """Register a predictor.

    Args:
        name (str): Name of the predictor.

    Returns:
        Callable: Decorator for registering the predictor.
    """

    def decorator(predictor: Type[Predictor]) -> Callable:
        assert issubclass(
            predictor, Predictor
        ), "Predictor must be a subclass of Predictor."
        PREDICTORS[name] = predictor
        return predictor

    return decorator


class Predictor(ABC):
    """Predictor base class.

    See child classes for more detail.
    """

    def __init__(
        self,
        potential_fn: PosteriorScoreBasedPotential,
    ):
        """Initialize predictor.

        Args:
            potential_fn: potential with gradient from which to sample.
        """
        self.potential_fn = potential_fn
        self.device = potential_fn.device

        # Extract relevant functions from the score function
        self.drift = self.potential_fn.score_estimator.drift_fn
        self.diffusion = self.potential_fn.score_estimator.diffusion_fn

    def __call__(self, theta: Tensor, t1: Tensor, t0: Tensor) -> Tensor:
        """Run prediction.

        Args:
            theta: Parameters.
            t1: Time.
            t0: Time.
        """
        return self.predict(theta, t1, t0)

    @abstractmethod
    def predict(self, theta: Tensor, t1: Tensor, t0: Tensor) -> Tensor:
        """Run prediction.

        Args:
            theta: Parameters.
            t1: Time.
            t0: Time.
        """
        pass


@register_predictor("euler_maruyama")
class EulerMaruyama(Predictor):
    def __init__(
        self,
        potential_fn: PosteriorScoreBasedPotential,
        eta: float = 1.0,
    ):
        """Simple Euler-Maruyama discretization of the associated family of reverse
        SDEs.

        Args:
            potential_fn: Score-based potential to predict.
            eta: Mediates how much noise is added during sampling i.e.
                for values approaching 0 this becomes the deterministic probabilifty
                flow ODE. For large values it becomes a more stochastic reverse SDE.
                Defaults to 1.0.
        """
        super().__init__(potential_fn)
        assert eta > 0, "eta must be positive."
        self.eta = eta

    def predict(self, theta: Tensor, t1: Tensor, t0: Tensor):
        dt = t1 - t0
        dt_sqrt = torch.sqrt(dt)
        f = self.drift(theta, t1)
        g = self.diffusion(theta, t1)
        score = self.potential_fn.gradient(theta, t1)
        f_backward = f - (1 + self.eta**2) / 2 * g**2 * score
        g_backward = self.eta * g
        return theta - f_backward * dt + g_backward * torch.randn_like(theta) * dt_sqrt
