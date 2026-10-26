# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from typing import Callable, Type

import torch
from torch import Tensor

PREDICTORS = {}


def get_predictor(
    name: str,
    vector_field_potential: 'VectorFieldBasedPotential',  # noqa: F821 # type: ignore
    **kwargs,
) -> "Predictor":
    """Helper function to get predictor by name.

    Args:
        name: Name of the predictor.
        vector_field_potential: Vector field potential to initialize the predictor.
    """
    return PREDICTORS[name](vector_field_potential, **kwargs)


def register_predictor(name: str) -> Callable:
    """Register a predictor.

    Args:
        name (str): Name of the predictor.

    Returns:
        Callable: Decorator for registering the predictor.
    """

    def decorator(predictor: Type[Predictor]) -> Callable:
        assert issubclass(predictor, Predictor), (
            "Predictor must be a subclass of Predictor."
        )
        PREDICTORS[name] = predictor
        return predictor

    return decorator


class Predictor(ABC):
    """Predictor base class.

    See child classes for more detail.
    """

    def __init__(
        self,
        potential_fn: 'VectorFieldBasedPotential',  # noqa: F821 # type: ignore
    ):
        """Initialize predictor.

        Args:
            potential_fn: potential with gradient from which to sample.
        """
        self.potential_fn = potential_fn
        self.device = potential_fn.device

        # Extract relevant functions from the score function
        self.drift = self.potential_fn.vector_field_estimator.drift_fn
        self.diffusion = self.potential_fn.vector_field_estimator.diffusion_fn

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
        potential_fn: 'VectorFieldBasedPotential ',  # noqa: F821 # type: ignore
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
        f = self.drift(theta, t1).to(dt.device)
        g = self.diffusion(theta, t1).to(dt.device)
        score = self.potential_fn.gradient(theta, t1).to(dt.device)
        f_backward = f - (1 + self.eta**2) / 2 * g**2 * score
        g_backward = self.eta * g
        return theta - f_backward * dt + g_backward * torch.randn_like(theta) * dt_sqrt


@register_predictor("heun")
class Heun(Predictor):
    def __init__(
        self,
        potential_fn: 'VectorFieldBasedPotential ',  # noqa: F821 # type: ignore
    ):
        """Heun's method, a second-order Runge-Kutta integrator for the
        probability flow ODE (the deterministic sampler of EDM [1]). It solves the
        same ODE as `sample_with="ode"`, but on the fixed time grid of `steps` or
        `ts`.

        Each step evaluates the network twice, so a grid of `steps` points costs
        `2 * (steps - 1)` evaluations. At equal evaluations, it is slightly worse
        than first-order Euler on the same ODE (`euler_maruyama` with a small
        `eta`) below about 64 evaluations and better from there on.

        [1] Karras et al., Elucidating the Design Space of Diffusion-Based
        Generative Models, NeurIPS 2022.

        Args:
            potential_fn: Score-based potential to predict.
        """
        super().__init__(potential_fn)

    def _probability_flow_drift(
        self, theta: Tensor, t: Tensor, device: torch.device
    ) -> Tensor:
        """Drift of the probability flow ODE, i.e. the Euler-Maruyama drift at
        eta = 0."""
        f = self.drift(theta, t).to(device)
        g = self.diffusion(theta, t).to(device)
        score = self.potential_fn.gradient(theta, t).to(device)
        return f - 0.5 * g**2 * score

    def predict(self, theta: Tensor, t1: Tensor, t0: Tensor):
        dt = t1 - t0
        k1 = self._probability_flow_drift(theta, t1, dt.device)
        theta_predicted = theta - k1 * dt
        if t0 == 0:
            # The VP and sub-VP scores diverge at t=0, so skip the correction and
            # keep the Euler step, which evaluates only at t1.
            return theta_predicted
        k2 = self._probability_flow_drift(theta_predicted, t0, dt.device)
        return theta - 0.5 * (k1 + k2) * dt
