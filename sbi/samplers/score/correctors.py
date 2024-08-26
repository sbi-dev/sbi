from abc import ABC, abstractmethod
from typing import Callable, Optional, Type

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
        assert issubclass(
            corrector, Corrector
        ), "Corrector must be a subclass of Corrector."
        CORRECTORS[name] = corrector
        return corrector

    return decorator


class Corrector(ABC):
    def __init__(
        self,
        predictor: Predictor,
    ):
        """Base class for correctors.

        Args:
            predictor (Predictor): The associated predictor.
        """
        self.predictor = predictor
        self.potential_fn = predictor.potential_fn
        self.device = predictor.device

    def __call__(
        self, theta: Tensor, t0: Tensor, t1: Optional[Tensor] = None
    ) -> Tensor:
        return self.correct(theta, t0, t1)

    @abstractmethod
    def correct(self, theta: Tensor, t0: Tensor, t1: Optional[Tensor] = None) -> Tensor:
        pass
