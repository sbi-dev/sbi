from abc import abstractmethod

from torch import Tensor

from sbi.inference.potentials.score_based_potential import PosteriorScoreBasedPotential


class State:
    def __init__(self, input: Tensor, time: Tensor) -> None:
        self.input = input
        self.time = time


class Kernel:
    def __init__(self, score_fn: PosteriorScoreBasedPotential) -> None:
        self.score_fn = score_fn

    @abstractmethod
    def __call__(
        self, state: State, time: Tensor, track_gradients: bool = True
    ) -> Tensor:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"
