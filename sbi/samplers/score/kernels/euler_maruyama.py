from abc import abstractmethod

import torch
from torch import Tensor

from sbi.inference.potentials.score_based_potential import ScoreFunction
from sbi.samplers.score.kernels.base import Kernel, State


class EulerMaruyama(Kernel):
    def __init__(self, score_fn: ScoreFunction, eta=1.0) -> None:
        self.score_fn = score_fn
        self.drift_forward = score_fn.score_estimator.drift_fn
        self.diffusion_forward = score_fn.score_estimator.diffusion_fn
        self.eta = eta

    @abstractmethod
    def __call__(self, state: State, time: Tensor) -> Tensor:
        time_old = state.time
        input_old = state.input
        delta_t = time - time_old
        f = self.drift_forward(input_old, time_old)
        g = self.eta * self.diffusion_forward(input_old, time_old)
        score = self.score_fn(input_old, time_old)
        f_backward = f - (1 + self.eta**2) / 2 * g**2 * score

        new_input = (
            input_old
            + f_backward * delta_t
            + g * torch.randn_like(input_old) * torch.sqrt(delta_t)
        )

        return State(input=new_input, time=time)
