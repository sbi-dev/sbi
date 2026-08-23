# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License, Version 2.0, see <https://www.apache.org/licenses/licenses/>

import torch

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.base_potential import BasePotential


class DummyPotential(BasePotential):
    def __call__(
        self, theta: torch.Tensor, track_gradients: bool = True
    ) -> torch.Tensor:
        return -theta.square().sum(dim=-1)


class DummyPosterior(NeuralPosterior):
    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def sample_batched(self, *args, **kwargs):
        raise NotImplementedError


def test_reinitializing_posterior_preserves_default_x():
    """A posterior rebuild must preserve its default observation."""
    posterior = DummyPosterior(DummyPotential(prior=None))
    posterior.set_default_x(torch.tensor([1.0, 2.0]))

    NeuralPosterior.__init__(posterior, DummyPotential(prior=None), device="cpu")

    assert torch.equal(posterior.default_x, torch.tensor([[1.0, 2.0]]))
