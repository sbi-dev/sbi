# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
import torch
from torch import Tensor

from sbi.inference.potentials.score_based_potential import (
    score_estimator_based_potential_gradient,
)
from sbi.neural_nets.score_nets import build_score_estimator
from sbi.samplers.score.score import Diffuser


@pytest.mark.parametrize(
    "sde_type",
    [
        "vp",
        "ve",
        "subvp",
    ],
)
@pytest.mark.parametrize("predictor", ("euler_maruyama", "ddim"))
@pytest.mark.parametrize("corrector", (None, "langevin", "gibbs"))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("mu", (-1.0, 0.0, 1.0))
@pytest.mark.parametrize("std", (1.0, 0.1))
def test_score_estimator_forward_shapes(
    sde_type, predictor, corrector, input_event_shape, mu, std
):
    mean0 = mu * torch.ones(input_event_shape)
    std0 = std * torch.ones(input_event_shape)

    score_fn = _build_gaussian_score_estimator(sde_type, input_event_shape, mean0, std0)

    sampler = Diffuser(score_fn, predictor, corrector)

    T_min = score_fn.score_estimator.T_min
    T_max = score_fn.score_estimator.T_max
    ts = torch.linspace(T_max, T_min, 500)
    samples = sampler.run(10_000, ts)

    mean_est = samples[0].mean(0)
    std_est = samples[0].std(0)

    assert torch.allclose(mean_est, mean0, atol=1e-1)
    assert torch.allclose(std_est, std0, atol=1e-1)


def _build_gaussian_score_estimator(
    sde_type: str,
    input_event_shape: Tuple[int],
    mean0: Tensor,
    std0: Tensor,
):
    """Helper function for all tests that deal with shapes of density estimators."""

    # Use discrete thetas such that categorical density esitmators can also use them.
    building_thetas = (
        torch.randn((1000, *input_event_shape), dtype=torch.float32) * std0 + mean0
    )
    building_xs = torch.ones((1000, 1))

    # Note the precondition predicts a correct Gaussian score by default if the neural
    # net predicts 0!
    class DummyNet(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], *input_event_shape))

    score_estimator = build_score_estimator(
        building_thetas,
        building_xs,
        sde_type=sde_type,
        score_net=DummyNet(),
    )

    score_fn, _ = score_estimator_based_potential_gradient(
        score_estimator, prior=None, x_o=torch.ones((1,))
    )

    return score_fn
