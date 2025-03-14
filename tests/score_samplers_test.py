# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
import torch
from torch import Tensor
from torch.distributions import Gamma, Independent, MultivariateNormal, Normal, Uniform

from sbi.inference.potentials.score_based_potential import (
    score_estimator_based_potential,
)
from sbi.neural_nets.net_builders import build_score_estimator
from sbi.samplers.score import Diffuser
from sbi.utils import BoxUniform, MultipleIndependent


@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize(
    "iid_method",
    [
        "fnpe",
        "gauss",
        "auto_gauss",
        "jac_gauss",
    ],
)
@pytest.mark.parametrize("num_dim", [1, 2, 3])
def test_score_fn_iid_on_different_priors(sde_type, iid_method, num_dim):
    """Test the the iid methods work with the most common priors that are used in
    practice (or are implemented in this library).

    This is mostly becuase "gauss" based methods do need to perform analytical
    integration over the prior (which is automatically done in this library). It ensures
    that it doesn't lead to any errors, but does not test the correctness of the
    integration!
    """
    mean0 = torch.zeros(num_dim)
    std0 = torch.ones(num_dim)
    score_fn = _build_gaussian_score_estimator(sde_type, (num_dim,), mean0, std0)
    # Diag normal prior
    prior1 = Independent(Normal(torch.zeros(num_dim), torch.ones(num_dim)), 1)
    # Uniform prior
    prior2 = Independent(Uniform(torch.zeros(num_dim), torch.ones(num_dim)), 1)
    prior2_2 = BoxUniform(torch.zeros(num_dim), torch.ones(num_dim))
    # Multivariate normal prior
    prior3 = MultivariateNormal(torch.zeros(num_dim), torch.eye(num_dim))
    # Gamma prior - analytical not implemented but should fall back to general case
    prior4 = Independent(Gamma(torch.ones(num_dim), torch.ones(num_dim)), 1)
    # Multiple independent prior
    if num_dim == 1:
        prior5 = Independent(Normal(torch.zeros(1), torch.ones(1)), 1)
    else:
        prior5 = MultipleIndependent([
            Normal(torch.zeros(1), torch.ones(1)) for _ in range(num_dim)
        ])

    priors = [prior1, prior2, prior2_2, prior3, prior4, prior5]
    x_o_iid = torch.ones((5, 1))
    score_fn.set_x(x_o_iid, x_is_iid=True, iid_method=iid_method)
    inputs = torch.ones((1, 1, num_dim))
    for prior in priors:
        time = torch.ones(1)
        score_fn.prior = prior
        output = score_fn.gradient(inputs, time=time)

        assert output.shape == (1, 1, num_dim), (
            f"Expected shape {(1, 1, num_dim)}, got {output.shape}"
        )
        assert torch.isfinite(output).all(), "Output contains non-finite values"


@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize("predictor", ("euler_maruyama",))
@pytest.mark.parametrize("corrector", (None, "gibbs", "langevin"))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("mu", (-1.0, 0.0, 1.0))
@pytest.mark.parametrize("std", (1.0, 0.1))
def test_gaussian_score_sampling(
    sde_type, predictor, corrector, input_event_shape, mu, std
):
    mean0 = mu * torch.ones(input_event_shape)
    std0 = std * torch.ones(input_event_shape)

    score_fn = _build_gaussian_score_estimator(sde_type, input_event_shape, mean0, std0)

    sampler = Diffuser(score_fn, predictor, corrector)

    t_min = score_fn.score_estimator.t_min
    t_max = score_fn.score_estimator.t_max
    ts = torch.linspace(t_max, t_min, 500)
    samples = sampler.run(1_000, ts)

    mean_est = samples.mean(0)
    std_est = samples.std(0)

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
        def __init__(self):
            super().__init__()
            self.dummy_param_for_device_detection = torch.nn.Linear(1, 1)

        def forward(self, input, condition, time):
            return torch.zeros_like(input)

    score_estimator = build_score_estimator(
        building_thetas,
        building_xs,
        sde_type=sde_type,
        score_net=DummyNet(),
    )

    score_fn, _ = score_estimator_based_potential(
        score_estimator, prior=None, x_o=torch.ones((1,))
    )

    return score_fn
