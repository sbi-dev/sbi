# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import pytest
import torch
from torch import Tensor, eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import (
    ImportanceSamplingPosterior,
    MCMCPosterior,
    RejectionPosterior,
    VIPosterior,
)
from sbi.inference.potentials.base_potential import CallablePotentialWrapper
from sbi.utils import BoxUniform
from sbi.utils.conditional_density_utils import ConditionedPotential


@pytest.mark.parametrize(
    "sampling_method",
    [
        ImportanceSamplingPosterior,
        pytest.param(MCMCPosterior, marks=pytest.mark.mcmc),
        RejectionPosterior,
        VIPosterior,
    ],
)
def test_callable_potential(sampling_method, mcmc_params_accurate: dict):
    """Test whether callable potentials can be used to sample from a Gaussian."""
    dim = 2
    mean = 2.5
    cov = 2.0
    x_o = 1 * ones((dim,))
    target_density = MultivariateNormal(mean * ones((dim,)), cov * eye(dim))

    def potential(theta, x_o):
        return target_density.log_prob(theta + x_o)

    proposal = MultivariateNormal(zeros((dim,)), 5 * eye(dim))

    if sampling_method == ImportanceSamplingPosterior:
        approx_density = sampling_method(
            potential_fn=potential, proposal=proposal, method="sir"
        )
        approx_samples = approx_density.sample((1024,), oversampling_factor=1024, x=x_o)
    elif sampling_method == MCMCPosterior:
        approx_density = sampling_method(potential_fn=potential, proposal=proposal)
        approx_samples = approx_density.sample(
            (1024,), x=x_o, method="slice_np_vectorized", **mcmc_params_accurate
        )
    elif sampling_method == VIPosterior:
        approx_density = sampling_method(
            potential_fn=potential, prior=proposal
        ).set_default_x(x_o)
        approx_density = approx_density.train()
        approx_samples = approx_density.sample((1024,))
    elif sampling_method == RejectionPosterior:
        approx_density = sampling_method(
            potential_fn=potential, proposal=proposal
        ).set_default_x(x_o)
        approx_samples = approx_density.sample((1024,))

    sample_mean = torch.mean(approx_samples, dim=0)
    sample_std = torch.std(approx_samples, dim=0)
    assert torch.allclose(sample_mean, torch.as_tensor(mean) - x_o, atol=0.2)
    assert torch.allclose(sample_std, torch.sqrt(torch.as_tensor(cov)), atol=0.1)


@pytest.mark.parametrize(
    "condition",
    [
        torch.rand(1, 2),
        pytest.param(
            torch.rand(2, 2),
            marks=pytest.mark.xfail(
                raises=ValueError,
                match="Condition with batch size > 1 not supported",
            ),
        ),
    ],
)
def test_conditioned_potential(condition: Tensor):
    potential_fn = CallablePotentialWrapper(
        potential_fn=lambda theta, x_o: theta,
        prior=BoxUniform(low=zeros(2), high=ones(2)),
    )

    ConditionedPotential(potential_fn, condition=condition, dims_to_sample=[0])
