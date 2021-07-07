from __future__ import annotations

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import SNL, prepare_for_sbi, simulate_for_sbi
from sbi.simulators.linear_gaussian import (
    diagonal_linear_gaussian,
)
from sbi.vi.divergence_optimizers import (
    ElboOptimizer,
    IWElboOptimizer,
    RenjeyDivergenceOptimizer,
    TailAdaptivefDivergenceOptimizer,
    ForwardKLOptimizer,
)

OPTIMIZERS = [
    ElboOptimizer,
    IWElboOptimizer,
    RenjeyDivergenceOptimizer,
    TailAdaptivefDivergenceOptimizer,
    ForwardKLOptimizer,
]


@pytest.mark.slow
@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
def test_base_api(optimizer_type):
    """Test API for the ElboOptimizer"""
    num_dim = 10
    num_samples = 10
    FLOW_PARAS = [
        {"flow": "spline_autoregressive"},
        {"flow": "affine_autoregressive"},
        {"num_components": 2},
        {"num_components": 2, "rsample": True},
    ]
    # TODO Unbatched observation raises error...
    x_o = zeros(1,num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    simulator, prior = prepare_for_sbi(diagonal_linear_gaussian, prior)
    inference = SNL(prior, show_progress_bars=False,)

    theta, x = simulate_for_sbi(simulator, prior, 1000, simulation_batch_size=50)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=5)

    for para in FLOW_PARAS:
        # Not implemented for this loss
        if (
            para.get("num_components", 1) > 1
            and optimizer_type == TailAdaptivefDivergenceOptimizer
        ):
            continue

        posterior = inference.build_posterior(sample_with="vi",vi_parameters=para)
        
        posterior.set_default_x(x_o.reshape(1,-1))

        optimizer = optimizer_type(posterior)
        #optimizer.step(x_o)
        #assert optimizer.losses[0] != 1
        
        kwargs1 = {"gamma": 0.9, "lr": 0.5}
        optimizer.update(kwargs1)
        assert optimizer._optimizer.param_groups[0]["lr"] == 0.5
        assert optimizer._scheduler.gamma == 0.9

        kwargs2 = {"n_particles": 10, "clip_value": 0.5}
        optimizer.update(kwargs2)
        assert optimizer.n_particles == 10
        assert optimizer.clip_value == 0.5

        kwargs3 = {"n_particles": 1, "clip_value": 0.1, "gamma": 0.1}
        optimizer.update(kwargs3)
        assert optimizer.n_particles == 1
        assert optimizer.clip_value == 0.1
        assert optimizer._scheduler.gamma == 0.1

        assert not optimizer.converged()

