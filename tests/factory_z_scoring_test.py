# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Tests that posterior factories map z-scoring arguments to the intended variable."""

import pytest
import torch

from sbi.neural_nets.factory import posterior_flow_nn, posterior_score_nn

THETA_OFFSET = 100.0
X_OFFSET = -50.0


@pytest.mark.parametrize("factory", [posterior_flow_nn, posterior_score_nn])
def test_z_scoring_arguments_reach_their_own_variable(factory):
    """Check that z_score_theta governs theta and z_score_x governs the observation."""
    theta = THETA_OFFSET + torch.randn(200, 2)
    x = X_OFFSET + torch.randn(200, 3)

    theta_off = factory(z_score_theta="none", z_score_x="independent")(theta, x)
    assert torch.all(theta_off.mean_0 == 0)
    assert torch.all(theta_off.std_0 == 1)
    assert torch.allclose(
        theta_off.embedding_net(x).mean(0), torch.zeros(x.shape[1]), atol=1e-4
    )

    theta_on = factory(z_score_theta="independent", z_score_x="none")(theta, x)
    assert torch.allclose(theta_on.mean_0.flatten(), theta.mean(0), atol=1e-4)
    assert torch.allclose(theta_on.std_0.flatten(), theta.std(0), atol=1e-4)
    assert torch.allclose(theta_on.embedding_net(x).mean(0), x.mean(0), atol=1e-4)
