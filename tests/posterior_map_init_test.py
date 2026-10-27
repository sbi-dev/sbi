# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch

from sbi.inference import NPE
from sbi.utils import BoxUniform


@pytest.fixture(scope="module")
def direct_posterior():
    """A trained NPE (DirectPosterior), which has no proposal."""
    prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
    theta = prior.sample((200,))
    x = theta + 0.05 * torch.randn_like(theta)
    inference = NPE(prior=prior)
    inference.append_simulations(theta, x).train(max_num_epochs=2)
    return inference.build_posterior().set_default_x(torch.zeros(2))


def test_map_init_method_posterior(direct_posterior):
    """init_method='posterior' works."""
    theta_map = direct_posterior.map(
        init_method="posterior",
        num_init_samples=10,
        num_iter=2,
        show_progress_bars=False,
        force_update=True,
    )
    assert theta_map.shape[-1] == 2


def test_map_init_method_tensor(direct_posterior):
    """A tensor of initial locations is accepted (e.g. prior samples)."""
    inits = direct_posterior.prior.sample((10,))
    theta_map = direct_posterior.map(
        init_method=inits,
        num_init_samples=10,
        num_iter=2,
        show_progress_bars=False,
        force_update=True,
    )
    assert theta_map.shape[-1] == 2


def test_map_init_method_proposal_unavailable(direct_posterior):
    """init_method='proposal' fails informatively when there is no proposal."""
    with pytest.raises(ValueError, match="no proposal"):
        direct_posterior.map(
            init_method="proposal",
            num_init_samples=10,
            num_iter=2,
            show_progress_bars=False,
            force_update=True,
        )


def test_map_init_method_invalid(direct_posterior):
    """An unknown init_method raises an informative ValueError."""
    with pytest.raises(ValueError, match="Invalid init_method"):
        direct_posterior.map(
            init_method="not_a_method",
            num_init_samples=10,
            num_iter=2,
            show_progress_bars=False,
            force_update=True,
        )
