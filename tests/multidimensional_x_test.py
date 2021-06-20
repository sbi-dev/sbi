from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import zeros

from sbi import utils as utils
from sbi.inference import SNLE, SNPE, SNRE, prepare_for_sbi, simulate_for_sbi


# Minimal 2D simulator.
def simulator_2d(theta):
    x = torch.rand(theta.shape[0], 32, 32)
    return x


class CNNEmbedding(nn.Module):
    """Big CNN embedding net to levarage GPU computation."""

    def __init__(self):
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=6, kernel_size=5, padding=2)
        # Maxpool layer that reduces 32x32 image to 4x4
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8)
        # Fully connected layer taking as input the 6 flattened output arrays from the
        # maxpooling layer
        self.fc = nn.Linear(in_features=6 * 4 * 4, out_features=8)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 4 * 4)
        x = F.relu(self.fc(x))
        return x


@pytest.mark.parametrize(
    "embedding, method",
    (
        pytest.param(
            nn.Identity, SNPE, marks=pytest.mark.xfail(reason="Invalid embedding.")
        ),
        pytest.param(
            nn.Identity,
            SNLE,
            marks=pytest.mark.xfail(reason="SNLE cannot handle multiD x."),
        ),
        pytest.param(
            CNNEmbedding,
            SNRE,
        ),
        pytest.param(CNNEmbedding, SNPE),
    ),
)
def test_inference_with_2d_x(embedding, method):

    num_dim = 2
    num_samples = 10
    num_simulations = 100

    prior = utils.BoxUniform(zeros(num_dim), torch.ones(num_dim))

    simulator, prior = prepare_for_sbi(simulator_2d, prior)

    theta_o = torch.ones(1, num_dim)

    if method == SNPE:
        net_provider = utils.posterior_nn(
            model="mdn",
            embedding_net=embedding(),
        )
        sample_kwargs = {"sample_with_mcmc": True}
        num_trials = 1
    elif method == SNLE:
        net_provider = utils.likelihood_nn(model="mdn", embedding_net=embedding())
        sample_kwargs = {}
        num_trials = 2
    else:
        net_provider = utils.classifier_nn(
            model="mlp",
            embedding_net_x=embedding(),
        )
        sample_kwargs = {
            "mcmc_method": "slice_np_vectorized",
            "mcmc_parameters": {"num_chains": 2},
        }
        num_trials = 2

    inference = method(prior, net_provider, show_progress_bars=False)
    theta, x = simulate_for_sbi(simulator, prior, num_simulations)
    _ = inference.append_simulations(theta, x).train(
        training_batch_size=100, max_num_epochs=10
    )
    x_o = simulator(theta_o.repeat(num_trials, 1))
    posterior = inference.build_posterior(**sample_kwargs).set_default_x(x_o)

    posterior.log_prob(posterior.sample((num_samples,), show_progress_bars=False))
