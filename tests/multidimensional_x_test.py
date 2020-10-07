from __future__ import annotations

from torch import zeros
import torch
import pytest
from sbi import utils as utils
from sbi.inference import SNPE, SNLE, SNRE, prepare_for_sbi
import torch.nn as nn
import torch.nn.functional as F


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
            SNRE,
            marks=pytest.mark.xfail(reason="SNLE cannot handle multiD x."),
        ),
        pytest.param(
            CNNEmbedding,
            SNLE,
            marks=pytest.mark.xfail(reason="SNRE cannot handle multiD x."),
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
    x_o = simulator(theta_o)

    if method == SNPE:
        net_provider = utils.posterior_nn(model="mdn", embedding_net=embedding(),)
    elif method == SNLE:
        net_provider = utils.likelihood_nn(model="mdn", embedding_net=embedding())
    else:
        net_provider = utils.classifier_nn(model="mlp", embedding_net_x=embedding(),)

    infer = method(simulator, prior, 1, 1, net_provider, show_progress_bars=False,)

    posterior = infer(
        num_simulations=num_simulations, training_batch_size=100, max_num_epochs=10
    ).set_default_x(x_o)

    posterior.log_prob(posterior.sample((num_samples,), show_progress_bars=False))
