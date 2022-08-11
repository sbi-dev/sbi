from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import zeros

from sbi import utils as utils
from sbi.inference import (
    SNLE,
    SNPE,
    SNRE,
    MCMCPosterior,
    likelihood_estimator_based_potential,
    posterior_estimator_based_potential,
    prepare_for_sbi,
    ratio_estimator_based_potential,
    simulate_for_sbi,
)


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
        x = x.reshape(-1, 6 * 4 * 4)
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
        num_trials = 1
    elif method == SNLE:
        net_provider = utils.likelihood_nn(model="mdn", embedding_net=embedding())
        num_trials = 2
    else:
        net_provider = utils.classifier_nn(
            model="mlp",
            z_score_theta="structured",  # Test that structured z-scoring works.
            embedding_net_x=embedding(),
        )
        num_trials = 2

    if method == SNRE:
        inference = method(classifier=net_provider, show_progress_bars=False)
    else:
        inference = method(density_estimator=net_provider, show_progress_bars=False)
    theta, x = simulate_for_sbi(simulator, prior, num_simulations)
    estimator = inference.append_simulations(theta, x).train(
        training_batch_size=100, max_num_epochs=10
    )
    x_o = simulator(theta_o.repeat(num_trials, 1))

    if method == SNLE:
        potential_fn, theta_transform = likelihood_estimator_based_potential(
            estimator, prior, x_o
        )
    elif method == SNPE:
        potential_fn, theta_transform = posterior_estimator_based_potential(
            estimator, prior, x_o
        )
    elif method == SNRE:
        potential_fn, theta_transform = ratio_estimator_based_potential(
            estimator, prior, x_o
        )
    else:
        raise NotImplementedError

    posterior = MCMCPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        proposal=prior,
        method="slice_np_vectorized",
        num_chains=2,
    )

    posterior.potential(posterior.sample((num_samples,), show_progress_bars=False))
