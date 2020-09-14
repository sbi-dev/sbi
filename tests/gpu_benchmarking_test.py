# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

# This provides a test for gpu performance. A simulator of images is defined so that
# preprocessing with an embedding net becomes necessary. The test shows that only when
# using a CNN embedding net a substantial speed up of training can be achieved by using
# the GPU. For comparison a tall linear network is used as embedding net - without
# speed up.

from __future__ import annotations

import pytest
from torch import zeros
import time
import numpy as np
import torch
from sbi import utils as utils
from sbi.inference import SNPE, SNLE, SNRE, prepare_for_sbi
import torch.nn as nn
import torch.nn.functional as F
from sbi.simulators import simulate_in_batches
from sbi.utils.torchutils import process_device


# Simulator of 32x32 images.
def simulator_model(params, return_points=False):
    """ Simulator model with two-dimensional input parameter and 1024-dimensional output

    This simulator serves as a basic example for using a neural net for learning
    summary features.
    It has only two input parameters but generates high-dimensional output vectors.
    The data is generated as follows:
        (-) Input:  parameter = [r, theta]
        (1) Generate 100 two-dimensional points centered around (r cos(theta),r sin
            (theta)) and perturb by a Gaussian noise with variance 0.01
        (2) Create a grayscale image of the scattered points with dimensions 32 by 32
        (3) Perturb the image with an uniform noise with values betweeen 0 and 0.2
    Parameters
    ----------
    parameter : array-like, shape (2)
        The two input parameters of the model, ordered as [r, theta]
    return_points : bool (default: False)
        Whether the simulator should return the coordinates of the simulated data
        points as well

    Returns
    -------
    image: torch tensor, shape (1, 1024)
        Output flattened image
    (optional) points: array-like, shape (100, 2)
        Coordinates of the 2D simulated data points
    """
    r, theta = params

    sigma_points = 0.10
    npoints = 100
    nx = 32
    ny = 32
    sigma_image = 0.20

    points = []

    # Generate points according to params and add noise.
    points = torch.tensor(
        [[r * torch.cos(theta), r * torch.sin(theta)]]
    ) + sigma_points * torch.randn(npoints, 2)

    # Find indices of points within unit circle in image.
    image = zeros((nx, ny))
    for point in points:
        pi = int((point[0] - (-1)) / ((+1) - (-1)) * nx)
        pj = int((point[1] - (-1)) / ((+1) - (-1)) * ny)
        if (pi < nx) and (pj < ny):
            image[pi, pj] = 1
    # Add uniform noise.
    image += sigma_image * torch.rand(nx, ny)
    image = image.T
    image = image.reshape(1, -1)

    if return_points:
        return image, points
    else:
        return image


class CNNEmbedding(nn.Module):
    """Big CNN embedding net to levarage GPU computation."""

    def __init__(self):
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=6, kernel_size=5, padding=2)
        # Maxpool layer that reduces 32x32 image to 4x4
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8)
        # Fully connected layer taking as input the 6 flattened output arrays from the
        # maxpooling layer
        self.fc = nn.Linear(in_features=6 * 4 * 4, out_features=8)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 6 * 4 * 4)
        x = F.relu(self.fc(x))
        return x


@pytest.mark.slow
@pytest.mark.requires_cuda
@pytest.mark.parametrize(
    "method, model",
    [(SNPE, "mdn"), (SNPE, "maf"), (SNLE, "nsf"), (SNRE, "mlp"), (SNRE, "resnet")],
)
def test_gpu_training(method, model):

    num_dim = 2
    num_samples = 10
    num_simulations = 500
    num_workers = 10
    max_num_epochs = 5

    prior = utils.BoxUniform(zeros(num_dim), torch.tensor([1.0, 2.0 * np.pi]))

    simulator, prior = prepare_for_sbi(simulator_model, prior)

    theta_o = torch.tensor([[0.7, np.pi / 4.0]])
    x_o = simulator(theta_o)

    # Pre simulate to have same training data for cpu and gpu.
    thetas = prior.sample((num_simulations,))
    xos = simulate_in_batches(
        simulator, thetas, num_workers=num_workers, show_progress_bars=False
    )

    if method == SNPE:
        kwargs = dict(
            density_estimator=utils.posterior_nn(
                model=model,
                # Needed to avoid doubles for some testing scenarios.
                embedding_net=CNNEmbedding(),
                hidden_features=40,
                num_transforms=2,
            ),
            sample_with_mcmc=True,
            mcmc_method="slice_np",
        )
    elif method == SNLE:
        kwargs = dict(
            density_estimator=utils.likelihood_nn(
                model=model,
                # Needed to avoid doubles for some testing scenarios.
                hidden_features=40,
                num_transforms=2,
            ),
            mcmc_method="slice",
        )
    elif method == SNRE:
        kwargs = dict(
            classifier=utils.classifier_nn(
                model=model,
                # Needed to avoid doubles for some testing scenarios.
                embedding_net_x=CNNEmbedding(),
                hidden_features=40,
            ),
            mcmc_method="slice",
        )
    else:
        raise ValueError()

    # Record cpu and gpu runtime during training
    training_times = []
    for device in ["cpu", "cuda:0"]:
        infer = method(
            simulator, prior, show_progress_bars=False, device=device, **kwargs,
        )

        infer.provide_presimulated(thetas, xos)

        tic = time.time()
        posterior = infer(
            num_simulations=0, training_batch_size=100, max_num_epochs=max_num_epochs
        ).set_default_x(x_o)
        toc = time.time() - tic
        print(f"{device} training time: {toc:.2f}")
        training_times.append(toc)

        tic = time.time()
        samples = posterior.sample((num_samples,), show_progress_bars=False)
        print(f"{device} sampling time: {time.time() - tic:.2f}")

    assert (
        training_times[0] > training_times[1]
    ), "For CNN embedding GPU must be faster than CPU."


@pytest.mark.parametrize("device", ["cpu", "gpu", "cuda", "cuda:0", "cuda:42"])
def test_process_device(device: str):
    process_device(device)
