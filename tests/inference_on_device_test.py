# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
import torch
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import SNLE, SNPE_A, SNPE_C, SNRE_A, SNRE_B, simulate_for_sbi
from sbi.simulators import linear_gaussian
from sbi.utils.torchutils import BoxUniform, process_device


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize(
    "method, model, mcmc_method",
    [
        (SNPE_C, "mdn", "rejection"),
        (SNPE_C, "maf", "rejection"),
        (SNLE, "maf", "slice"),
        (SNLE, "nsf", "slice_np"),
        (SNRE_A, "mlp", "slice_np_vectorized"),
        (SNRE_B, "resnet", "nuts"),
    ],
)
@pytest.mark.parametrize("data_device", ("cpu", "cuda:0"))
@pytest.mark.parametrize(
    "training_device, prior_device",
    [
        pytest.param("cpu", "cuda", marks=pytest.mark.xfail),
        pytest.param("cuda:0", "cpu", marks=pytest.mark.xfail),
        ("cuda:0", "cuda:0"),
        ("cuda:0", "cuda:0"),
        ("cpu", "cpu"),
    ],
)
def test_training_and_mcmc_on_device(
    method,
    model,
    data_device,
    mcmc_method,
    training_device,
    prior_device,
    prior_type="gaussian",
):
    """Test training on devices.

    This test does not check training speeds.

    """

    num_dim = 2
    num_samples = 10
    num_simulations = 100
    max_num_epochs = 5

    x_o = zeros(1, num_dim).to(data_device)
    likelihood_shift = -1.0 * ones(num_dim).to(prior_device)
    likelihood_cov = 0.3 * eye(num_dim).to(prior_device)

    if prior_type == "gaussian":
        prior_mean = zeros(num_dim).to(prior_device)
        prior_cov = eye(num_dim).to(prior_device)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    else:
        prior = BoxUniform(
            low=-2 * torch.ones(num_dim),
            high=2 * torch.ones(num_dim),
            device=prior_device,
        )

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    training_device = process_device(training_device, prior)

    if method in [SNPE_A, SNPE_C]:
        kwargs = dict(density_estimator=utils.posterior_nn(model=model))
        mcmc_kwargs = (
            dict(sample_with="rejection", mcmc_method=mcmc_method)
            if method == SNPE_C
            else {}
        )
    elif method == SNLE:
        kwargs = dict(density_estimator=utils.likelihood_nn(model=model))
        mcmc_kwargs = dict(sample_with="mcmc", mcmc_method=mcmc_method)
    elif method in (SNRE_A, SNRE_B):
        kwargs = dict(classifier=utils.classifier_nn(model=model))
        mcmc_kwargs = dict(sample_with="mcmc", mcmc_method=mcmc_method)
    else:
        raise ValueError()

    inferer = method(prior, show_progress_bars=False, device=training_device, **kwargs)

    proposals = [prior]

    # Test for two rounds.
    for _ in range(2):
        theta, x = simulate_for_sbi(simulator, proposals[-1], num_simulations)
        theta, x = theta.to(data_device), x.to(data_device)

        _ = inferer.append_simulations(theta, x).train(
            training_batch_size=100, max_num_epochs=max_num_epochs
        )
        posterior = inferer.build_posterior(**mcmc_kwargs).set_default_x(x_o)
        proposals.append(posterior)

    # Check for default device for inference object
    weights_device = next(inferer._neural_net.parameters()).device
    assert torch.device(training_device) == weights_device
    samples = proposals[-1].sample(sample_shape=(num_samples,), x=x_o, **mcmc_kwargs)
    proposals[-1].log_prob(samples)


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "gpu", "cuda", "cuda:0", "cuda:42"])
def test_process_device(device: str):
    process_device(device)
