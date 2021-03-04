# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import pytest
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import SNLE, SNPE, SNRE_A, SNRE_B, simulate_for_sbi
from sbi.simulators import linear_gaussian
from sbi.utils.torchutils import process_device


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize(
    "method, model",
    [
        (SNPE, "mdn"),
        (SNPE, "maf"),
        (SNPE, "nsf"),
        (SNLE, "maf"),
        (SNLE, "nsf"),
        (SNRE_A, "mlp"),
        (SNRE_B, "resnet"),
    ],
)
@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_training_and_mcmc_on_device(method, model, device):
    """Test training on devices.

    This test does not check training speeds.

    """
    device = process_device(device)

    num_dim = 2
    num_samples = 10
    num_simulations = 500
    max_num_epochs = 5

    x_o = zeros(1, num_dim)
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior_mean = zeros(num_dim)
    prior_cov = eye(num_dim)
    prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    if method == SNPE:
        kwargs = dict(
            density_estimator=utils.posterior_nn(model=model),
        )
        mcmc_kwargs = dict(
            sample_with_mcmc=True,
            mcmc_method="slice_np",
        )
    elif method == SNLE:
        kwargs = dict(
            density_estimator=utils.likelihood_nn(model=model),
        )
        mcmc_kwargs = dict(mcmc_method="slice")
    elif method in (SNRE_A, SNRE_B):
        kwargs = dict(
            classifier=utils.classifier_nn(model=model),
        )
        mcmc_kwargs = dict(
            mcmc_method="slice_np_vectorized",
        )
    else:
        raise ValueError()

    inferer = method(prior, show_progress_bars=False, device=device, **kwargs)

    proposals = [prior]

    # Test for two rounds.
    for r in range(2):
        (
            theta,
            x,
        ) = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_simulations)
        _ = inferer.append_simulations(theta, x).train(
            training_batch_size=100, max_num_epochs=max_num_epochs
        )
        posterior = inferer.build_posterior(**mcmc_kwargs).set_default_x(x_o)
        proposals.append(posterior)

    proposals[-1].sample(sample_shape=(num_samples,), x=x_o, **mcmc_kwargs)


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "gpu", "cuda", "cuda:0", "cuda:42"])
def test_process_device(device: str):
    process_device(device)
