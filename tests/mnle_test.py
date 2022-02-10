# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import pytest
import torch

from sbi.inference import MNLE, MCMCPosterior, likelihood_estimator_based_potential
from sbi.utils import BoxUniform


@pytest.mark.parametrize(
    "sampler",
    (
        "mcmc",
        "rejection",
        "vi",
    ),
)
def test_api_mnle(sampler):

    # Generate mixed data.
    num_simulations = 1000
    theta = torch.rand(num_simulations, 2)
    x = torch.cat(
        (torch.rand(num_simulations, 1), torch.randint(0, 2, (num_simulations, 1))),
        dim=1,
    )

    # Train and infer.
    prior = BoxUniform(torch.zeros(2), torch.ones(2))
    x_o = x[0]
    trainer = MNLE()
    mnle = trainer.append_simulations(theta, x).train(max_num_epochs=1)

    # Test different samplers.
    posterior = trainer.build_posterior(prior=prior, sample_with=sampler)
    posterior.set_default_x(x_o)
    if sampler == "vi":
        posterior.train()
    posterior.sample((1,))

    # MNLE should work with the default potential as well.
    potential_fn, parameter_transform = likelihood_estimator_based_potential(
        mnle, prior, x_o
    )
    posterior = MCMCPosterior(
        potential_fn,
        proposal=prior,
        theta_transform=parameter_transform,
        init_strategy="proposal",
    )
    posterior.sample((1,))
