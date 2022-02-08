import torch

from sbi.inference import (
    MNLE,
    MCMCPosterior,
    likelihood_estimator_based_potential,
    mixed_likelihood_estimator_based_potential,
)
from sbi.utils import BoxUniform


def test_mnle():

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
    posterior = trainer.build_posterior(prior=prior)

    # build posterior manually
    potential_fn, parameter_transform = mixed_likelihood_estimator_based_potential(
        mnle, prior, x_o
    )
    posterior = MCMCPosterior(
        potential_fn,
        proposal=prior,
        theta_transform=parameter_transform,
        init_strategy="proposal",
    )
    posterior.sample((1,), x=x_o)

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
    posterior.sample((1,), x=x_o)
