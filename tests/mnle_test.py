import torch

from sbi.inference import SNLE
from sbi.utils import BoxUniform, likelihood_nn


def test_mnle():

    # Generate mixed data.
    num_simulations = 1000
    theta = torch.rand(num_simulations, 2)
    x = torch.cat(
        (torch.rand(num_simulations, 1), torch.randint(0, 2, (num_simulations, 1))),
        dim=1,
    )

    # Train and infer.
    trainer = SNLE(density_estimator=likelihood_nn(model="mnle", z_score_x=None))
    trainer.append_simulations(theta, x).train()
    posterior = trainer.build_posterior(prior=BoxUniform(torch.zeros(2), torch.ones(2)))
    posterior.sample((1,), x=x[0])
