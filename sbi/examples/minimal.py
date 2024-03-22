import torch

from sbi.inference import SNPE, infer, simulate_for_sbi
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian
from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
)


def simple():
    num_dim = 3
    x_o = torch.ones(1, num_dim)
    prior_mean = torch.zeros(num_dim)
    prior_cov = torch.eye(num_dim)

    # simple interface
    prior = torch.distributions.MultivariateNormal(
        loc=prior_mean, covariance_matrix=prior_cov
    )
    posterior = infer(
        diagonal_linear_gaussian, prior, "snpe", num_simulations=500, num_workers=1
    )
    posterior.sample((100,), x=x_o)

    return posterior


def flexible():
    num_dim = 3
    x_o = torch.ones(1, num_dim)
    prior_mean = torch.zeros(num_dim)
    prior_cov = torch.eye(num_dim)
    simulator = diagonal_linear_gaussian

    # flexible interface
    prior = torch.distributions.MultivariateNormal(
        loc=prior_mean, covariance_matrix=prior_cov
    )
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)

    inference = SNPE(prior)

    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=500)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    posterior.sample((100,), x=x_o)

    return posterior


if __name__ == "__main__":
    print("Simple interface:")
    print(simple())
    print("Flexible interface:")
    print(flexible())
