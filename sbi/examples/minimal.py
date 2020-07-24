import torch

from sbi.inference import SNPE, infer, prepare_for_sbi
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian


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

    # flexible interface
    prior = torch.distributions.MultivariateNormal(
        loc=prior_mean, covariance_matrix=prior_cov
    )
    simulator, prior = prepare_for_sbi(diagonal_linear_gaussian, prior)
    inference = SNPE(simulator, prior)
    posterior = inference(num_simulations=500)
    posterior.sample((100,), x=x_o)

    return posterior


if __name__ == "__main__":
    print("Simple interface:")
    print(simple())
    print("Flexible interface:")
    print(flexible())
