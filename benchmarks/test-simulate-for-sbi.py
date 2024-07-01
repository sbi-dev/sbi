import random

import torch

from sbi.inference import simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior, process_simulator


def simulation(theta: torch.Tensor):
    '''Some generic simulation function.'''

    # theta = torch.from_numpy(theta)
    for _ in range(1000):
        x = random.random() * theta

    return x


if __name__ == '__main__':
    prior = BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([10.0]), device='cpu')

    num_simulations = 1000000
    num_processes = 16

    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulation, prior, prior_returns_numpy)

    theta, x = simulate_for_sbi(
        simulator_wrapper,
        prior,
        num_simulations,
        num_processes,
        1,
        show_progress_bar=True,
    )

    print(x.shape)
