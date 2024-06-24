import random
import time

import pytest
import torch
from tqdm import tqdm

from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior, process_simulator


def simulation(theta):
    '''Some generic simulation function.'''

    for _ in range(100):
        x = random.random() * theta

    return x


@pytest.mark.parametrize('type', ['list', 'torch', 'numpy'])
def test_theta_type(type):
    # Length of theta, basically
    num_simulations = 10000

    # SBI boilerplate
    prior = BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([10.0]), device='cpu')

    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulation, prior, prior_returns_numpy)
    thetas = prior.sample((num_simulations,))  # this returns a torch.Tensor

    # Testing choices
    if type == 'list':

        def benchmark():
            for theta in tqdm(thetas.tolist()):
                simulator_wrapper(*theta)

    if type == 'torch':

        def benchmark():
            for theta in tqdm(thetas):
                simulator_wrapper(theta)

    if type == 'numpy':

        def benchmark():
            for theta in tqdm(thetas.numpy()):
                simulator_wrapper(theta)

    # Run the benchmark
    tic = time.time()
    benchmark()
    tac = time.time() - tic

    print(f'Theta type: {type}: {tac:.2f} seconds\n')
