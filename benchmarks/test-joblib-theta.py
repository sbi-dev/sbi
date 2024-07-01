import random
import time

import pytest
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior, process_simulator


def simulation(theta: torch.Tensor):
    '''Some generic simulation function.
    You can change this to study, e.g. the impact of numpy's multithreading
    on the code's multiprocessing - np.linalg.eig'''

    for _ in range(100):
        x = random.random() * theta

    return x


# I am adding a dummy run since it seems that there is some overhead in the
# very first benchmark (spawning of the processes?)
@pytest.mark.parametrize('type', ['dummy', 'torch', 'numpy'])
def test_theta_type(type):
    prior = BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([10.0]), device='cpu')

    # num_simulations = 10000
    num_processes = -1

    match type:
        case 'dummy':
            num_simulations = 100
            thetas = prior.sample((num_simulations,))
        case 'torch':
            num_simulations = 50000
            thetas = prior.sample((num_simulations,))
        case 'numpy':
            num_simulations = 50000
            thetas = prior.sample((num_simulations,)).numpy()

    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulation, prior, prior_returns_numpy)

    tic = time.time()

    _x = [
        r
        for r in tqdm(
            Parallel(return_as="generator", n_jobs=num_processes)(
                delayed(simulator_wrapper)(*theta) for theta in thetas
            ),
            total=num_simulations,
        )
    ]

    tac = time.time() - tic
    print(f'Theta type: {type}; runtime {tac:.2f} s')
