import random
import time

import numpy as np
import pytest
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior, process_simulator


def no_casting_simulation(theta: np.ndarray):
    '''Some generic simulation function. Using ndarray as input'''

    for _ in range(100):
        x = random.random() * theta

    return x


def as_tensor_simulation(theta: np.ndarray):
    '''Some generic simulation function, casting ndarray to Tensor with
    as_tensor'''

    theta_tens = torch.as_tensor(theta)
    for _ in range(100):
        x = random.random() * theta_tens

    return x


def from_numpy_simulation(theta: np.ndarray):
    '''Some generic simulation function, casting ndarray to Tensor with
    from_numpy'''

    theta_tens = torch.from_numpy(theta)
    for _ in range(100):
        x = random.random() * theta_tens

    return x


@pytest.mark.parametrize('type', ['dummy', 'no_cast', 'as_tensor', 'from_numpy'])
def test_casting_type(type):
    match type:
        case 'dummy':
            simulation = no_casting_simulation
        case 'no_cast':
            simulation = no_casting_simulation
        case 'as_tensor':
            simulation = as_tensor_simulation
        case 'from_numpy':
            simulation = from_numpy_simulation

    prior = BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([10.0]), device='cpu')

    num_simulations = 10000
    num_processes = 16

    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulation, prior, prior_returns_numpy)
    thetas = prior.sample((num_simulations,)).numpy()

    tic = time.time()

    _x = [
        r
        for r in tqdm(
            Parallel(return_as="generator", n_jobs=num_processes)(
                delayed(simulator_wrapper)(theta) for theta in thetas
            ),
            total=num_simulations,
        )
    ]

    tac = time.time() - tic
    print(f'Casting type: {type}; runtime {tac:.2f} s')
