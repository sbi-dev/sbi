import os
from datetime import datetime, timedelta
from multiprocessing import Pool
from time import time

import numpy as np
import torch
from joblib import Parallel, delayed
from p_tqdm import p_map
from tqdm import tqdm

from sbi.inference import simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior, process_simulator

os.environ['OMP_NUM_THREADS'] = '1'


def timeit(func):
    '''Decorator method for printing the execution time'''

    def wrapper(*args, **kwargs):
        print(f'\n > Generation started:\t{datetime.now().strftime("%H:%M:%S")}')
        beg = time()

        func(*args, **kwargs)

        end = time()
        print(f' > Generation complete:\t{datetime.now().strftime("%H:%M:%S")}')

        tds = str(timedelta(seconds=end - beg)).split(':')
        print(f' > Total runtime:\t{tds[0]}h {tds[1]}min {tds[2]:.4}s')

    return wrapper


def simulation(theta):
    '''Some generic simulation function.
    You can change this to study, e.g. the impact of numpy's multithreading
    on the code's multiprocessing - np.linalg.eig'''

    # matrix = np.random.rand(20,20)/theta
    # eigval, eigvec = np.linalg.eig(matrix)
    for _ in range(100):
        np.random.randn(10000)

    return 1


@timeit
def run_simulations_pool(prior, num_simulations, num_processes):
    '''Generates the joint using python's native Pool multiprocessing'''

    theta_low = prior.base_dist.low.cpu().numpy()
    theta_high = prior.base_dist.high.cpu().numpy()
    theta_range = theta_high - theta_low

    thetas = np.random.uniform(size=(num_simulations, 1)) * theta_range + theta_low

    with Pool(num_processes) as p:
        x = p.map(simulation, thetas)

    print(type(x))


@timeit
def run_simulations_p_map(prior, num_simulations, num_processes):
    '''Generates the joint using p_map from the p_tqdm library'''

    # theta_low = prior.base_dist.low.cpu().numpy()
    # theta_high = prior.base_dist.high.cpu().numpy()
    # theta_range = theta_high - theta_low
    #
    # thetas = np.random.uniform(size=(num_simulations,1)) * theta_range + theta_low
    thetas = prior.sample((num_simulations,))
    _x = p_map(simulation, thetas, num_cpus=num_processes)


@timeit
def run_simulations_sim_for_sbi(prior, num_simulations, num_procsses):
    '''Generates the joint using sbi.inference.simulate_for_sbi'''

    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(simulation, prior, prior_returns_numpy)

    _theta, _x = simulate_for_sbi(
        simulator_wrapper, prior, num_simulations, num_procsses
    )


@timeit
def run_simulations_joblib(prior, num_simulations, num_processes):
    '''Generates the joint using joblib'''

    # theta_low = prior.base_dist.low.cpu().numpy()
    # theta_high = prior.base_dist.high.cpu().numpy()
    # theta_range = theta_high - theta_low
    #
    # thetas = np.random.uniform(size=(num_simulations, 1)) * theta_range + theta_low
    thetas = prior.sample((num_simulations,))

    # return 0

    _x = [
        r
        for r in tqdm(
            Parallel(return_as="generator", n_jobs=num_processes)(
                delayed(simulation)(theta) for theta in thetas
            ),
            total=num_simulations,
        )
    ]


if __name__ == '__main__':
    prior = BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([10.0]), device='cpu')

    num_simulations = 10000
    num_processes = 16

    # theta_low = prior.base_dist.low.cpu().numpy()
    # theta_high = prior.base_dist.high.cpu().numpy()
    # theta_range = theta_high - theta_low
    #
    # thetas = np.random.uniform(size=(num_simulations, 1)) * theta_range + theta_low
    # thetas = prior.sample((num_simulations,))

    # Uncomment the benchmark that you want to run.
    # The the native pool benchmark is just as a ground truth to test.

    # print('Benchmarking: sbi.inference.simulate_for_sbi')
    # run_simulations_sim_for_sbi(prior, num_simulations, num_processes)

    # print('Benchmarking: p_map (p_tqdm)')
    # run_simulations_p_map(prior, num_simulations, num_processes)

    # print('Benchmarking: native Pool')
    # run_simulations_pool(prior, num_simulations, num_processes)

    print('Benchmarking: Joblib')
    run_simulations_joblib(prior, num_simulations, num_processes)
