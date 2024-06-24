import random
from datetime import datetime, timedelta
from time import time

import torch
from joblib import Parallel, delayed
from p_tqdm import p_map
from tqdm import tqdm

from sbi.inference import simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior, process_simulator


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


def simulation(theta: torch.Tensor):
    '''Some generic simulation function.
    You can change this to study, e.g. the impact of numpy's multithreading
    on the code's multiprocessing - np.linalg.eig'''

    # matrix = np.random.rand(20,20)/theta
    # eigval, eigvec = np.linalg.eig(matrix)
    for _ in range(1000):
        x = random.random() * theta

    return x


@timeit
def run_simulations_sim_for_sbi(prior, num_simulations, num_procsses):
    '''Generates the joint using sbi.inference.simulate_for_sbi'''

    # prior, num_parameters, prior_returns_numpy = process_prior(prior)
    # simulator_wrapper = process_simulator(simulation, prior, prior_returns_numpy)

    _theta, _x = simulate_for_sbi(
        simulation,
        prior,
        num_simulations,
        num_workers=-1,
        # num_workers=num_procsses,
    )


@timeit
def run_simulations_joblib(prior, num_simulations, num_processes):
    '''Generates the joint using joblib'''

    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    print(prior_returns_numpy)
    simulator_wrapper = process_simulator(simulation, prior, False)
    thetas = prior.sample((num_simulations,)).numpy()

    _x = [
        r
        for r in tqdm(
            Parallel(return_as="generator", n_jobs=num_processes)(
                delayed(simulator_wrapper)(*theta) for theta in thetas
            ),
            total=num_simulations,
        )
    ]


@timeit
def run_simulations_p_map(prior, num_simulations, num_processes):
    '''Generates the joint using joblib'''

    thetas = prior.sample((num_simulations,)).tolist()
    _x = p_map(simulation, thetas, num_cpus=num_processes)


if __name__ == '__main__':
    prior = BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([10.0]), device='cpu')

    num_simulations = 1000000
    num_processes = 16

    # prior, num_parameters, prior_returns_numpy = process_prior(prior)
    # simulator_wrapper = process_simulator(simulation, prior, prior_returns_numpy)
    #
    # theta, x = simulate_for_sbi(simulator_wrapper,
    #                             prior,
    #                             num_simulations,
    #                             num_workers=-1,
    #                             # num_workers=num_procsses
    #                             # simulation_batch_size=None,
    #                             )

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

    print('Benchmarking: Joblib')
    run_simulations_joblib(prior, num_simulations, num_processes)
