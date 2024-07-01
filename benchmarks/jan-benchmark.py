import time

import numpy as np
import pytest
import torch
from joblib import Parallel, delayed
from torch import Tensor
from tqdm import tqdm


def numpy_simulator(theta: np.ndarray) -> np.ndarray:
    if isinstance(theta, Tensor):
        theta = theta.numpy()
    for _ in range(100):
        np.random.randn(10000) + np.random.randn(*theta.shape)

    return theta


def numpy_torch_simulator(theta: np.ndarray) -> np.ndarray:
    if isinstance(theta, np.ndarray):
        theta = torch.from_numpy(theta)
    for _ in range(100):
        torch.randn(10000) + torch.randn_like(theta)

    return theta.numpy()


def torch_simulator(theta: Tensor) -> Tensor:
    if isinstance(theta, np.ndarray):
        theta = torch.from_numpy(theta)
    for _ in range(100):
        torch.randn(10000) + torch.randn_like(theta)

    return theta


def torch_numpy_simulator(theta: Tensor) -> Tensor:
    if isinstance(theta, Tensor):
        theta = theta.numpy()
    for _ in range(100):
        np.random.randn(10000) + np.random.randn(*theta.shape)

    return torch.from_numpy(theta)


@pytest.mark.parametrize("type", ["torch"])
@pytest.mark.parametrize(
    "simulator",
    # [numpy_simulator]
    # [numpy_torch_simulator]
    [torch_simulator],
)
def test_joblib_benchmark(simulator, type):
    num_simulations = 1000000
    if type == "torch":
        theta = torch.distributions.Uniform(-1, 1).sample((num_simulations,))
    elif type == "numpy":
        theta = np.random.uniform(size=(num_simulations, 1))
    elif type == "list":
        theta = [np.random.uniform(size=(1,)) for _ in range(num_simulations)]
    num_processes = 10

    tic = time.time()
    _x = [
        r
        for r in tqdm(
            Parallel(return_as="generator", n_jobs=num_processes)(  # type: ignore
                delayed(simulator)(batch) for batch in theta
            ),
            total=num_simulations,
            disable=not True,
        )
    ]
    toc_joblib = time.time() - tic
    # print the time for given simulator
    print(f"{simulator.__name__}; arg type: {type}: {toc_joblib:.2f} seconds \n")
