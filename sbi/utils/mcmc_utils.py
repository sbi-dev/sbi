# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import logging
from typing import Callable, Optional, Union

from numpy import ndarray
from torch import Tensor

from sbi.samplers.mcmc import SliceSamplerSerial, SliceSamplerVectorized
from sbi.utils.torchutils import tensor2numpy


def get_mcmc_samples(
    potential_fn: Callable,
    initial_params: Union[ndarray, Tensor],
    num_samples: int = 1000,
    vectorized: bool = True,
    num_workers: int = 1,
    num_chains: int = 1,
    x: Optional[Union[ndarray, Tensor]] = None,
    **kwargs,
) -> ndarray:
    """
    Run slice MCMC sampling using a specified potential function.

    Args:
        potential_fn: User defined potential function. Must be of type Callable.
        initial_params: Initial parameters for the MCMC sampler. First dimension
            must match num_chains (default=1).
        num_samples: Number of samples to draw.
        vectorized: Whether to use the vectorized implementation.
        num_workers: Number of workers to use for vectorized sampling.
        x: The input to the potential function.

    Returns:
        Samples from the posterior distribution of shape
            [num_chains, num_samples, parameter_dim]
    """
    # Check that the first dimension of initial_params matches num_chains.
    assert initial_params.shape[0] == num_chains, (
        "first dimension of initial_params must match num_chains"
    )

    # Convert initial_params to NumPy array if necessary.
    if isinstance(initial_params, Tensor):
        initial_params = tensor2numpy(initial_params)

    if x is None:
        logging.warning(
            "x has not been provided. Using unconditional potential function."
        )
    else:
        if isinstance(x, Tensor):
            x = tensor2numpy(x)

    # Define a log probability function that the sampler expects.
    def lp_fn(theta: ndarray, x: Optional[ndarray] = x) -> float:
        logp = potential_fn(theta, x=x)
        return logp

    if vectorized:
        # Instantiate the vectorized slice sampler.
        sampler = SliceSamplerVectorized(
            log_prob_fn=lp_fn,
            init_params=initial_params,
            num_workers=num_workers,
            num_chains=num_chains,
            **kwargs,
        )
    else:
        # Instantiate the serial slice sampler.
        sampler = SliceSamplerSerial(
            log_prob_fn=lp_fn,
            init_params=initial_params,
            num_chains=num_chains,
            **kwargs,
        )

    # Run the sampler
    samples = sampler.run(num_samples)
    return samples
