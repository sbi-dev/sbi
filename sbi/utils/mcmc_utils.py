# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import inspect
import logging
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from sbi.inference import MCMCPosterior
from sbi.utils import mcmc_transform


def get_direct_sampler(
    potential_fn: Callable, prior: Any, x: Optional[Tensor] = None, **kwargs
) -> Callable:
    """
    Returns a sampler from a MCMCPosterior object, given user-defined potential
    function and prior.

    The user-defined potential can be conditional (accepts theta and x as arguments)
    or unconditional (accepting only theta).

    Args:
        potential_fn: User defined potential function. Must be of type Callable.
        prior: Prior distribution for parameter transformation and initialization.
        x: Conditional x value. Provided if using a conditional potential function.

    Returns:
        Callable sampling function from MCMCPosterior object.
    """
    # build transformation to unrestricted space for sampling
    transform = mcmc_transform(prior)

    # potential_fn must take 1 or 2 required arguments: (theta) or (theta, x)
    num_args = num_required_args(potential_fn)
    assert num_args > 0 and num_args < 3, (
        "potential_fn must take 1-2 required arguments"
    )
    is_conditional = num_required_args == 2

    if is_conditional:
        assert x is not None, "x must be provided if potential_fn is conditional"
        posterior = MCMCPosterior(potential_fn, prior, theta_transform=transform)
        posterior.set_default_x(x)

    else:
        logging.warning(
            "x has not been provided. Using unconditional potential function."
        )

        # define an unconditional potential function (ignores x)
        def unconditional_potential_fn(theta, x):
            return potential_fn(theta)

        posterior = MCMCPosterior(
            unconditional_potential_fn, prior, theta_transform=transform, **kwargs
        )
        posterior.set_default_x(torch.zeros(1))  # set default_x to dummy value

    return posterior.sample


def num_required_args(func):
    """
    Utility for counting the number of positional args in a function.

    Args:
        func: A callable function.

    Returns:
        Number of required positional arguments.
    """
    sig = inspect.signature(func)
    return sum(
        1
        for param in sig.parameters.values()
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and param.default is inspect._empty
    )
