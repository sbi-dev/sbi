import time
import warnings
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

import sbi.utils as utils


class Standardize(nn.Module):
    """
    Standardize inputs, i.e. subtract mean and divide by standard deviation. Inherits
     from nn.Module so we can use it in nn.Sequential
    """

    def __init__(self, mean, std):
        super(Standardize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        # XXX guard against std \sim 0 (epsilon or raise)
        return (tensor - self.mean) / self.std


def match_shapes_of_theta_and_x(
    theta: Union[Sequence[float], float],
    x: Union[Sequence[float], float],
    x_o: Tensor,
    correct_for_leakage: bool,
) -> (Tensor, Tensor):
    r"""
    Formats parameters theta and simulation outputs x into shapes that can be processed
     by neural density estimators for the posterior $p(\theta|x)$.

    Neural density estimators require the first dimension of theta and x to
     match, `theta.shape == (N, dim_theta)` and `x.shape == (N, dim_x)`,
     with N being the number of data points where we evaluate the density estimator.
     In this function, we match the shape of x to the shape of theta.
    If x has shape (dim_x) or (1, dim_x), we build
     `x = torch.tensor([x, x,..., x])` such that we can later evaluate
     $p(\theta_n|x)$ for every parameter set $\theta_n$ in theta
    If x is None, we build `x = torch.tensor([x_o, x_o,..., x_o])` such that
     we can later evaluate $p(\theta_n|x_o)$ for every parameter set $\theta_n$ in theta
    If x has shape or (N, dim_x) and theta has shape (N, dim_theta), we leave
     x unaltered as `x = torch.tensor([x_1, x_2,..., x_N])` such that we can
     later evaluate $p(\theta_n|x_n)$ for every parameter set $\theta_n$ in theta with
     n={1,...,N}

    Args:
        theta: parameters $\theta$
        x: conditioning variables $x$. If None, x is ignored.
        x_o: if x=None, replace it with x_o
        correct_for_leakage:
            If True, we normalize the output density
            by drawing samples, estimating the acceptance
            ratio, and then scaling the probability with it

    Returns:
        theta, x with same batch dimension
    """

    # cast theta to tensor if they are not already
    theta = torch.as_tensor(theta)

    # add batch dimension to `theta` if needed. `theta` how has shape
    # (1, shape_of_single_theta) or (N, shape_of_single_theta), but not
    # (shape_of_single_theta)
    theta = utils.torchutils.ensure_theta_batched(theta)

    # use x_o if x=None is provided
    if x is None:
        x = x_o
    # cast x to tensor if they are not already
    x = torch.as_tensor(x)

    # add batch dimension to `x` if needed. `x` how has shape
    # (1, shape_of_single_x) or (N, shape_of_single_x), but not (shape_of_single_x)
    # todo: this will break if we have a multi-dimensional x, e.g. images
    if len(x.shape) == 1:
        x = x.unsqueeze(0)

    # if multiple observations, with snpe avoid expensive leakage
    # correction by rejection sampling
    if x.shape[0] > 1 and correct_for_leakage:
        raise ValueError(
            "Only a single conditioning variable x allowed for log-prob when "
            "normalizing the density. Please use a for-loop over your theta and x."
        )

    if x.shape[0] != theta.shape[0]:
        # multiple parameter sets theta, single observation x:
        # repeat the x to match the parameters theta
        x = x.repeat(theta.shape[0], 1)

    if theta.shape[0] != x.shape[0]:
        # catch all remaining errors after shape-mangling above
        # THIS SHOULD NEVER HAPPEN
        raise ValueError("Number of theta items must be equal to number of x items.")

    return theta, x


def sample_posterior_within_prior(
    posterior_nn: torch.nn.Module,
    prior: torch.distributions.Distribution,
    x: Tensor,
    num_samples: int = 1,
    patience: int = 5,
) -> Tuple[Tensor, float]:
    r"""Return samples from a posterior $p(\theta|x)$ within the support of the prior
     via rejection sampling.

    This is relevant for snpe methods and flows for which the posterior tends to have
     mass outside the prior boundaries.

    This function uses rejection sampling with samples from posterior, to do two things:
        1) obtain posterior samples within the prior support.
        2) calculate the fraction of accepted samples as a proxy for correcting the
         density during evaluation of the posterior.

    Args:
        posterior_nn: neural net representing the posterior
        prior: torch distribution prior
        x: conditioning variable $x$ for the posterior $p(\theta|x)$
        num_samples: number of sample to generate
        patience: upper time bound in minutes, in case sampling takes too long
         due to strong leakage

    Returns:
        Accepted samples, and estimated acceptance
         probability
    """

    assert (
        not posterior_nn.training
    ), "posterior nn is in training mode, but has to be in eval mode for sampling."

    samples = []
    num_remaining = num_samples
    num_sampled_total = 0
    tstart = time.time()
    time_over = time.time() - tstart > (patience * 60)

    while num_remaining > 0 and not time_over:

        # XXX: we need this reshape here because posterior_nn.sample sometimes return
        # leading singleton dimension instead of (num_samples), e.g., (1, 10000, 4)
        # instead of (10000, 4). and this cant be handle by IndependentJoint.
        sample = posterior_nn.sample(num_remaining, context=x).reshape(
            num_remaining, -1
        )
        num_sampled_total += num_remaining

        is_within_prior = torch.isfinite(prior.log_prob(sample))
        num_in_prior = is_within_prior.sum().item()

        if num_in_prior > 0:
            samples.append(sample[is_within_prior,].reshape(num_in_prior, -1))
            num_remaining -= num_in_prior

        # update timer
        time_over = time.time() - tstart > (patience * 60)

    # collect all samples in the list into one tensor
    samples = torch.cat(samples)

    # estimate acceptance probability
    acceptance_prob = float((samples.shape[0]) / num_sampled_total)

    if num_remaining > 0:
        warnings.warn(
            f"""Beware: Rejection sampling resulted in only {samples.shape[0]} samples
            within patience of {patience} min. Consider having more patience, leakage
            is {1-acceptance_prob}."""
        )

    return samples, acceptance_prob
