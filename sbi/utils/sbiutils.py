import time
import warnings
from typing import Tuple, Any

import torch
import torch.nn as nn
import sbi.utils as utils


# XXX standardize? zscore?
# XXX want to insert it in Sequential
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        # XXX guard against std \sim 0 (epsilon or raise)
        return (tensor - self.mean) / self.std


def match_shapes_of_inputs_and_contexts(
    inputs: Any, context: Any, true_context: torch.Tensor, correct_for_leakage: bool
) -> (torch.Tensor, torch.Tensor):
    """
    Formats inputs and contexts into shapes that can be processed by neural density
     estimators.

    Neural density estimators require the first dimension of inputs and contexts to
     match, `inputs.shape == (N, dim_inputs)` and `context.shape == (N, dim_context)`,
     with N being the number of data points where we evaluate the density estimator.
     In this function, we match the shape of context to the shape of inputs.
    Assume that x is the context, x_o is the true_context, theta are inputs/parameters.
    If context has shape (dim_x) or (1, dim_x), we build
     `context = torch.tensor([x, x,..., x])` such that we can later evaluate
     p(theta_n|x) for every parameter set theta_n in inputs
    If context is None, we build `context = torch.tensor([x_o, x_o,..., x_o])` such that
     we can later evaluate p(theta_n|x_o) for every parameter set theta_n in inputs
    If context has shape or (N, dim_x) and inputs has shape (N, dim_theta), we leave
     context unaltered as `context = torch.tensor([x_1, x_2,..., x_N])` such that we can
     later evaluate p(theta_n|x_n) for every parameter set theta_n in inputs with
     n={1,...,N}

    Args:
        inputs: input variables / parameters / thetas
        context: conditioning variables / contexts / x. If None, the context is ignored.
        true_context: if context=None, replace it with true_context
        correct_for_leakage:
            If True, we normalize the output density
            by drawing samples, estimating the acceptance
            ratio, and then scaling the probability with it

    Returns:
        inputs, context with same batch dimension
    """

    # cast inputs to tensor if they are not already
    inputs = utils.torchutils.ensure_tensor(inputs)

    # add batch dimension to `inputs` if needed. `inputs` how has shape
    # (1, num_dim_inputs) or (N, num_dim_inputs), but not (num_dim_inputs)
    inputs = utils.torchutils.ensure_parameter_batched(inputs)

    # use "default context" if None is provided
    if context is None:
        context = true_context
    # cast context to tensor if they are not already
    context = utils.torchutils.ensure_tensor(context)

    # add batch dimension to `context` if needed. `context` how has shape
    # (1, num_dim_context) or (N, num_dim_context), but not (num_dim_context)
    # todo: this will break if we have a multi-dimensional context, e.g. images
    if len(context.shape) == 1:
        context = context[
            None,
        ]
    context = torch.as_tensor(context)

    # if multiple observations, with snpe avoid expensive leakage
    # correction by rejection sampling
    if context.shape[0] > 1 and correct_for_leakage:
        raise ValueError(
            "Only a single context allowed for log-prob when normalizing the density."
            "Please use a for-loop over your inputs and contexts."
        )

    if context.shape[0] != inputs.shape[0]:
        # multiple parameters, single observation:
        # repeat the context to match the parameters
        context = context.repeat(inputs.shape[0], 1)

    if inputs.shape[0] != context.shape[0]:
        # catch all remaining errors after shape-mangling above
        # THIS SHOULD NEVER HAPPEN
        raise ValueError(
            "Number of input items must be equal to number of context items."
        )

    return inputs, context


def sample_posterior_within_prior(
    posterior_nn: torch.nn.Module,
    prior: torch.distributions.Distribution,
    context: torch.Tensor,
    num_samples: int = 1,
    patience: int = 5,
) -> Tuple[torch.Tensor, float]:
    """Return samples from a posterior within the support of the prior via rejection sampling. 
    
    This is relevant for snpe methods and flows for which the posterior tends to have mass outside the prior boundaries. 
    
    This function uses rejection sampling with samples from posterior, to do two things: 
        1) obtain posterior samples within the prior support. 
        2) calculate the fraction of accepted samples as a proxy for correcting the density during evaluation of the posterior. 
    
    Arguments:
        posterior_nn {torch.nn.Module} -- neural net representing the posterior
        prior {torch.distributions.Distribution} -- torch distribution prior
        context {torch.Tensor} -- context for the posterior, i.e., the observed data to condition on. 
    
    Keyword Arguments:
        num_samples {int} -- number of sample to generate (default: {1})
        patience {int} -- upper time bound in minutes, in case sampling takes too long due to strong leakage (default: {5})
    
    Returns:
        Tuple[torch.Tensor, float] -- Accepted samples, and estimated acceptance probability
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

        sample = posterior_nn.sample(num_remaining, context=context)
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
            f"Beware: Rejection sampling resulted in only {samples.shape[0]} samples within patience of {patience} min. Consider having more patience, leakage is {1-acceptance_prob}"
        )

    return samples, acceptance_prob
