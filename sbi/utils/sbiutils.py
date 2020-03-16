import time
from typing import Tuple

import torch
import torch.nn as nn


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


def build_inputs_and_contexts(inputs, context, true_context, correct_for_leakage):
    """
    Formats inputs and context into the correct shape

    Args:
        inputs: Tensor, input variables.
        context: Tensor or None, conditioning variables. If a Tensor, it must have the same number or rows as the inputs. If None, the context is ignored.
        true_context: if context=None, replace it with true_context
        correct_for_leakage:
            If True, we normalize the output density
            by drawing samples, estimating the acceptance
            ratio, and then scaling the probability with it

    Returns:
        inputs, context as torch.tensors
    """

    inputs = torch.as_tensor(inputs)

    if len(inputs.shape) == 1:
        inputs = inputs[
            None,
        ]  # append batch dimension

    # use "default context" if None is provided
    if context is None:
        context = true_context

    # if multiple observations, with snape avoid expensive leakage
    # correction by rejection sampling
    if len(context.shape) > 1 and context.shape[0] > 1 and correct_for_leakage:
        raise ValueError(
            "Only a single context is allowed for log-prob when normalizing the density. "
            "Please use a for-loop over your inputs and contexts."
        )

    # @ append batch dim
    if len(context.shape) == 1:
        context = context[
            None,
        ]
    context = torch.as_tensor(context)

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
    """Generate samples from a posterior that have support under the prior using rejection sampling. 
    
    This is relevant for posteriors with leakage outside the prior:
    Sample for the posterior (torch neural net) and reject samples if outside of prior support. Estimate acceptance prob by counting accepted samples, used e.g., for correcting the posterior density for evaluation. 
    
    Arguments:
        posterior_nn {torch.nn.Module} -- neural net representing the posterio
        prior {torch.distributions.Distribution} -- torch distribution prior
        context {torch.Tensor} -- context for the posterior, i.e., the observed data to condition on. 
    
    Keyword Arguments:
        num_samples {int} -- number of sample to generate (default: {1})
        patience {int} -- anker in time in case sampling takes too long due to strong leakage (default: {5})
    
    Returns:
        Tuple[torch.Tensor, float] -- Accepted samples, and estimated acceptance probability
    """

    # turn on nn eval mode
    posterior_nn.eval()

    samples = []
    num_accepted = 0
    num_sampled = 0
    tstart = time.time()
    time_over = time.time() - tstart > (patience * 60)

    # sample until done or patience over
    while num_accepted < num_samples and not time_over:

        n_remaining = num_samples - num_accepted

        sample = posterior_nn.sample(n_remaining, context=context)
        num_sampled += n_remaining

        # get mask of samples within prior
        mask = torch.isfinite(
            prior.log_prob(sample)
        )  # log prob is inf outside of prior
        num_valid = mask.sum()

        if num_valid > 0:
            samples.append(sample[mask,].reshape(num_valid, -1))
            num_accepted += num_valid

        # update timer
        time_over = time.time() - tstart > (patience * 60)

    # accumulate list of accepted samples in single tensor
    samples = torch.stack(samples).reshape(num_accepted, -1)

    # estimate acceptance probability
    acceptance_prob = float(num_accepted / num_sampled)

    # turn back on training mode
    posterior_nn.train()

    assert (
        samples.shape[0] == num_samples
    ), f"sampling from posterior within prior with patience {patience} failed : {samples.shape[0]} vs. {num_samples}."

    return samples, acceptance_prob
