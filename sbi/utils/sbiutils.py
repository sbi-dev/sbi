import logging
import warnings
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor, as_tensor

import sbi.utils as utils
from tqdm.auto import tqdm


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
) -> Tuple[Tensor, Tensor]:
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
    # TODO: can we move this test to sbi_posterior::get_leakage_correction, or will it
    # eventually break if run after x = x.repeat(theta.shape[0], 1) below. If so,
    # consider moving that line to where needed.
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
    posterior_nn: nn.Module,
    prior,
    x: Tensor,
    num_samples: int = 1,
    show_progressbar: bool = False,
    warn_acceptance: float = 0.01,
) -> Tuple[Tensor, Tensor]:
    r"""Return samples from a posterior $p(\theta|x)$ only within the prior support.

    This is relevant for snpe methods and flows for which the posterior tends to have
     mass outside the prior boundaries.

    This function uses rejection sampling with samples from posterior in order to
        1) obtain posterior samples within the prior support, and
        2) calculate the fraction of accepted samples as a proxy for correcting the
           density during evaluation of the posterior.

    Args:
        posterior_nn: Neural net representing the posterior.
        prior: Distribution-like object that evaluates probabilities with `log_prob`.
        x: Conditioning variable $x$ for the posterior $p(\theta|x)$.
        num_samples: Desired number of samples.
        show_progressbar: Whether to show a progressbar during sampling.
        warn_acceptance: A minimum acceptance rate under which to warn about slowness.

    Returns:
        Accepted samples and acceptance rate as scalar Tensor.
    """

    assert not posterior_nn.training, "Posterior nn must be in eval mode for sampling."

    # Progress bar can be skipped, e.g. when sampling after each round just for logging.
    pbar = tqdm(
        disable=not show_progressbar,
        total=num_samples,
        desc=f"Drawing {num_samples} posterior samples",
    )

    num_remaining, num_sampled_total = num_samples, 0
    accepted, acceptance_rate = [], float("Nan")
    leakage_warning_raised = False
    # In each iteration of the loop we sample the remaining number of samples from the
    # posterior. Some of these samples have 0 probability under the prior, i.e. there
    # is leakage (acceptance rate<1) so sample again until reaching `num_samples`.
    while num_remaining > 0:

        candidates = posterior_nn.sample(num_remaining, context=x)
        # TODO we need this reshape here because posterior_nn.sample sometimes return
        # leading singleton dimension instead of (num_samples), e.g., (1, 10000, 4)
        # instead of (10000, 4). This can't be handled by MultipleIndependent, see #141.
        candidates = candidates.reshape(num_remaining, -1)
        num_sampled_total += num_remaining

        are_within_prior = torch.isfinite(prior.log_prob(candidates))
        accepted.append(candidates[are_within_prior])

        num_accepted = are_within_prior.sum().item()
        pbar.update(num_accepted)

        # To avoid endless sampling when leakage is high, we raise a warning if the
        # acceptance rate is too low after the first 1_000 samples.
        acceptance_rate = (num_samples - num_remaining) / num_sampled_total
        if (
            num_sampled_total > 1000
            and acceptance_rate < warn_acceptance
            and not leakage_warning_raised
        ):
            warnings.warn(
                f"""Only {acceptance_rate:.0%} posterior samples are within the
                    prior support. It may take a long time to collect the remaining
                    {num_remaining} samples. Consider interrupting (Ctrl-C)
                    and switching to `sample_with_mcmc=True`."""
            )
            leakage_warning_raised = True  # Ensure warning is raised just once.

        num_remaining -= num_accepted

    pbar.close()

    return torch.cat(accepted), as_tensor(acceptance_rate)


def find_nan_in_simulations(x: Tensor) -> Tensor:
    """Return mask for finding simulated data that contain NaNs from a batch of x.

    Args:
        x: a batch of simulated data x

    Returns:
        mask: True for NaN simulations.
    """

    # Check for any NaNs in every simulation (dim 1 checks across columns).
    return torch.isnan(x).any(dim=1)


def mark_nans_with_zero(x: Tensor) -> Tensor:
    """Return 0-1 Tensor marking NaN simulations with zero for a batch of x.

    This is used a calibration kernel for the SNPE loss. Simulations containing NaNs 
    are indicated with zero such that they do not effect the loss term."""

    kernel_vals = torch.ones(len(x))
    # Set NaN simualtions to zero.
    x_is_nan = find_nan_in_simulations(x)
    kernel_vals[x_is_nan] = 0

    return kernel_vals


def warn_on_too_many_nans(x: Tensor, percent_nan_threshold=0.5) -> None:
    """Raises warning if too many (above threshold) NaNs are in given batch of x."""

    percent_nan = find_nan_in_simulations(x).sum() / float(len(x))

    if percent_nan > percent_nan_threshold:
        logging.warning(
            f"""Found {100 * percent_nan} NaNs in simulations. They
            will be excluded from training which the effective number of
            training samples and can impact training performance."""
        )
