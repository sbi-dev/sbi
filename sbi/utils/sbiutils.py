import logging
from typing import Tuple, Dict, Sequence, Any
import warnings

import torch
from torch import Tensor, as_tensor
import torch.nn as nn
from tqdm.auto import tqdm


def del_entries(dic: Dict[str, Any], entries: Sequence = ()):
    return {k: v for k, v in dic.items() if k not in entries}


def clamp_and_warn(name: str, value: float, min_val: float, max_val: float) -> float:
    clamped_val = max(min_val, min(value, max_val))
    if clamped_val != value:
        logging.warning(
            f"{name}={value} was clamped to {clamped_val}; "
            "must be in [{min_val},{max_val}] range"
        )

    return clamped_val


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
        # TODO Guard against std \sim 0 (epsilon or raise).
        return (tensor - self.mean) / self.std


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
        num_remaining -= num_accepted

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


def warn_on_too_many_nans(x: Tensor, percent_nan_threshold=0.5) -> None:
    """Raises warning if too many (above threshold) NaNs are in given batch of x."""

    percent_nan = find_nan_in_simulations(x).sum() / float(len(x))

    if percent_nan > percent_nan_threshold:
        logging.warning(
            f"""Found {100 * percent_nan} NaNs in simulations. They
            will be excluded from training which the effective number of
            training samples and can impact training performance."""
        )
