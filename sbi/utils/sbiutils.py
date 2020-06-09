import logging
from typing import Tuple, Dict, Sequence, Any

import torch
from torch import Tensor, as_tensor, ones
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


@torch.no_grad()
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
            logging.warning(
                f"""Only {acceptance_rate:.0%} posterior samples are within the
                    prior support. It may take a long time to collect the remaining
                    {num_remaining} samples. Consider interrupting (Ctrl-C)
                    and switching to `sample_with_mcmc=True`."""
            )
            leakage_warning_raised = True  # Ensure warning is raised just once.

    pbar.close()

    return torch.cat(accepted), as_tensor(acceptance_rate)


def handle_invalid_x(
    x: Tensor, exclude_invalid_x: bool = True
) -> Tuple[Tensor, int, int]:
    """Return Tensor mask that is True where simulations x are valid.

    Additionally return number of NaNs and Infs that were found.

    Note: If `exclude_invalid` is False, then mask will be True everywhere, ignoring
    potential NaNs and Infs.
    """

    batch_size = x.shape[0]

    x_is_nan = torch.isnan(x).any(dim=1)
    x_is_inf = torch.isinf(x).any(dim=1)
    num_nans = int(x_is_nan.sum().item())
    num_infs = int(x_is_inf.sum().item())

    if exclude_invalid_x:
        x_is_valid = torch.logical_and(
            torch.logical_not(x_is_nan), torch.logical_not(x_is_inf)
        )
    else:
        x_is_valid = ones(batch_size, dtype=torch.bool)

    return x_is_valid, num_nans, num_infs


def warn_on_invalid_x(num_nans: int, num_infs: int, exclude_invalid_x) -> None:
    """Warn if there are NaNs or Infs. Warning text depends on `exclude_invalid_x`."""

    if num_nans + num_infs > 0:
        if exclude_invalid_x:
            logging.warning(
                f"Found {num_nans} NaN simulations and {num_infs} Inf simulations. "
                "They will be excluded from training."
            )
        else:
            logging.warning(
                f"Found {num_nans} NaN simulations and {num_infs} Inf simulations. "
                "Training might fail. Consider setting `exclude_invalid_x=True`."
            )
