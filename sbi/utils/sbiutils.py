# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch
from pyknos.nflows import transforms
from torch import Tensor, as_tensor
from torch import nn as nn
from torch import ones, zeros
from tqdm.auto import tqdm


def x_shape_from_simulation(batch_x: Tensor) -> torch.Size:
    assert batch_x.ndim == 2, "Simulated data must be a batch with two dimensions."
    return batch_x[0].unsqueeze(0).shape


def del_entries(dic: Dict[str, Any], entries: Sequence = ()):
    """Delete entries from a dictionary.

    This is typically used to forward arguments to a method selectively, e.g. ignore
    'self' and '__class__' from `locals()`.
    """
    return {k: v for k, v in dic.items() if k not in entries}


def clamp_and_warn(name: str, value: float, min_val: float, max_val: float) -> float:
    """Return clamped value, logging an informative warning if different from value."""
    clamped_val = max(min_val, min(value, max_val))
    if clamped_val != value:
        logging.warning(
            f"{name}={value} was clamped to {clamped_val}; "
            "must be in [{min_val},{max_val}] range"
        )

    return clamped_val


def standardizing_transform(
    batch_t: Tensor, min_std: float = 1e-14
) -> transforms.AffineTransform:
    """Builds standardizing transform

    Args:
        batch_t: Batched tensor from which mean and std deviation (across
            first dimension) are computed.
        min_std:  Minimum value of the standard deviation to use when z-scoring to
            avoid division by zero.

    Returns:
        Affine transform for z-scoring
    """

    is_valid_t, *_ = handle_invalid_x(batch_t, True)

    t_mean = torch.mean(batch_t[is_valid_t], dim=0)
    t_std = torch.std(batch_t[is_valid_t], dim=0)
    t_std[t_std < min_std] = min_std

    return transforms.AffineTransform(shift=-t_mean / t_std, scale=1 / t_std)


class Standardize(nn.Module):
    def __init__(self, mean, std):
        super(Standardize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        return (tensor - self.mean) / self.std


def standardizing_net(batch_t: Tensor, min_std: float = 1e-7) -> nn.Module:
    """Builds standardizing network

    Args:
        batch_t: Batched tensor from which mean and std deviation (across
            first dimension) are computed.
        min_std:  Minimum value of the standard deviation to use when z-scoring to
            avoid division by zero.

    Returns:
        Neural network module for z-scoring
    """

    is_valid_t, *_ = handle_invalid_x(batch_t, True)

    t_mean = torch.mean(batch_t[is_valid_t], dim=0)
    t_std = torch.std(batch_t[is_valid_t], dim=0)
    t_std[t_std < min_std] = min_std

    return Standardize(t_mean, t_std)


@torch.no_grad()
def sample_posterior_within_prior(
    posterior_nn: nn.Module,
    prior,
    x: Tensor,
    num_samples: int = 1,
    show_progress_bars: bool = False,
    warn_acceptance: float = 0.01,
    sample_for_correction_factor: bool = False,
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
        show_progress_bars: Whether to show a progressbar during sampling.
        warn_acceptance: A minimum acceptance rate under which to warn about slowness.
        sample_for_correction_factor: True if this function was called by
            `leakage_correction()`. False otherwise. Will be used to adapt the leakage
             warning.

    Returns:
        Accepted samples and acceptance rate as scalar Tensor.
    """

    assert not posterior_nn.training, "Posterior nn must be in eval mode for sampling."

    # Progress bar can be skipped, e.g. when sampling after each round just for logging.
    pbar = tqdm(
        disable=not show_progress_bars,
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
            if sample_for_correction_factor:
                logging.warning(
                    f"""Drawing samples from posterior to estimate the normalizing
                        constant for `log_prob()`. However, only {acceptance_rate:.0%}
                        posterior samples are within the prior support. It may take a
                        long time to collect the remaining {num_remaining} samples.
                        Consider interrupting (Ctrl-C) and either basing the estimate
                        of the normalizing constant on fewer samples (by calling
                        `posterior.leakage_correction(x_o, num_rejection_samples=N)`,
                        where `N` is the number of samples you want to base the
                        estimate on (default N=10000), or not estimating the
                        normalizing constant at all
                        (`log_prob(..., norm_posterior=False)`. The latter will result
                        in an unnormalized `log_prob()`."""
                )
            else:
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
    """Return Tensor mask that is True where simulations `x` are valid.

    Additionally return number of NaNs and Infs that were found.

    Note: If `exclude_invalid_x` is False, then mask will be True everywhere, ignoring
        potential NaNs and Infs.
    """

    batch_size = x.shape[0]

    x_is_nan = torch.isnan(x).any(dim=1)
    x_is_inf = torch.isinf(x).any(dim=1)
    num_nans = int(x_is_nan.sum().item())
    num_infs = int(x_is_inf.sum().item())

    if exclude_invalid_x:
        is_valid_x = ~x_is_nan & ~x_is_inf
    else:
        is_valid_x = ones(batch_size, dtype=torch.bool)

    return is_valid_x, num_nans, num_infs


def warn_on_invalid_x(num_nans: int, num_infs: int, exclude_invalid_x: bool) -> None:
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


def warn_on_invalid_x(num_nans: int, num_infs: int, exclude_invalid_x: bool) -> None:
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


def warn_on_invalid_x_for_snpec_leakage(
    num_nans: int, num_infs: int, exclude_invalid_x: bool, algorithm: str, round_: int
) -> None:
    """Give a dedicated warning about invalid data for multi-round SNPE-C"""

    if num_nans + num_infs > 0 and exclude_invalid_x:
        if algorithm == "SNPE_C" and round_ > 0:
            logging.warning(
                f"When invalid simulations are excluded, multi-round SNPE-C"
                f" can `leak` into the regions where parameters led to"
                f" invalid simulations. This can lead to poor results."
            )


def get_data_since_round(
    data: List, data_round_indices: List, starting_round_index: int
) -> Tensor:
    """
    Returns tensor with all data coming from a round >= `starting_round`.

    Args:
        data: Each list entry contains a set of data (either parameters, simulation
            outputs, or prior masks).
        data_round_indices: List with same length as data, each entry is an integer that
            indicates which round the data is from.
        starting_round_index: From which round onwards to return the data. We start
            counting from 0.
    """
    return torch.cat(
        [t for t, r in zip(data, data_round_indices) if r >= starting_round_index]
    )


def mask_sims_from_prior(round_: int, num_simulations: int) -> Tensor:
    """Returns Tensor True where simulated from prior parameters.

    Args:
        round_: Current training round, starting at 0.
        num_simulations: Actually performed simulations. This number can be below
            the one fixed for the round if leakage correction through sampling is
            active and `patience` is not enough to reach it.
    """

    prior_mask_values = ones if round_ == 0 else zeros
    return prior_mask_values((num_simulations, 1), dtype=torch.bool)


def batched_mixture_vmv(matrix: Tensor, vector: Tensor) -> Tensor:
    """
    Returns (vector.T * matrix * vector).

    Doing this with einsum() allows for vector and matrix to be batched and have
    several mixture components. In other words, we deal with cases where the matrix and
    vector have two leading dimensions (batch_dim, num_components, **).

    Args:
        matrix: Matrix of shape
            (batch_dim, num_components, parameter_dim, parameter_dim).
        vector: Vector of shape (batch_dim, num_components, parameter_dim).

    Returns:
        Product (vector.T * matrix * vector) of shape (batch_dim, num_components).
    """
    return torch.einsum("bci, bci -> bc", vector, batched_mixture_mv(matrix, vector))


def batched_mixture_mv(matrix: Tensor, vector: Tensor) -> Tensor:
    """
    Returns (matrix * vector).

    Doing this with einsum() allows for vector and matrix to be batched and have
    several mixture components. In other words, we deal with cases where the matrix and
    vector have two leading dimensions (batch_dim, num_components, **).

    Args:
        matrix: Matrix of shape
            (batch_dim, num_components, parameter_dim, parameter_dim).
        vector: Vector of shape (batch_dim, num_components, parameter_dim).

    Returns:
        Product (matrix * vector) of shape (batch_dim, num_components, parameter_dim).
    """
    return torch.einsum("bcij,bcj -> bci", matrix, vector)
