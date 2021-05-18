# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import logging
import warnings
from math import pi
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from pyknos.nflows import transforms
from pyro.distributions import Empirical
from torch import Tensor, as_tensor
from torch import nn as nn
from torch import ones, zeros
from torch.distributions import Independent
from torch.distributions.distribution import Distribution
from tqdm.auto import tqdm

from sbi import utils as utils
from sbi.utils.torchutils import atleast_2d


def warn_if_zscoring_changes_data(x: Tensor, duplicate_tolerance: float = 0.1) -> None:
    """Raise warning if z-scoring would create duplicate data points.

    Args:
        x: Simulation outputs.
        duplicate_tolerance: Tolerated proportion of duplicates after z-scoring.
    """

    # Count unique xs.
    num_unique = torch.unique(x, dim=0).numel()

    # z-score.
    zx = (x - x.mean(0)) / x.std(0)

    # Count again and warn on too many new duplicates.
    num_unique_z = torch.unique(zx, dim=0).numel()

    if num_unique_z < num_unique * (1 - duplicate_tolerance):
        warnings.warn(
            """Z-scoring these simulation outputs resulted in {num_unique_z} unique
            datapoints. Before z-scoring, it had been {num_unique}. This can occur due
            to numerical inaccuracies when the data covers a large range of values.
            Consider either setting `z_score_x=False` (but beware that this can be
            problematic for training the NN) or exclude outliers from your dataset.
            Note: if you have already set `z_score_x=False`, this warning will still be
            displayed, but you can ignore it.""",
            UserWarning,
        )


def x_shape_from_simulation(batch_x: Tensor) -> torch.Size:
    ndims = batch_x.ndim
    assert ndims >= 2, "Simulated data must be a batch with at least two dimensions."

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
            f"must be in [{min_val},{max_val}] range"
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
    def __init__(self, mean: Union[Tensor, float], std: Union[Tensor, float]):
        super(Standardize, self).__init__()
        mean, std = map(torch.as_tensor, (mean, std))
        self.mean = mean
        self.std = std
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, tensor):
        return (tensor - self._mean) / self._std


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
    if len(batch_t > 1):
        t_std = torch.std(batch_t[is_valid_t], dim=0)
        t_std[t_std < min_std] = min_std
    else:
        t_std = 1
        logging.warning(
            """Using a one-dimensional batch will instantiate a Standardize transform
            with (mean, std) parameters which are not representative of the data. We
            allow this behavior because you might be loading a pre-trained. If this is
            not the case, please be sure to use a larger batch."""
        )

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
    max_sampling_batch_size: int = 10_000,
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
        max_sampling_batch_size: Batch size for drawing samples from the posterior.
            Takes effect only in the second iteration of the loop below, i.e., in case
            of leakage or `num_samples>max_sampling_batch_size`. Larger batch size
            speeds up sampling.

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

    num_sampled_total, num_remaining = 0, num_samples
    accepted, acceptance_rate = [], float("Nan")
    leakage_warning_raised = False

    # To cover cases with few samples without leakage:
    sampling_batch_size = min(num_samples, max_sampling_batch_size)
    while num_remaining > 0:

        # Sample and reject.
        candidates = (
            posterior_nn.sample(sampling_batch_size, context=x)
            .reshape(sampling_batch_size, -1)
            .cpu()  # Move to cpu to evaluate under prior.
        )
        are_within_prior = within_support(prior, candidates)
        samples = candidates[are_within_prior]
        accepted.append(samples)

        # Update.
        num_sampled_total += sampling_batch_size
        num_remaining -= samples.shape[0]
        pbar.update(samples.shape[0])

        # To avoid endless sampling when leakage is high, we raise a warning if the
        # acceptance rate is too low after the first 1_000 samples.
        acceptance_rate = (num_samples - num_remaining) / num_sampled_total

        # For remaining iterations (leakage or many samples) continue sampling with
        # fixed batch size.
        sampling_batch_size = max_sampling_batch_size
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

    # When in case of leakage a batch size was used there could be too many samples.
    samples = torch.cat(accepted)[:num_samples]
    assert (
        samples.shape[0] == num_samples
    ), "Number of accepted samples must match required samples."

    return samples, as_tensor(acceptance_rate)


def handle_invalid_x(
    x: Tensor, exclude_invalid_x: bool = True
) -> Tuple[Tensor, int, int]:
    """Return Tensor mask that is True where simulations `x` are valid.

    Additionally return number of NaNs and Infs that were found.

    Note: If `exclude_invalid_x` is False, then mask will be True everywhere, ignoring
        potential NaNs and Infs.
    """

    batch_size = x.shape[0]

    # Squeeze to cover all dimensions in case of multidimensional x.
    x = x.reshape(batch_size, -1)

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


def warn_on_iid_x(num_trials):
    """Warn if more than one x was passed."""

    if num_trials > 1:
        warnings.warn(
            f"An x with a batch size of {num_trials} was passed. "
            + """It will be interpreted as a batch of independent and identically
            distributed data X={x_1, ..., x_n}, i.e., data generated based on the
            same underlying (unknown) parameter. The resulting posterior will be with
            respect to entire batch, i.e,. p(theta | X)."""
        )


def warn_on_invalid_x_for_snpec_leakage(
    num_nans: int, num_infs: int, exclude_invalid_x: bool, algorithm: str, round_: int
) -> None:
    """Give a dedicated warning about invalid data for multi-round SNPE-C"""

    if num_nans + num_infs > 0 and exclude_invalid_x:
        if algorithm == "SNPE_C" and round_ > 0:
            logging.warning(
                "When invalid simulations are excluded, multi-round SNPE-C"
                " can `leak` into the regions where parameters led to"
                " invalid simulations. This can lead to poor results."
            )


def check_warn_and_setstate(
    state_dict: Dict, key_name: str, replacement_value: Any, warning_msg: str = ""
) -> Tuple[Dict, str]:
    """
    Check if `key_name` is in `state_dict` and add it if not.

    If the key already existed in the `state_dict`, the dictionary remains
    unaltered. This function also appends to a warning string.

    For developers: The reason that this method only appends to a warning string
    instead of warning directly is that the user might get multiple very similar
    warnings if multiple attributes had to be replaced. Thus, we start off with an
    emtpy string and keep appending all missing attributes. Then, in the end,
    all attributes are displayed along with a full description of the warning.

    Args:
        attribute_name: The name of the attribute to check.
        state_dict: The dictionary to search (and write to if the key does not yet
            exist).
        replacement_value: The value to be written to the `state_dict`.
        warning_msg: String to which the warning message should be appended to.

    Returns:
        A dictionary which contains the key `attribute_name` and a string with an
        appended warning message.
    """

    if key_name not in state_dict.keys():
        state_dict[key_name] = replacement_value
        warning_msg += " `self." + key_name + f" = {str(replacement_value)}`"
    return state_dict, warning_msg


def get_simulations_since_round(
    data: List,
    data_round_indices: List,
    starting_round_index: int,
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


def expit(theta_t: Tensor, lower_bound: Tensor, upper_bound: Tensor) -> Tensor:
    """
    Return the expit() of an input.

    The `expit` transforms an unbounded input to the interval
    `[lower_bound, upper_bound]`.

    Args:
        theta_t: Input to be transformed.
        lower_bound: Lower bound of the transformation.
        upper_bound: Upper bound of the transformation.

    Returns: theta that is bounded between `lower_bound` and `upper_bound`.
    """
    range_ = upper_bound - lower_bound
    return range_ / (1 + torch.exp(-theta_t)) + lower_bound


def logit(theta: Tensor, lower_bound: Tensor, upper_bound: Tensor) -> Tensor:
    """
    Return the logit() of an input.

    The `logit` maps the interval `[lower_bound, upper_bound]` to an unbounded space.

    Args:
        theta: Input to be transformed.
        lower_bound: Lower bound of the transformation.
        upper_bound: Upper bound of the transformation.

    Returns: theta_t that is unbounded.
    """
    range_ = upper_bound - lower_bound
    theta_01 = (theta - lower_bound) / range_

    return torch.log(theta_01 / (1 - theta_01))


def check_dist_class(
    dist, class_to_check: Union[Distribution, Sequence[Distribution]]
) -> Tuple[bool, Optional[Distribution]]:
    """Returns whether the `dist` is instance of `class_to_check`.

    The dist can be hidden in an Independent distribution, a Boxuniform or in a wrapper.
    E.g., when the user called `prepare_for_sbi`, the distribution will in fact be a
    `PytorchReturnTypeWrapper`. Thus, we need additional checks.

    Args:
        dist: Distribution to be checked.

    Returns:
        Whether the `dist` is `Uniform` and the `Uniform` itself.
    """

    # Direct check.
    if isinstance(dist, class_to_check):
        return True, dist
    # Reveal prior dist wrapped by user input checks or BoxUniform / Independent.
    else:
        if hasattr(dist, "prior"):
            dist = dist.prior
        if isinstance(dist, Independent):
            dist = dist.base_dist

        # Check dist.
        if isinstance(dist, class_to_check):
            is_instance = True
            return_dist = dist
        else:
            is_instance = False
            return_dist = None

        return is_instance, return_dist


def within_support(distribution: Any, samples: Tensor) -> Tensor:
    """
    Return whether the samples are within the support or not.

    If first checks whether the `distribution` has a `support` attribute (as is the
    case for `torch.distribution`). If it does not, it evaluates the log-probabilty and
    returns whether it is finite or not (this hanldes e.g. `NeuralPosterior`). Only
    checking whether the log-probabilty is not `-inf` will not work because, as of
    torch v1.8.0, a `torch.distribution` will throw an error at `log_prob()` when the
    sample is out of the support (see #451). In `prepare_for_sbi()`, we set
    `validate_args=False`. This would take care of this, but requires running
    `prepare_for_sbi()` and otherwise throws a cryptic error.

    Args:
        distribution: Distribution under which to evaluate the `samples`, e.g., a
            PyTorch distribution or NeuralPosterior.
        samples: Samples at which to evaluate.

    Returns:
        Tensor of bools indicating whether each sample was within the support.
    """
    # Try to check using the support property, use log prob method otherwise.
    try:
        sample_check = distribution.support.check(samples)

        # Before torch v1.7.0, `support.check()` returned bools for every element.
        # From v1.8.0 on, it directly considers all dimensions of a sample. E.g.,
        # for a single sample in 3D, v1.7.0 would return [[True, True, True]] and
        # v1.8.0 would return [True].
        if sample_check.ndim > 1:
            return torch.all(sample_check, axis=1)
        else:
            return sample_check
    # Falling back to log prob method of either the NeuralPosterior's net, or of a
    # custom wrapper distribution's.
    except (NotImplementedError, AttributeError):
        return torch.isfinite(distribution.log_prob(samples))


class ImproperEmpirical(Empirical):
    """
    Wrapper around pyro's `Emprirical` distribution that returns constant `log_prob()`.

    This class is used in SNPE when no prior is passed. Having a constant
    log-probability will lead to no samples being rejected during rejection-sampling.

    The default behavior of `pyro.distributions.Empirical` is that it returns `-inf`
    for any value that does not **exactly** match one of the samples passed at
    initialization. Thus, all posterior samples would be rejected for not fitting this
    criterion.
    """

    def log_prob(self, value: Tensor) -> Tensor:
        """
        Return ones as a constant log-prob for each input.

        Args:
            value: The parameters at which to evaluate the log-probability.

        Returns:
            Tensor of as many ones as there were parameter sets.
        """
        value = atleast_2d(value)
        return zeros(value.shape[0])


def mog_log_prob(
    theta: Tensor,
    logits_pp: Tensor,
    means_pp: Tensor,
    precisions_pp: Tensor,
) -> Tensor:
    r"""
    Returns the log-probability of parameter sets $\theta$ under a mixture of Gaussians.

    Note that the mixture can have different logits, means, covariances for any theta in
    the batch. This is because these values were computed from a batch of $x$ (and the
    $x$ in the batch are not the same).

    This code is similar to the code of mdn.py in pyknos, but it does not use
    log(det(Cov)) = -2*sum(log(diag(L))), L being Cholesky of Precision. Instead, it
    just computes log(det(Cov)). Also, it uses the above-defined helper
    `_batched_vmv()`.

    Args:
        theta: Parameters at which to evaluate the mixture.
        logits_pp: (Unnormalized) mixture components.
        means_pp: Means of all mixture components. Shape
            (batch_dim, num_components, theta_dim).
        precisions_pp: Precisions of all mixtures. Shape
            (batch_dim, num_components, theta_dim, theta_dim).

    Returns: The log-probability.
    """

    _, _, output_dim = means_pp.size()
    theta = theta.view(-1, 1, output_dim)

    # Split up evaluation into parts.
    weights = logits_pp - torch.logsumexp(logits_pp, dim=-1, keepdim=True)
    constant = -(output_dim / 2.0) * torch.log(torch.tensor([2 * pi]))
    log_det = 0.5 * torch.log(torch.det(precisions_pp))
    theta_minus_mean = theta.expand_as(means_pp) - means_pp
    exponent = -0.5 * utils.batched_mixture_vmv(precisions_pp, theta_minus_mean)

    return torch.logsumexp(weights + constant + log_det + exponent, dim=-1)
