# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import logging
import warnings
from math import pi
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import pyknos.nflows.transforms as transforms
from pyro.distributions import Empirical
from torch import Tensor, as_tensor, float32
from torch import nn as nn
from torch import ones, optim, zeros
from torch.distributions import Distribution, Independent, biject_to
import torch.distributions.transforms as torch_tf
from tqdm.auto import tqdm

from sbi import utils as utils
from sbi.utils.torchutils import BoxUniform, atleast_2d


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
def rejection_sample_posterior_within_prior(
    posterior_nn: Any,
    prior: Callable,
    x: Tensor,
    num_samples: int,
    show_progress_bars: bool = False,
    warn_acceptance: float = 0.01,
    sample_for_correction_factor: bool = False,
    max_sampling_batch_size: int = 10_000,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    r"""Return samples from a posterior $p(\theta|x)$ only within the prior support.

    This is relevant for snpe methods and flows for which the posterior tends to have
    mass outside the prior support.

    This function could in principle be integrated into `rejection_sample()`. However,
    to keep the warnings clean, to avoid additional code for integration, and confusing
    if-cases, we decided to keep two separate functions.

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
             warning and to decide whether we have to search for the maximum.
        max_sampling_batch_size: Batch size for drawing samples from the posterior.
            Takes effect only in the second iteration of the loop below, i.e., in case
            of leakage or `num_samples>max_sampling_batch_size`. Larger batch size
            speeds up sampling.
        kwargs: Absorb additional unused arguments that can be passed to
            `rejection_sample()`. Warn if not empty.

    Returns:
        Accepted samples and acceptance rate as scalar Tensor.
    """

    if kwargs:
        logging.warn(
            f"You passed arguments to `rejection_sampling_parameters` that "
            f"are unused when you do not specify a `proposal` in the same "
            f"dictionary. The unused arguments are: {kwargs}"
        )

    # Progress bar can be skipped, e.g. when sampling after each round just for
    # logging.
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
        candidates = posterior_nn.sample(sampling_batch_size, context=x).reshape(
            sampling_batch_size, -1
        )

        # SNPE-style rejection-sampling when the proposal is the neural net.
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
                        constant for `log_prob()`. However, only
                        {acceptance_rate:.0%} posterior samples are within the
                        prior support. It may take a long time to collect the
                        remaining {num_remaining} samples.
                        Consider interrupting (Ctrl-C) and either basing the
                        estimate of the normalizing constant on fewer samples (by
                        calling `posterior.leakage_correction(x_o,
                        num_rejection_samples=N)`, where `N` is the number of
                        samples you want to base the
                        estimate on (default N=10000), or not estimating the
                        normalizing constant at all
                        (`log_prob(..., norm_posterior=False)`. The latter will
                        result in an unnormalized `log_prob()`."""
                )
            else:
                logging.warning(
                    f"""Only {acceptance_rate:.0%} posterior samples are within the
                        prior support. It may take a long time to collect the
                        remaining {num_remaining} samples. Consider interrupting
                        (Ctrl-C) and switching to `sample_with='mcmc'`."""
                )
            leakage_warning_raised = True  # Ensure warning is raised just once.

    pbar.close()

    # When in case of leakage a batch size was used there could be too many samples.
    samples = torch.cat(accepted)[:num_samples]
    assert (
        samples.shape[0] == num_samples
    ), "Number of accepted samples must match required samples."

    return samples, as_tensor(acceptance_rate)


def rejection_sample(
    potential_fn: Callable,
    proposal: Any,
    num_samples: int = 1,
    show_progress_bars: bool = False,
    warn_acceptance: float = 0.01,
    max_sampling_batch_size: int = 10_000,
    num_samples_to_find_max: int = 10_000,
    num_iter_to_find_max: int = 100,
    m: float = 1.2,
) -> Tuple[Tensor, Tensor]:
    r"""Return samples from a `potential_fn` obtained via rejection sampling.

    This function uses rejection sampling with samples from posterior in order to
        1) obtain posterior samples within the prior support, and
        2) calculate the fraction of accepted samples as a proxy for correcting the
           density during evaluation of the posterior.

    Args:
        potential_fn: The potential to sample from. The potential should be passed as
            the logarithm of the desired distribution.
        proposal: The proposal from which to draw candidate samples. Must have a
            `sample()` and a `log_prob()` method.
        num_samples: Desired number of samples.
        show_progress_bars: Whether to show a progressbar during sampling.
        warn_acceptance: A minimum acceptance rate under which to warn about slowness.
        max_sampling_batch_size: Batch size for drawing samples from the posterior.
            Takes effect only in the second iteration of the loop below, i.e., in case
            of leakage or `num_samples>max_sampling_batch_size`. Larger batch size
            speeds up sampling.
        num_samples_to_find_max: Number of samples that are used to find the maximum
            of the `potential_fn / proposal` ratio.
        num_iter_to_find_max: Number of gradient ascent iterations to find the maximum
            of the `potential_fn / proposal` ratio.
        m: Multiplier to the maximum ratio between potential function and the
            proposal. This factor is applied after already having scaled the proposal
            with the maximum ratio of the `potential_fn / proposal` ratio. A higher
            value will ensure that the samples are indeed from the correct
            distribution, but will increase the fraction of rejected samples and thus
            computation time.

    Returns:
        Accepted samples and acceptance rate as scalar Tensor.
    """

    samples_to_find_max = proposal.sample((num_samples_to_find_max,))

    # Define a potential as the ratio between target distribution and proposal.
    def potential_over_proposal(theta):
        return potential_fn(theta) - proposal.log_prob(theta)

    # Search for the maximum of the ratio.
    _, max_log_ratio = optimize_potential_fn(
        potential_fn=potential_over_proposal,
        inits=samples_to_find_max,
        dist_specifying_bounds=proposal,
        num_iter=num_iter_to_find_max,
        learning_rate=0.01,
        num_to_optimize=max(1, int(num_samples_to_find_max / 10)),
        show_progress_bars=False,
    )

    if m < 1.0:
        warnings.warn("A value of m < 1.0 will lead to systematically wrong results.")

    class ScaledProposal:
        """
        Proposal for rejection sampling which is strictly larger than the potential_fn.
        """

        def __init__(self, proposal: Any, max_log_ratio: float, log_m: float):
            self.proposal = proposal
            self.max_log_ratio = max_log_ratio
            self.log_m = log_m

        def sample(self, sample_shape: torch.Size, **kwargs) -> Tensor:
            """
            Samples from the `ScaledProposal` are samples from the `proposal`.
            """
            return self.proposal.sample((sample_shape,), **kwargs)

        def log_prob(self, theta: Tensor, **kwargs) -> Tensor:
            """
            The log-prob is scaled such that the proposal is always above the potential.
            """
            return (
                self.proposal.log_prob(theta, **kwargs)
                + self.max_log_ratio
                + self.log_m
            )

    proposal = ScaledProposal(proposal, max_log_ratio, torch.log(torch.as_tensor(m)))

    with torch.no_grad():
        # Progress bar can be skipped, e.g. when sampling after each round just for
        # logging.
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
            candidates = proposal.sample(sampling_batch_size).reshape(
                sampling_batch_size, -1
            )

            target_proposal_ratio = torch.exp(
                potential_fn(candidates) - proposal.log_prob(candidates)
            )
            uniform_rand = torch.rand(target_proposal_ratio.shape)
            samples = candidates[target_proposal_ratio > uniform_rand]

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
                logging.warning(
                    f"""Only {acceptance_rate:.0%} proposal samples were accepted. It
                        may take a long time to collect the remaining {num_remaining}
                        samples. Consider interrupting (Ctrl-C) and switching to
                        `sample_with='mcmc`."""
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


def optimize_potential_fn(
    potential_fn: Callable,
    inits: Tensor,
    dist_specifying_bounds: Optional[Any] = None,
    num_iter: int = 1_000,
    num_to_optimize: int = 100,
    learning_rate: float = 0.01,
    save_best_every: int = 10,
    show_progress_bars: bool = False,
    interruption_note: str = "",
) -> Tuple[Tensor, Tensor]:
    """
    Returns the `argmax` and `max` of a `potential_fn`.

    The method can be interrupted (Ctrl-C) when the user sees that the log-probability
    converges. The best estimate will be returned.

    The maximum is obtained by running gradient ascent from given starting parameters.
    After the optimization is done, we select the parameter set that has the highest
    `potential_fn` value after the optimization.

    Warning: The default values used by this function are not well-tested. They might
    require hand-tuning for the problem at hand.

    Args:
        potential_fn: The function on which to optimize.
        inits: The initial parameters at which to start the gradient ascent steps.
        dist_specifying_bounds: Distribution the specifies bounds for the optimization.
            If it is a `sbi.utils.BoxUniform`, we transform the space into
            unconstrained space and carry out the optimization there.
        num_iter: Number of optimization steps that the algorithm takes
            to find the MAP.
        num_to_optimize: From the drawn `num_init_samples`, use the `num_to_optimize`
            with highest log-probability as the initial points for the optimization.
        learning_rate: Learning rate of the optimizer.
        save_best_every: The best log-probability is computed, saved in the
            `map`-attribute, and printed every `save_best_every`-th iteration.
            Computing the best log-probability creates a significant overhead (thus,
            the default is `10`.)
        show_progress_bars: Whether or not to show a progressbar for the optimization.
        interruption_note: The message printed when the user interrupts the
            optimization.

    Returns:
        The `argmax` and `max` of the `potential_fn`.
    """

    # If the prior is `BoxUniform`, define a transformation to optimize in
    # unbounded space.
    if dist_specifying_bounds is not None:
        is_boxuniform, boxuniform = check_dist_class(dist_specifying_bounds, BoxUniform)
    else:
        is_boxuniform = False

    if is_boxuniform:

        def tf_inv(theta_t):
            return utils.expit(
                theta_t,
                torch.as_tensor(
                    boxuniform.support.base_constraint.lower_bound, dtype=float32
                ),
                torch.as_tensor(
                    boxuniform.support.base_constraint.upper_bound, dtype=float32
                ),
            )

        def tf(theta):
            return utils.logit(
                theta,
                torch.as_tensor(
                    boxuniform.support.base_constraint.lower_bound, dtype=float32
                ),
                torch.as_tensor(
                    boxuniform.support.base_constraint.upper_bound, dtype=float32
                ),
            )

    else:

        def tf_inv(theta_t):
            return theta_t

        def tf(theta):
            return theta

    init_probs = potential_fn(inits).detach()

    # Pick the `num_to_optimize` best init locations.
    sort_indices = torch.argsort(init_probs, dim=0)
    sorted_inits = inits[sort_indices]
    optimize_inits = sorted_inits[-num_to_optimize:]

    # The `_overall` variables store data accross the iterations, whereas the
    # `_iter` variables contain data exclusively extracted from the current
    # iteration.
    best_log_prob_iter = torch.max(init_probs)
    best_theta_iter = sorted_inits[-1]
    best_theta_overall = best_theta_iter.detach().clone()
    best_log_prob_overall = best_log_prob_iter.detach().clone()

    argmax_ = best_theta_overall
    max_val = best_log_prob_overall

    optimize_inits = tf(optimize_inits)
    optimize_inits.requires_grad_(True)
    optimizer = optim.Adam([optimize_inits], lr=learning_rate)

    iter_ = 0

    # Try-except block in case the user interrupts the program and wants to fall
    # back on the last saved `.map_`. We want to avoid a long error-message here.
    try:

        while iter_ < num_iter:

            optimizer.zero_grad()
            probs = potential_fn(tf_inv(optimize_inits)).squeeze()
            loss = -probs.sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if iter_ % save_best_every == 0 or iter_ == num_iter - 1:
                    # Evaluate the optimized locations and pick the best one.
                    log_probs_of_optimized = potential_fn(tf_inv(optimize_inits))
                    best_theta_iter = optimize_inits[
                        torch.argmax(log_probs_of_optimized)
                    ]
                    best_log_prob_iter = potential_fn(tf_inv(best_theta_iter))
                    if best_log_prob_iter > best_log_prob_overall:
                        best_theta_overall = best_theta_iter.detach().clone()
                        best_log_prob_overall = best_log_prob_iter.detach().clone()

                if show_progress_bars:
                    print(
                        f"""Optimizing MAP estimate. Iterations: {iter_+1} /
                        {num_iter}. Performance in iteration
                        {divmod(iter_+1, save_best_every)[0] * save_best_every}:
                        {best_log_prob_iter.item():.2f} (= unnormalized log-prob""",
                        end="\r",
                    )
                argmax_ = tf_inv(best_theta_overall)
                max_val = best_log_prob_overall

            iter_ += 1

    except KeyboardInterrupt:
        interruption = f"Optimization was interrupted after {iter_} iterations. "
        print(interruption + interruption_note)
        return argmax_, max_val

    return tf_inv(best_theta_overall), max_val


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


def mcmc_transform(
    prior: Any,
    num_prior_samples_for_zscoring: int = 1000,
    enable_transform: bool = True,
    device: str = "cpu",
    **kwargs,
) -> torch_tf.Transform:
    """
    Builds a transform that is applied to parameters during MCMC.

    The resulting transform is defined such that the forward mapping maps from
    constrained to unconstrained space.

    It does two things:
    1) When the prior support is bounded, it transforms the parameters into unbounded
        space.
    2) It z-scores the parameters such that MCMC is performed in a z-scored space.

    Args:
        prior: The prior distribution.
        num_prior_samples_for_zscoring: The number of samples drawn from the prior
            to infer the `mean` and `stddev` of the prior used for z-scoring. Unused if
            the prior has bounded support or when the prior has `mean` and `stddev`
            attributes.
        enable_transform: Whether or not to use a transformation during MCMC.

    Returns: A transformation that transforms whose `forward()` maps from unconstrained
        (or z-scored) to constrained (or non-z-scored) space.
    """

    if enable_transform:
        if hasattr(prior.support, "base_constraint") and hasattr(
            prior.support.base_constraint, "upper_bound"
        ):
            transform = biject_to(prior.support)
        else:
            if hasattr(prior, "mean") and hasattr(prior, "stddev"):
                prior_mean = prior.mean.to(device)
                prior_std = prior.stddev.to(device)
            else:
                theta = prior.sample((num_prior_samples_for_zscoring,))
                prior_mean = theta.mean(dim=0).to(device)
                prior_std = theta.std(dim=0).to(device)

            transform = torch_tf.AffineTransform(loc=prior_mean, scale=prior_std)
    else:
        transform = torch_tf.identity_transform

    # Pytorch `transforms` do not sum the determinant over the parameters. However, if
    # the `transform` explicitly is an `IndependentTransform`, it does. Since our
    # `BoxUniform` is a `Independent` distribution, it will also automatically get a
    # `IndependentTransform` wrapper in `biject_to`. Our solution here is to wrap all
    # transforms as `IndependentTransform`.
    if not isinstance(transform, torch_tf.IndependentTransform):
        transform = torch_tf.IndependentTransform(
            transform, reinterpreted_batch_ndims=1
        )

    return transform.inv


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
    theta: Tensor, logits_pp: Tensor, means_pp: Tensor, precisions_pp: Tensor
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
