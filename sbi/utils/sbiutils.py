# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import logging
import random
import warnings
from math import pi
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pyknos.nflows.transforms as transforms
import torch
import torch.distributions.transforms as torch_tf
from pyro.distributions import Empirical
from torch import Tensor
from torch import nn as nn
from torch import ones, optim, zeros
from torch.distributions import Distribution, Independent, biject_to, constraints

from sbi import utils as utils
from sbi.types import TorchTransform
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


def z_score_parser(z_score_flag: Optional["str"]) -> Tuple[bool, bool]:
    """Parses string z-score flag into booleans.

    Converts string flag into booleans denoting whether to z-score or not, and whether
    data dimensions are structured or independent.

    Args:
        z_score_flag: str flag for z-scoring method stating whether the data
            dimensions are "structured" or "independent", or does not require z-scoring
            ("none" or None).

    Returns:
        Flag for whether or not to z-score, and whether data is structured
    """
    if type(z_score_flag) is bool:
        # Raise warning if boolean was passed.
        warnings.warn(
            "Boolean flag for z-scoring is deprecated as of sbi v0.18.0. It will be "
            "removed in a future release. Use 'none', 'independent', or 'structured' "
            "to indicate z-scoring option."
        )
        z_score_bool, structured_data = z_score_flag, False

    elif (z_score_flag is None) or (z_score_flag == "none"):
        # Return Falses if "none" or None was passed.
        z_score_bool, structured_data = False, False

    elif (z_score_flag == "independent") or (z_score_flag == "structured"):
        # Got one of two valid z-scoring methods.
        z_score_bool = True
        structured_data = True if z_score_flag == "structured" else False

    else:
        # Return warning due to invalid option, defaults to not z-scoring.
        raise ValueError(
            "Invalid z-scoring option. Use 'none', 'independent', or 'structured'."
        )

    return z_score_bool, structured_data


def standardizing_transform(
    batch_t: Tensor, structured_dims: bool = False, min_std: float = 1e-14
) -> transforms.AffineTransform:
    """Builds standardizing transform

    Args:
        batch_t: Batched tensor from which mean and std deviation (across
            first dimension) are computed.
        structured_dim: Whether data dimensions are structured (e.g., time-series,
            images), which requires computing mean and std per sample first before
            aggregating over samples for a single standardization mean and std for the
            batch, or independent (default), which z-scores dimensions independently.
        min_std:  Minimum value of the standard deviation to use when z-scoring to
            avoid division by zero.

    Returns:
        Affine transform for z-scoring
    """

    is_valid_t, *_ = handle_invalid_x(batch_t, True)

    if structured_dims:
        # Structured data so compute a single mean over all dimensions
        # equivalent to taking mean over per-sample mean, i.e.,
        # `torch.mean(torch.mean(.., dim=1))`.
        t_mean = torch.mean(batch_t[is_valid_t])
        # Compute std per-sample first.
        sample_std = torch.std(batch_t[is_valid_t], dim=1)
        sample_std[sample_std < min_std] = min_std
        # Average over all samples for batch std.
        t_std = torch.mean(sample_std)
    else:
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


def standardizing_net(
    batch_t: Tensor,
    structured_dims: bool = False,
    min_std: float = 1e-7,
) -> nn.Module:
    """Builds standardizing network

    Args:
        batch_t: Batched tensor from which mean and std deviation (across
            first dimension) are computed.
        structured_dim: Whether data dimensions are structured (e.g., time-series,
            images), which requires computing mean and std per sample first before
            aggregating over samples for a single standardization mean and std for the
            batch, or independent (default), which z-scores dimensions independently.
        min_std:  Minimum value of the standard deviation to use when z-scoring to
            avoid division by zero.

    Returns:
        Neural network module for z-scoring
    """

    is_valid_t, *_ = handle_invalid_x(batch_t, True)

    if structured_dims:
        # Structured data so compute a single mean over all dimensions
        # equivalent to taking mean over per-sample mean, i.e.,
        # `torch.mean(torch.mean(.., dim=1))`.
        t_mean = torch.mean(batch_t[is_valid_t])
    else:
        # Compute per-dimension (independent) mean.
        t_mean = torch.mean(batch_t[is_valid_t], dim=0)

    if len(batch_t > 1):
        if structured_dims:
            # Compute std per-sample first.
            sample_std = torch.std(batch_t[is_valid_t], dim=1)
            sample_std[sample_std < min_std] = min_std
            # Average over all samples for batch std.
            t_std = torch.mean(sample_std)
        else:
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


def npe_msg_on_invalid_x(
    num_nans: int, num_infs: int, exclude_invalid_x: bool, algorithm: str
) -> None:
    """Warn if there are NaNs or Infs, appropriate to single-round NPE.

    NPE allows to discard invalid simulations. Thus, we simply warn if invalid
    simulations are found (and they are removed from the dataset in
    `append_simulations()`.
    """

    if num_nans + num_infs > 0:
        if exclude_invalid_x:
            logging.warning(
                f"Found {num_nans} NaN simulations and {num_infs} Inf simulations. "
                "They will be excluded from training."
            )
        else:
            logging.warning(
                f"Found {num_nans} NaN simulations and {num_infs} Inf simulations. "
                "They are not excluded from training due to `exclude_invalid_x=False`."
                "Training will likely fail, we strongly recommend "
                f"`exclude_invalid_x=True` for {algorithm}."
            )


def nle_nre_apt_msg_on_invalid_x(
    num_nans: int, num_infs: int, exclude_invalid_x: bool, algorithm: str
) -> None:
    """Warn or raise if there are NaNs or Infs, appropriate to SNLE, SNRE, or APT.

    This will raise an error in the default case of `exclude_invalid_x=False` since
    SNLE/SNRE/APT do not allow to discard invalid simulations (Glöckler et al. 2021).
    If `exclude_invalid_x` has explicitly been set to `True` by the user, this
    function will give a warning about the systematic error.
    """

    if num_nans + num_infs > 0:
        if exclude_invalid_x:
            logging.warn(
                f"Found {num_nans} NaN simulations and {num_infs} Inf simulations."
                f"These will be discarded from training due to "
                f"`exclude_invalid_x=True`. Please be aware that this gives "
                f"systematically wrong results for {algorithm} and is only recommended "
                f"for expert users."
            )
        else:
            raise ValueError(
                f"Found {num_nans} NaN simulations and {num_infs} Inf simulations."
                f"{algorithm} does not allow invalid simulations."
                f"Replace the invalid values with an unreasonably low or high value."
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
    dist, class_to_check: Union[Type, Tuple[Type]]
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
            return torch.all(sample_check, dim=1)
        else:
            return sample_check
    # Falling back to log prob method of either the NeuralPosterior's net, or of a
    # custom wrapper distribution's.
    except (NotImplementedError, AttributeError):
        return torch.isfinite(distribution.log_prob(samples))


def match_theta_and_x_batch_shapes(theta: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Return $\theta$ and `x` with batch shape matched to each other.
    When `x` is just a single observation it is repeated for all entries in the
    batch of $\theta$s. When there is a batch of multiple `x`, i.e., iid `x`, then
    individual `x` are repeated in the pattern AABBCC and individual $\theta$ are
    repeated in the pattern ABCABC to cover all combinations.
    This is needed in nflows in order to have matching shapes of theta and context
    `x` when evaluating the neural network.
    Args:
        x: (a batch of iid) data
        theta: a batch of parameters
    Returns:
        theta: with shape (theta_batch_size * x_batch_size, *theta_shape)
        x: with shape (theta_batch_size * x_batch_size, *x_shape)
    """

    # Theta and x are ensured to have a batch dim, get the shape.
    theta_batch_size, *theta_shape = theta.shape
    x_batch_size, *x_shape = x.shape

    # Repeat iid trials as AABBCC.
    x_repeated = x.repeat_interleave(theta_batch_size, dim=0)
    # Repeat theta as ABCABC.
    theta_repeated = theta.repeat(x_batch_size, 1)

    # Double check: batch size for log prob evaluation must match.
    assert x_repeated.shape == torch.Size([theta_batch_size * x_batch_size, *x_shape])
    assert theta_repeated.shape == torch.Size(
        [theta_batch_size * x_batch_size, *theta_shape]
    )

    return theta_repeated, x_repeated


def mcmc_transform(
    prior: Distribution,
    num_prior_samples_for_zscoring: int = 1000,
    enable_transform: bool = True,
    device: str = "cpu",
    **kwargs,
) -> TorchTransform:
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
        enable_transform: Whether to transform parameters to unconstrained space.
            When False, an identity transform will be returned for `theta_transform`.

    Returns: A transformation that transforms whose `forward()` maps from unconstrained
        (or z-scored) to constrained (or non-z-scored) space.
    """

    if enable_transform:
        # Some distributions have a support argument but it raises a
        # NotImplementedError. We catch this case here.
        try:
            _ = prior.support
            has_support = True
        except (NotImplementedError, AttributeError):
            # NotImplementedError -> Distribution that inherits from torch dist but
            # does not implement support.
            # AttributeError -> Custom distribution that has no support attribute.
            warnings.warn(
                """The passed prior has no support property, transform will be
                constructed from mean and std. If the passed prior is supposed to be
                bounded consider implementing the prior.support property."""
            )
            has_support = False

        # If the distribution has a `support`, check if the support is bounded.
        # If it is not bounded, we want to z-score the space. This is not done
        # by `biject_to()`, so we have to deal with this case separately.
        if has_support:
            if hasattr(prior.support, "base_constraint"):
                constraint = prior.support.base_constraint  # type: ignore
            else:
                constraint = prior.support
            if isinstance(constraint, constraints._Real):
                support_is_bounded = False
            else:
                support_is_bounded = True
        else:
            support_is_bounded = False

        # Prior with bounded support, e.g., uniform priors.
        if has_support and support_is_bounded:
            transform = biject_to(prior.support)
        # For all other cases build affine transform with mean and std.
        else:
            try:
                prior_mean = prior.mean.to(device)
                prior_std = prior.stddev.to(device)
            except (NotImplementedError, AttributeError):
                # NotImplementedError -> Distribution that inherits from torch dist but
                # does not implement mean, e.g., TransformedDistribution.
                # AttributeError -> Custom distribution that has no mean/std attribute.
                warnings.warn(
                    """The passed prior has no mean or stddev attribute, estimating
                    them from samples to build affimed standardizing transform."""
                )
                theta = prior.sample(torch.Size((num_prior_samples_for_zscoring,)))
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

    check_transform(prior, transform)  # type: ignore

    return transform.inv  # type: ignore


def check_transform(
    prior: Distribution, transform: TorchTransform, atol: float = 1e-3
) -> None:
    """Check validity of transformed and re-transformed samples."""

    theta = prior.sample(torch.Size((2,)))

    theta_unconstrained = transform.inv(theta)
    assert (
        theta_unconstrained.shape == theta.shape  # type: ignore
    ), """Mismatch between transformed and untransformed space. Note that you cannot
    use a transforms when using a MultipleIndependent prior with a Dirichlet prior."""

    assert torch.allclose(
        theta, transform(theta_unconstrained), atol=atol  # type: ignore
    ), "Original and re-transformed parameters must be close to each other."


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


def gradient_ascent(
    potential_fn: Callable,
    inits: Tensor,
    theta_transform: Optional[torch_tf.Transform] = None,
    num_iter: int = 1_000,
    num_to_optimize: int = 100,
    learning_rate: float = 0.01,
    save_best_every: int = 10,
    show_progress_bars: bool = False,
    interruption_note: str = "",
) -> Tuple[Tensor, Tensor]:
    """Returns the `argmax` and `max` of a `potential_fn` via gradient ascent.

    The method can be interrupted (Ctrl-C) when the user sees that the potential_fn
    converges. The currently best estimate will be returned.

    The maximum is obtained by running gradient ascent from given starting parameters.
    After the optimization is done, we select the parameter set that has the highest
    `potential_fn` value after the optimization.

    Warning: The default values used by this function are not well-tested. They might
    require hand-tuning for the problem at hand.

    TODO: find a way to tell pyright that transform(...) does not return None.

    Args:
        potential_fn: The function on which to optimize.
        inits: The initial parameters at which to start the gradient ascent steps.
        theta_transform: If passed, this transformation will be applied during the
            optimization.
        num_iter: Number of optimization steps that the algorithm takes
            to find the MAP.
        num_to_optimize: From the drawn `num_init_samples`, use the `num_to_optimize`
            with highest log-probability as the initial points for the optimization.
        learning_rate: Learning rate of the optimizer.
        save_best_every: The best log-probability is computed, saved in the
            `map`-attribute, and printed every `save_best_every`-th iteration.
            Computing the best log-probability creates a significant overhead (thus,
            the default is `10`.)
        show_progress_bars: Whether to show a progressbar for the optimization.
        interruption_note: The message printed when the user interrupts the
            optimization.

    Returns:
        The `argmax` and `max` of the `potential_fn`.
    """

    if theta_transform is None:
        theta_transform = torch_tf.IndependentTransform(
            torch_tf.identity_transform, reinterpreted_batch_ndims=1
        )
    else:
        theta_transform = theta_transform

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

    optimize_inits = theta_transform(optimize_inits)
    optimize_inits.requires_grad_(True)  # type: ignore
    optimizer = optim.Adam([optimize_inits], lr=learning_rate)  # type: ignore

    iter_ = 0

    # Try-except block in case the user interrupts the program and wants to fall
    # back on the last saved `.map_`. We want to avoid a long error-message here.
    try:

        while iter_ < num_iter:

            optimizer.zero_grad()
            probs = potential_fn(theta_transform.inv(optimize_inits)).squeeze()
            loss = -probs.sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if iter_ % save_best_every == 0 or iter_ == num_iter - 1:
                    # Evaluate the optimized locations and pick the best one.
                    log_probs_of_optimized = potential_fn(
                        theta_transform.inv(optimize_inits)
                    )
                    best_theta_iter = optimize_inits[  # type: ignore
                        torch.argmax(log_probs_of_optimized)
                    ]
                    best_log_prob_iter = potential_fn(
                        theta_transform.inv(best_theta_iter)
                    )
                    if best_log_prob_iter > best_log_prob_overall:
                        best_theta_overall = best_theta_iter.detach().clone()
                        best_log_prob_overall = best_log_prob_iter.detach().clone()

                if show_progress_bars:
                    print(
                        "\r",
                        f"Optimizing MAP estimate. Iterations: {iter_+1} / "
                        f"{num_iter}. Performance in iteration "
                        f"{divmod(iter_+1, save_best_every)[0] * save_best_every}: "
                        f"{best_log_prob_iter.item():.2f} (= unnormalized log-prob)",
                        end="",
                    )
                argmax_ = theta_transform.inv(best_theta_overall)
                max_val = best_log_prob_overall

            iter_ += 1

    except KeyboardInterrupt:
        interruption = f"Optimization was interrupted after {iter_} iterations. "
        print(interruption + interruption_note)
        return argmax_, max_val  # type: ignore

    return theta_transform.inv(best_theta_overall), max_val  # type: ignore


def seed_all_backends(seed: Optional[int] = None) -> None:
    """Sets all python, numpy and pytorch seeds."""

    if seed is None:
        seed = int(torch.randint(1_000_000, size=(1,)))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
