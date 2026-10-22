# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from torch import Tensor
from torch.distributions.transforms import IndependentTransform, identity_transform

from sbi.sbi_types import transform_types


class KDEWrapper:
    """Wrapper class to enable sampling and evaluation with a kde object fitted on
    transformed parameters.

    Applies inverse transforms on samples and log abs det Jacobian on log prob.
    """

    def __init__(self, kde, transform):
        self.kde = kde
        self.transform = transform

    def sample(self, *args, **kwargs):
        Y = torch.from_numpy(self.kde.sample(*args, **kwargs).astype(np.float32))
        return self.transform.inv(Y)

    def log_prob(self, parameters_constrained):
        parameters_unconstrained = self.transform(parameters_constrained)
        log_probs = torch.from_numpy(
            self.kde.score_samples(parameters_unconstrained.numpy()).astype(np.float32)
        )
        log_probs += self.transform.log_abs_det_jacobian(
            parameters_constrained, parameters_unconstrained
        )
        assert (
            log_probs.numel() == parameters_constrained.shape[0]
        ), """batch shape mismatch, log_abs_det_jacobian not summing over event
              dimensions?"""
        return log_probs


_KDE_KWARGS = dict(
    kernel="gaussian",
    algorithm="auto",
    metric="euclidean",
    atol=0,
    rtol=0,
    breadth_first=True,
    leaf_size=40,
    metric_params=None,
)


def _unique_samples(
    samples: np.ndarray, weights: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the unique samples in their original order, their counts, and their
    total weights (the counts if there are no weights).

    A held-out sample with an exact copy in the training part makes the CV score grow
    without bound as the bandwidth shrinks.
    """
    _, first, inverse = np.unique(
        samples, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first)
    counts = np.bincount(inverse.ravel())[order]
    totals = counts if weights is None else np.bincount(inverse.ravel(), weights)[order]
    return samples[first[order]], counts, totals


def _scale(samples: np.ndarray, weights: Optional[np.ndarray]) -> float:
    """Return the weighted standard deviation, averaged over dimensions."""
    mean = np.average(samples, axis=0, weights=weights)
    variance = np.average((samples - mean) ** 2, axis=0, weights=weights)
    return float(np.sqrt(variance).mean())


def _cv_log_likelihoods(
    samples: np.ndarray,
    weights: Optional[np.ndarray],
    bandwidths: np.ndarray,
    num_partitions: int,
) -> np.ndarray:
    """Return the weighted held-out log-likelihood per bandwidth, mean over folds."""
    scores = np.zeros(len(bandwidths))
    for train, test in KFold(num_partitions).split(samples):
        train_weights = None if weights is None else weights[train]
        test_weights = np.ones(len(test)) if weights is None else weights[test]
        # Zero-weight samples can have log-density -inf, and 0 * -inf is nan.
        test, test_weights = test[test_weights > 0], test_weights[test_weights > 0]
        if len(test) == 0:
            continue
        for i, bandwidth in enumerate(bandwidths):
            kde = KernelDensity(bandwidth=bandwidth, **_KDE_KWARGS)
            kde.fit(samples[train], sample_weight=train_weights)
            scores[i] += np.sum(test_weights * kde.score_samples(samples[test]))
    return scores / num_partitions


def _weighted_cv_is_possible(weights: np.ndarray, num_partitions: int) -> bool:
    """Return whether a fold has positive weight in its training and held-out part."""
    return any(
        weights[train].sum() > 0 and weights[test].sum() > 0
        for train, test in KFold(num_partitions).split(weights)
    )


def _cv_bandwidth(
    samples: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    num_partitions: int,
    num_repetitions: int,
) -> float:
    """Return the bandwidth with the best weighted cross-validation score.

    Args:
        samples: Unique samples, at least two of them with positive weight.
        weights: Non-negative total weight per sample.
        counts: Number of copies per sample, used if the weights are too
            concentrated for a weighted cross-validation.
        num_partitions: Number of cross-validation folds.
        num_repetitions: Maximal number of grids to search.
    """
    num_samples = len(samples)
    if num_partitions > num_samples:
        warnings.warn(
            f"Only {num_samples} distinct samples, so the cross-validation uses "
            f"{num_samples} partitions instead of {num_partitions}.",
            stacklevel=3,
        )
        num_partitions = num_samples
    if not _weighted_cv_is_possible(weights, num_partitions):
        warnings.warn(
            "The sample weights are concentrated on too few samples for a weighted "
            "cross-validation. The bandwidth is selected without the weights; the "
            "KDE itself still uses them.",
            stacklevel=3,
        )
        weights = counts
    # Mean 1, so that the CV scores have the same scale as without weights.
    weights = weights * num_samples / weights.sum()
    # sklearn's weighted density evaluation is slower, so drop unit weights.
    cv_weights = None if np.all(weights == 1) else weights

    scale = _scale(samples, cv_weights)
    steps = 10
    lower, upper = 0.1 * scale, 0.5 * scale
    best_score = -np.inf
    best_bandwidth = upper

    # Every new grid contains the previous best bandwidth, so the score cannot
    # decrease.
    for _ in range(num_repetitions):
        bandwidth_range = np.linspace(lower, upper, steps)
        scores = _cv_log_likelihoods(
            samples, cv_weights, bandwidth_range, num_partitions
        )
        best_index, second_best_index = np.argsort(-scores, kind="stable")[:2]
        best_bandwidth = bandwidth_range[best_index]

        if scores[best_index] - best_score <= 0.001:
            break
        best_score = scores[best_index]

        if best_index == 0:
            lower, upper = 0.1 * best_bandwidth, best_bandwidth
        elif best_index == steps - 1:
            lower, upper = best_bandwidth, best_bandwidth + (upper - lower)
        else:
            lower, upper = sorted((best_bandwidth, bandwidth_range[second_best_index]))
    return float(best_bandwidth)


# The implementation of KDE was adapted from
# https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/utils/kde.py
def get_kde(
    samples: Tensor,
    bandwidth: Union[float, str] = "cv",
    transform: transform_types = None,
    sample_weights: Optional[Union[np.ndarray, Tensor]] = None,
    num_cv_partitions: int = 20,
    num_cv_repetitions: int = 5,
) -> KDEWrapper:
    """Get KDE estimator with selected bandwidth.

    Args:
        samples: Samples to perfrom KDE on
        bandwidth: Bandwidth method, 'silvermann' or 'scott' heuristics, or 'cv' for a
            tailored cross validation to find the best bandwidth for passed samples.
        transform: Optional transform applied before running kde.
        sample_weights: Sample weights attached to the samples, used to perform weighted
            KDE. With `bandwidth="cv"`, the cross-validation is weighted as well,
            unless the weights are concentrated on too few samples for it.
        num_cv_partitions: number of partitions for cross validation, reduced to the
            number of distinct samples if there are fewer.
        num_cv_repetitions: how many times to repeat the cross validation to zoom into
            the hyperparameter grid.

    Raises:
        ValueError: If the samples are not two-dimensional or not finite, if
            `sample_weights` has the wrong shape, negative or non-finite entries or
            only zeros, if `bandwidth` is invalid, or if `bandwidth="cv"` with
            `num_cv_partitions < 2` or `num_cv_repetitions < 1`.

    References:
    [1]: https://github.com/scikit-learn/scikit-learn/blob/
         0303fca35e32add9d7346dcb2e0e697d4e68706f/sklearn/neighbors/kde.py
    """
    transform_ = identity_transform if transform is None else transform

    # Make sure transform has event dimension and returns scalar log_prob.
    if transform_.event_dim == 0:
        transform_ = IndependentTransform(transform_, reinterpreted_batch_ndims=1)
    if isinstance(bandwidth, str) and bandwidth not in ["cv", "scott", "silvermann"]:
        raise ValueError("bandwidth must be positive, 'scott', 'silvermann' or 'cv'.")

    transformed_samples = transform_(samples).numpy()  # type: ignore
    if transformed_samples.ndim != 2:
        raise ValueError("samples must have shape (num_samples, num_dims).")
    num_samples, dim_samples = transformed_samples.shape
    if not np.isfinite(transformed_samples).all():
        raise ValueError("samples must be finite, after the transform if one is given.")
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if sample_weights.shape != (num_samples,):
            raise ValueError(
                f"sample_weights must have shape ({num_samples},), got "
                f"{sample_weights.shape}."
            )
        if not np.isfinite(sample_weights).all() or (sample_weights < 0).any():
            raise ValueError("sample_weights must be finite and non-negative.")
        if sample_weights.max() == 0:
            raise ValueError("sample_weights must not all be zero.")
        # Max 1, so that sums cannot overflow. The KDE does not depend on the scale.
        sample_weights = sample_weights / sample_weights.max()

    if bandwidth == "scott":
        bandwidth_selected = num_samples ** (-1.0 / (dim_samples + 4))
    elif bandwidth == "silvermann":
        bandwidth_selected = (num_samples * (dim_samples + 2) / 4.0) ** (
            -1.0 / (dim_samples + 4)
        )
    elif bandwidth == "cv":
        if num_cv_partitions < 2 or num_cv_repetitions < 1:
            raise ValueError(
                "num_cv_partitions must be at least 2 and num_cv_repetitions at "
                "least 1 for bandwidth='cv'."
            )
        unique_samples, counts, totals = _unique_samples(
            transformed_samples, sample_weights
        )
        # Count, not spread: a weighted spread of copies need not be exactly zero.
        if np.count_nonzero(totals) < 2:
            warnings.warn(
                "All samples are identical (ignoring samples with zero weight), so "
                "the KDE is a point mass with a small bandwidth.",
                stacklevel=2,
            )
            bandwidth_selected = 1e-5
        else:
            bandwidth_selected = _cv_bandwidth(
                unique_samples, totals, counts, num_cv_partitions, num_cv_repetitions
            )
    elif np.isfinite(float(bandwidth)) and float(bandwidth) > 0:
        bandwidth_selected = float(bandwidth)
    else:
        raise ValueError("bandwidth must be positive, 'scott', 'silvermann' or 'cv'.")

    # Run final fit with selected bandwidth.
    kde = KernelDensity(bandwidth=bandwidth_selected, **_KDE_KWARGS)
    kde.fit(transformed_samples, sample_weight=sample_weights)

    return KDEWrapper(kde, transform_)
