# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Optional, Union

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


def _cv_log_likelihoods(
    samples: np.ndarray,
    weights: Optional[np.ndarray],
    bandwidths: np.ndarray,
    num_partitions: int,
    **kde_kwargs,
) -> np.ndarray:
    """Return the weighted held-out log-likelihood per bandwidth, mean over folds."""
    scores = np.zeros(len(bandwidths))
    for train, test in KFold(num_partitions).split(samples):
        # sklearn's weighted density evaluation is slower, so fit without weights
        # if there are none.
        train_weights = None if weights is None else weights[train]
        test_weights = np.ones(len(test)) if weights is None else weights[test]
        # Zero-weight samples can have log-density -inf, and 0 * -inf is nan.
        test, test_weights = test[test_weights > 0], test_weights[test_weights > 0]
        if len(test) == 0:
            continue
        for i, bandwidth in enumerate(bandwidths):
            kde = KernelDensity(bandwidth=bandwidth, **kde_kwargs)
            kde.fit(samples[train], sample_weight=train_weights)
            scores[i] += np.sum(test_weights * kde.score_samples(samples[test]))
    return scores / num_partitions


def _weighted_cv_is_possible(weights: np.ndarray, num_partitions: int) -> bool:
    """Return whether a fold has positive weight in its training and held-out part."""
    return any(
        weights[train].sum() > 0 and weights[test].sum() > 0
        for train, test in KFold(num_partitions).split(weights)
    )


# The implementation of KDE was adapted from
# https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/utils/kde.py
def get_kde(
    samples: Tensor,
    bandwidth: Union[float, str] = "cv",
    transform: transform_types = None,
    sample_weights: Optional[np.ndarray] = None,
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
        num_cv_partitions: number of partitions for cross validation
        num_cv_repetitions: how many times to repeat the cross validation to zoom into
            the hyperparameter grid.

    References:
    [1]: https://github.com/scikit-learn/scikit-learn/blob/
         0303fca35e32add9d7346dcb2e0e697d4e68706f/sklearn/neighbors/kde.py
    """
    transform_ = identity_transform if transform is None else transform

    # Make sure transform has event dimension and returns scalar log_prob.
    if transform_.event_dim == 0:
        transform_ = IndependentTransform(transform_, reinterpreted_batch_ndims=1)
    if isinstance(bandwidth, str):
        assert bandwidth in ["cv", "scott", "silvermann"], "invalid kde bandwidth name."

    transformed_samples = transform_(samples).numpy()  # type: ignore
    num_samples, dim_samples = transformed_samples.shape

    kde_kwargs = dict(
        kernel="gaussian",
        algorithm="auto",
        metric="euclidean",
        atol=0,
        rtol=0,
        breadth_first=True,
        leaf_size=40,
        metric_params=None,
    )

    if bandwidth == "scott":
        bandwidth_selected = num_samples ** (-1.0 / (dim_samples + 4))
    elif bandwidth == "silvermann":
        bandwidth_selected = (num_samples * (dim_samples + 2) / 4.0) ** (
            -1.0 / (dim_samples + 4)
        )
    elif bandwidth == "cv":
        weights = None
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=np.float64)
            if _weighted_cv_is_possible(weights, num_cv_partitions):
                # Mean 1, so that the CV scores have the same scale as without weights.
                weights = weights * num_samples / weights.sum()
            else:
                warnings.warn(
                    "The sample weights are concentrated on too few samples for a "
                    "weighted cross-validation. The bandwidth is selected without "
                    "the weights; the KDE itself still uses them.",
                    stacklevel=2,
                )
                weights = None
        mean = np.average(transformed_samples, axis=0, weights=weights)
        variance = np.average(
            (transformed_samples - mean) ** 2, axis=0, weights=weights
        )
        scale = np.sqrt(variance).mean()
        if scale == 0:
            # All dimensions constant, or all weight on copies of one sample.
            scale = transformed_samples.std()
        steps = 10
        lower = 0.1 * scale
        upper = 0.5 * scale
        best_score = -np.inf

        # Run cv multiple times and to "zoom in" to better bandwidths. Every new
        # grid contains the previous best bandwidth, so the score cannot decrease.
        for _ in range(num_cv_repetitions):
            bandwidth_range = np.linspace(lower, upper, steps)
            scores = _cv_log_likelihoods(
                transformed_samples,
                weights,
                bandwidth_range,
                num_cv_partitions,
                **kde_kwargs,
            )
            best_index, second_best_index = np.argsort(-scores, kind="stable")[:2]
            bandwidth_selected = bandwidth_range[best_index]

            if scores[best_index] - best_score <= 0.001:
                break
            best_score = scores[best_index]

            if best_index == 0:
                lower, upper = 0.1 * bandwidth_selected, bandwidth_selected
            elif best_index == steps - 1:
                lower, upper = bandwidth_selected, bandwidth_selected + (upper - lower)
            else:
                lower, upper = sorted((
                    bandwidth_selected,
                    bandwidth_range[second_best_index],
                ))
    elif float(bandwidth) > 0:
        bandwidth_selected = float(bandwidth)
    else:
        raise ValueError("bandwidth must be positive, 'scott', 'silvermann' or 'cv'")

    # Run final fit with selected bandwidth.
    kde = KernelDensity(bandwidth=bandwidth_selected, **kde_kwargs)
    kde.fit(transformed_samples, sample_weight=sample_weights)

    return KDEWrapper(kde, transform_)
