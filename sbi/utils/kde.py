from typing import Optional, Union

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from torch import Tensor
from torch.distributions.transforms import IndependentTransform, identity_transform

from sbi.types import transform_types


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
            KDE.
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

    algorithm = "auto"
    kernel = "gaussian"
    metric = "euclidean"
    atol = 0
    rtol = 0
    breadth_first = True
    leaf_size = 40
    metric_params = None

    if bandwidth == "scott":
        bandwidth_selected = num_samples ** (-1.0 / (dim_samples + 4))
    elif bandwidth == "silvermann":
        bandwidth_selected = (num_samples * (dim_samples + 2) / 4.0) ** (
            -1.0 / (dim_samples + 4)
        )
    elif bandwidth == "cv":
        _std = transformed_samples.std()
        steps = 10
        lower = 0.1 * _std
        upper = 0.5 * _std
        current_best = -10000000

        # Run cv multiple times and to "zoom in" to better bandwidths.
        for _ in range(num_cv_repetitions):
            bandwidth_range = np.linspace(lower, upper, steps)
            grid = GridSearchCV(
                KernelDensity(
                    kernel=kernel,
                    algorithm=algorithm,
                    metric=metric,
                    atol=atol,
                    rtol=rtol,
                    breadth_first=breadth_first,
                    leaf_size=leaf_size,
                    metric_params=metric_params,
                ),
                {"bandwidth": bandwidth_range},
                cv=num_cv_partitions,
            )
            grid.fit(transformed_samples)

            # If new best score, update and zoom in.
            if abs(current_best - grid.best_score_) > 0.001:
                current_best = grid.best_score_
            else:
                break

            second_best_index = list(grid.cv_results_["rank_test_score"]).index(2)

            if (grid.best_index_ == 0) or (grid.best_index_ == steps):
                diff = (lower - upper) / steps
                lower = grid.best_index_ - diff
                upper = grid.best_index_ + diff
            else:
                upper = bandwidth_range[second_best_index]
                lower = bandwidth_range[grid.best_index_]

                if upper < lower:
                    upper, lower = lower, upper

        bandwidth_selected = grid.best_params_["bandwidth"]  # type: ignore
    elif float(bandwidth) > 0:
        bandwidth_selected = float(bandwidth)
    else:
        raise ValueError("bandwidth must be positive, 'scott', 'silvermann' or 'cv'")

    # Run final fit with selected bandwidth.
    kde = KernelDensity(
        kernel=kernel,
        algorithm=algorithm,
        metric=metric,
        atol=atol,
        rtol=rtol,
        breadth_first=breadth_first,
        leaf_size=leaf_size,
        metric_params=metric_params,
        bandwidth=bandwidth_selected,
    )
    kde.fit(transformed_samples, sample_weight=sample_weights)

    return KDEWrapper(kde, transform_)
