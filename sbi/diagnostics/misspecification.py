# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
# code for MMD from:
# https://github.com/mackelab/labproject/blob/main/labproject/metrics/MMD_torch.py


import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from sbi.inference.trainers.npe.npe_base import PosteriorEstimatorTrainer
from sbi.neural_nets.estimators import UnconditionalDensityEstimator
from sbi.utils.metrics import check_c2st
from sbi.utils.sbiutils import nan_tolerant_input_net
from sbi.utils.torchutils import assert_all_finite


def rbf_kernel(x: Tensor, y: Tensor, bandwidth: float):
    dist = torch.cdist(x, y)
    return torch.exp(-(dist**2) / (2.0 * bandwidth**2))


def median_heuristic(x: Tensor, y: Tensor):
    return torch.median(torch.cdist(x, y)).item()


def compute_rbf_mmd(x: Tensor, y: Tensor, bandwidth: float = 1.0, mode: str = "biased"):
    x_kernel = rbf_kernel(x, x, bandwidth)
    y_kernel = rbf_kernel(y, y, bandwidth)
    xy_kernel = rbf_kernel(x, y, bandwidth)
    if mode == "biased":
        mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    elif mode == "unbiased":
        mmd = (
            torch.sum(x_kernel) / (x_kernel.shape[0] * (x_kernel.shape[0] - 1))
            + torch.sum(y_kernel) / (y_kernel.shape[0] * (y_kernel.shape[0] - 1))
            - 2 * torch.mean(xy_kernel)
        )
    else:
        raise ValueError("mode should be either biased or unbiased")
    return mmd


def compute_rbf_mmd_median_heuristic(x: Tensor, y: Tensor, mode: str = "biased"):
    """Median heuristic for bandwidth parameter.

    Described in
    `Large sample analysis of the median heuristic`, Garreau et al, 2018
    (https://arxiv.org/abs/1707.07269)
    """
    bandwidth = median_heuristic(x, y)
    return compute_rbf_mmd(x, y, bandwidth, mode)


def _num_synthetic_samples(n_obs: int, n_total: int, max_samples: int) -> int:
    """Number of synthetic samples compared to ``n_obs`` samples, in each MMD.

    The observed MMD and every MMD under H_0 use this number, so that all of them
    are computed from sets of the same sizes.
    """
    if n_obs < 1:
        raise ValueError("x_obs must contain at least one observation.")
    if max_samples < 1:
        raise ValueError(f"max_samples must be at least 1, got {max_samples}.")
    if n_obs >= n_total:
        raise ValueError(
            "n of observed samples should be less than n of synthetic samples"
        )
    n_synthetic = min(max_samples, n_total - n_obs)
    if n_obs == 1 and n_synthetic == 1:
        raise ValueError(
            "With a single observation, the test needs at least two synthetic samples "
            "per MMD: the MMD between two single samples is the same for any two "
            "distinct samples. Increase max_samples or the number of samples in x."
        )
    return n_synthetic


def calculate_baseline_mmd(
    n_obs: int,
    y: Tensor,
    n_shuffle: int = 1_000,
    max_samples: int = 1_000,
    mode: str = "biased",
):
    """Calculates the MMD between two sets of synthetic data.

    Needed to compute the distribution of mmds under the null hypothesis
    that synthetic and observed samples come from the same distribution.

    Args:
        n_obs: number of observed data points,
            used to determine the number of samples for one set
        y: synthetic data
        n_shuffle: number of shuffles
        max_samples: maximum number of samples in the second set, i.e. the number
            of synthetic samples compared to the ``n_obs`` samples of the first set
        mode: mode of MMD calculation
    """
    if n_shuffle < 1:
        raise ValueError(f"n_shuffle must be at least 1, got {n_shuffle}.")
    n_synthetic = _num_synthetic_samples(n_obs, y.shape[0], max_samples)
    mmds = []
    for _ in range(n_shuffle):
        idx = torch.randperm(y.shape[0], device=y.device)
        mmds.append(
            compute_rbf_mmd_median_heuristic(
                y[idx[:n_obs]], y[idx[n_obs : n_obs + n_synthetic]], mode=mode
            )
        )
    return torch.stack(mmds)


def calculate_p_misspecification(
    x_obs: Tensor,
    x: Tensor,
    n_shuffle: int = 1_000,
    max_samples: int = 1_000,
    mode: str = "biased",
):
    """Calculate the p-value of the misspecification test.

    Args:
        x_obs: observed data
        x: synthetic data
        n_shuffle: number of shuffles
        max_samples: maximum number of synthetic samples to compare to the observed
            data
        mode: mode of MMD calculation ("biased" or "unbiased")
    """
    if x_obs.ndim != 2 or x.ndim != 2 or x_obs.shape[1] != x.shape[1]:
        raise ValueError(
            "x_obs and x must have shapes (num_samples, num_features) with the same "
            "num_features, got "
            f"{tuple(x_obs.shape)} and {tuple(x.shape)}."
        )
    if not (torch.isfinite(x_obs).all() and torch.isfinite(x).all()):
        raise ValueError(
            "x_obs and x must not contain NaN or inf values (in embedding mode, "
            "neither may their embeddings)."
        )
    mmds_baseline = calculate_baseline_mmd(
        x_obs.shape[0], x, n_shuffle=n_shuffle, max_samples=max_samples, mode=mode
    )
    # Random rows, so that the order of the rows in x does not matter.
    n_synthetic = _num_synthetic_samples(x_obs.shape[0], x.shape[0], max_samples)
    idx = torch.randperm(x.shape[0], device=x.device)[:n_synthetic]
    mmd = compute_rbf_mmd_median_heuristic(x_obs, x[idx], mode=mode)
    if not (torch.isfinite(mmds_baseline).all() and torch.isfinite(mmd)):
        raise ValueError(
            "The MMD is not finite, so the misspecification test has no valid "
            "p-value. This happens when the median distance between samples is zero "
            "(for example, when most samples are identical, as in sparse discrete "
            "data), or with mmd_mode='unbiased' and a single sample in either set."
        )
    # The observed MMD counts as one draw under H_0, so the p-value is never 0.
    p_val = (1 + (mmds_baseline >= mmd).sum().item()) / (1 + n_shuffle)
    return p_val, (mmds_baseline, mmd)


def calc_misspecification_mmd(
    x_obs: Tensor,
    x: Tensor,
    inference: Optional[PosteriorEstimatorTrainer] = None,
    mode: str = "x_space",
    n_shuffle: int = 1_000,
    max_samples: int = 1_000,
    mmd_mode: str = "biased",
):
    """Misspecification test based on MMD in data- or embedding space.

    The MMDs under H_0 are computed between synthetic samples only, so the p-value
    is approximate. It is closest to exact when ``x`` holds many more samples than
    ``x_obs``.

    Args:
        x_obs: observed data
        x: synthetic data
        inference: sbi inference object (only used if mode == "embedding")
        mode: space of MMD calculation ("x_space" or "embedding")
        n_shuffle: number of shuffles for computing mmds under H_0
        max_samples: maximum number of synthetic samples to compare to the observed
            data, for the observed MMD and for each MMD under H_0
        mmd_mode: approximation of MMD calculation ("biased" or "unbiased")

    returns:
        p_val, (mmd_baseline,mmd): p-value of the misspecification test
        (MMDs under H_0, mmd), on the device of the compared data (in embedding
        mode, the device of the trained net)
    """
    if mode == "x_space":
        z_obs = x_obs
        z = x
    elif mode == "embedding":
        if inference is None:
            raise ValueError(
                "inference should not be None if mode is 'embedding'. "
                "Please provide an sbi inference object."
            )
        if getattr(inference, "_neural_net", None) is None:
            raise ValueError(
                "No neural net found. The inference object must be trained before "
                "computing the MMD in mode 'embedding'."
            )
        if isinstance(inference._neural_net.embedding_net, nn.modules.linear.Identity):
            warnings.warn(
                "The embedding net might be the identity function, "
                "in that case the MMD is computed in the x-space.",
                stacklevel=2,
            )
        if inference._neural_net.embedding_net is None:
            raise AttributeError(
                "embedding_net attribute is None but is required for misspecification"
                " detection."
            )

        # NaN is allowed only where the embedding net accepts it as padding.
        nan_net = nan_tolerant_input_net(inference._neural_net.embedding_net)
        validate_nan_input = getattr(nan_net, "validate_nan_input", None)
        for data, name in ((x_obs, "x_obs"), (x, "x")):
            assert_all_finite(data, name, allow_nan=nan_net is not None)
            if validate_nan_input is not None and torch.isnan(data).any():
                validate_nan_input(data)
        with torch.no_grad():
            z_obs = inference._neural_net.embedding_net(x_obs.to(inference._device))
            z = inference._neural_net.embedding_net(x.to(inference._device))
    else:
        raise ValueError("mode should be either 'x_space' or 'embedding'")

    p_val, (mmds_baseline, mmd) = calculate_p_misspecification(
        z_obs, z, n_shuffle=n_shuffle, max_samples=max_samples, mode=mmd_mode
    )
    return p_val, (mmds_baseline, mmd)


def _log_prob_hypothesis_test(
    log_probs: Tensor, log_prob_xo: float, alpha: float = 0.05
):
    """Perform a hypothesis test to check if log_prob_xo is unusually low.

    The lo_prob_xo is compared to the given log probabilities from the distribution.

    Args:
        log_probs: array-like, log probabilities of known samples
        log_prob_xo: float, log probability of the test sample
        alpha: significance level (default 0.05)

    Returns:
        - p_value: float, proportion of log_probs below log_prob_xo
        - reject_H0: bool, whether to reject H0 at the given alpha level
    """
    # Compute empirical CDF value (proportion of samples with lower log prob)
    p_value = (log_probs <= log_prob_xo).float().mean()

    # Reject H0 if p_value is below the significance level
    reject_H0 = p_value < alpha

    return p_value, reject_H0


def calc_misspecification_logprob(
    x_val: Tensor,
    x_o: Tensor,
    estimator: UnconditionalDensityEstimator,
    alpha: float = 0.05,
):
    """Perform hypothesis test to check if ``estimator.log_prob(x_o)`` is unusually low.

    The ``estimator.log_prob(x_o)`` logcompared to the log probabilities of samples
    in ``x_val``. First it performs a c2st check of the `estimator`
    using ``x_val``, and warns the user if c2st is poor as test
    results might not be meaningful.

    Args:
        x_val: array-like, known samples to compute baseline logprobs
        x_o: array-like, the test sample or the observation
        estimator: marginal distribution estimator
        alpha: significance level (default 0.05)

    Returns:
        - p_value: float, proportion of log_probs below log_prob_xo
        - reject_H0: bool, whether to reject H0 at the given alpha level
    """
    # first do a c2st check and raise Warning if c2st is high (bad)
    log_probs_val = estimator.log_prob(x_val).detach()
    log_prob_xo = estimator.log_prob(x_o).detach().item()

    n_samples = x_val.shape[0]
    samples = estimator.sample(torch.Size((n_samples,)))
    try:
        check_c2st(x_val, samples, 'MarginalEstimator')
    except AssertionError as e:
        warnings.warn(
            f"{str(e)} "
            "\nProceeding with logprob test, but results might not"
            " be meaningful. Be careful with the interpretation!",
            stacklevel=2,
        )
    # then go ahead to do the logprob hypothesis test
    p_value, reject_H0 = _log_prob_hypothesis_test(
        log_probs_val, log_prob_xo, alpha=alpha
    )
    return p_value, reject_H0
