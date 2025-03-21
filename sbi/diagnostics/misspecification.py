# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
# code for MMD from:
# https://github.com/mackelab/labproject/blob/main/labproject/metrics/MMD_torch.py


import warnings

import torch
import torch.nn as nn


def rbf_kernel(x, y, bandwidth):
    dist = torch.cdist(x, y)
    return torch.exp(-(dist**2) / (2.0 * bandwidth**2))


def median_heuristic(x, y):
    return torch.median(torch.cdist(x, y)).item()


def compute_rbf_mmd(x, y, bandwidth=1.0, mode="biased"):
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


def compute_rbf_mmd_median_heuristic(x, y, mode="biased"):
    """median heuristic for bandwidth parameter as described in
    Large sample analysis of the median heuristic, Garreau et al, 2018
    (https://arxiv.org/abs/1707.07269)
    """
    bandwidth = median_heuristic(x, y)
    return compute_rbf_mmd(x, y, bandwidth, mode)


def calculate_baseline_mmd(n_obs, y, n_shuffle=1_000, max_samples=1_000, mode="biased"):
    """calculates the MMD between two sets of synthetic data.
       needed to compute the distribution of mmds under the null hypothesis
       that synthetic and observed samples come from the same distribution.
    n_obs: number of observed data points,
        used to determine the number of samples for one set
    y: synthetic data
    n_shuffle: number of shuffles
    max_samples: maximum number of samples to use
    mode: mode of MMD calculation
    """
    mmds = torch.zeros(n_shuffle)
    if n_obs > y.shape[0]:
        raise ValueError(
            "n of observed samples should be less than n of synthetic samples"
        )
    for i in range(n_shuffle):
        idx = torch.randperm(y.shape[0])[:max_samples]
        mmds[i] = compute_rbf_mmd_median_heuristic(
            y[idx[:n_obs]], y[idx[n_obs:]], mode=mode
        )
    return mmds


def calculate_p_misspecification(
    x_obs, x, n_shuffle=1_000, max_samples=1_000, mode="biased"
):
    """calculate the p-value of the misspecification test.
    x_obs: observed data
    x: synthetic data
    n_shuffle: number of shuffles
    max_samples: maximum number of samples to use
    mode: mode of MMD calculation ("biased" or "unbiased")
    """
    mmds_baseline = calculate_baseline_mmd(
        x_obs.shape[0], x, n_shuffle=n_shuffle, max_samples=max_samples, mode=mode
    )
    mmd = compute_rbf_mmd_median_heuristic(x_obs, x[:max_samples], mode=mode)
    p_val = 1 - (mmds_baseline < mmd).sum().item() / n_shuffle
    return p_val, (mmds_baseline, mmd)


def calc_misspecification_mmd(
    x_obs,
    x,
    inference=None,
    mode="x_space",
    n_shuffle=1_000,
    max_samples=1_000,
    mmd_mode="biased",
):
    """misspecification test based on MMD in data- or embedding space.
    x_obs: observed data
    x: synthetic data
    inference: sbi inference object (only used if mode == "embedding")
    mode: space of MMD calculation ("x_space" or "embedding")
    n_shuffle: number of shuffles for computing mmds under H_0
    max_samples: maximum number of samples to use
        (when we have too many synthetic samples x)
    mmd_mode: approximation of MMD calculation ("biased" or "unbiased")
    returns:
        p_val, (mmd_baseline,mmd): p-value of the misspecification test
                                    (MMDs under H_0, mmd)
    """
    if mode == "x_space":
        z_obs = x_obs
        z = x
    elif mode == "embedding":
        if inference is None:
            raise ValueError(
                "inference should not be None if mode is 'embedding'."
                "please provide an sbi inference object"
            )
        if not hasattr(inference, "_neural_net"):
            raise ValueError(
                "no neural net found,"
                "neural_net should not be None when mode is 'embedding'"
            )
        if isinstance(inference._neural_net.embedding_net, nn.modules.linear.Identity):
            warnings.warn(
                "The embedding net might be the identity function,"
                "in that case the MMD is computed in the x-space.",
                stacklevel=2,
            )
        z_obs = inference._neural_net.embedding_net(x_obs).detach()
        z = inference._neural_net.embedding_net(x).detach()
    else:
        raise ValueError("mode should be either 'x_space' or 'embedding'")

    p_val, (mmds_baseline, mmd) = calculate_p_misspecification(
        z_obs, z, n_shuffle=n_shuffle, max_samples=max_samples, mode=mmd_mode
    )
    return p_val, (mmds_baseline, mmd)


def log_prob_hypothesis_test(log_probs, log_prob_xo, alpha=0.05):
    """
    Perform a hypothesis test to check if log_prob_xo is unusually low
    compared to the given log probabilities from the distribution.

    Parameters:
    - log_probs: array-like, log probabilities of known samples
    - log_prob_xo: float, log probability of the test sample
    - alpha: significance level (default 0.05)

    Returns:
    - p_value: float, proportion of log_probs below log_prob_xo
    - reject_H0: bool, whether to reject H0 at the given alpha level
    """
    # Compute empirical CDF value (proportion of samples with lower log prob)
    p_value = (log_probs <= log_prob_xo).float().mean()

    # Reject H0 if p_value is below the significance level
    reject_H0 = p_value < alpha

    return p_value, reject_H0
