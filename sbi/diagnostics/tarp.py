# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""
Implementation taken from Lemos et al, 'Sampling-Based Accuracy Testing of
Posterior Estimators for General Inference' https://arxiv.org/abs/2302.03026

The TARP diagnostic is a global diagnostic which can be used to check a
trained posterior against a set of true values of theta.
"""

import warnings
from typing import Callable, Optional, Tuple

import torch
from scipy.stats import kstwo
from torch import Tensor

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils.diagnostics_utils import (
    get_posterior_samples_on_batch,
    remove_nans_and_infs_in_x,
)
from sbi.utils.metrics import l2


def run_tarp(
    thetas: Tensor,
    xs: Tensor,
    posterior: NeuralPosterior,
    references: Optional[Tensor] = None,
    num_posterior_samples: int = 1000,
    num_workers: int = 1,
    show_progress_bar: bool = True,
    distance: Callable = l2,
    num_bins: Optional[int] = None,
    z_score_theta: bool = True,
    use_batched_sampling: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Estimates coverage of samples given true values thetas with the TARP method.
    Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

    The TARP diagnostic is a global diagnostic which can be used to check a
    trained posterior against a set of true values of theta.

    Args:
        thetas: ground-truth parameters for tarp, simulated from the prior.
        xs: observed data for tarp, simulated from thetas.
        posterior: a posterior obtained from sbi.
        references: reference points, one per theta, of shape
            ``(num_tarp_samples, num_dims)``. Draw each reference independently of
            its theta and of the posterior samples, e.g. from the prior. If
            ``None``, they are drawn uniformly from a box that treats ``thetas`` and
            posterior samples alike, see ``get_tarp_references``.
        num_posterior_samples: number of approximate posterior samples used for ranking.
        num_workers: number of CPU cores to use for running inference in parallel.
        show_progress_bar: whether to display a progress over sbc runs.
        distance: the distance metric to use when computing the distance.
            Should be a callable function that accepts two tensors and
            computes the distance between them, e.g. given two tensors
            of shape ``(batch, 3)`` and ``(batch,3)``, this function should
            return ``(batch,1)`` distance values.
            Possible values: ``sbi.utils.metrics.l1`` or
            ``sbi.utils.metrics.l2``. ``l2`` is the default.
        num_bins: number of bins to use for the credibility values. If ``None``,
            one bin per simulation, so that the KS test in ``check_tarp``, which
            sees the coverage values only at the bin edges, loses little power.
        z_score_theta : whether to rescale each parameter dimension to ``[0, 1]``,
            using the minimum and maximum over thetas and posterior samples.
        use_batched_sampling: whether to use batched sampling for posterior samples.

    Returns:
        ecp: Expected coverage probability (``ecp``), see equation 4 of the paper,
            at each credibility value in ``alpha``. The coverage values behind it
            carry uniform noise, so that they are uniform on ``[0, 1]`` for a
            calibrated posterior and any ``num_posterior_samples``; ``ecp`` varies
            between calls unless the torch seed is fixed.
        alpha: credibility values, ``num_bins + 1`` points evenly spaced over
            ``[0, 1]``, see equation 2 of the paper
    """

    thetas, xs = remove_nans_and_infs_in_x(thetas, xs)

    num_tarp_samples, dim_theta = thetas.shape

    if num_tarp_samples < 100:
        warnings.warn(
            "Number of TARP samples should be on the order of 100s to give reliable "
            "results.",
            stacklevel=2,
        )

    posterior_samples = get_posterior_samples_on_batch(
        xs,
        posterior,
        (num_posterior_samples,),
        num_workers,
        show_progress_bar=show_progress_bar,
        use_batched_sampling=use_batched_sampling,
    )
    assert posterior_samples.shape == (
        num_posterior_samples,
        num_tarp_samples,
        dim_theta,
    ), f"Wrong posterior samples shape for TARP: {posterior_samples.shape}"

    # Sample reference points uniformly if not provided
    if references is None:
        references = get_tarp_references(thetas, posterior_samples)

    return _run_tarp(
        posterior_samples, thetas, references, distance, num_bins, z_score_theta
    )


def _run_tarp(
    posterior_samples: Tensor,
    thetas: Tensor,
    references: Tensor,
    distance: Callable = l2,
    num_bins: Optional[int] = None,
    z_score_theta: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Estimates coverage of samples given true values theta with the TARP method.
    Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

    The TARP diagnostic is a global diagnostic which can be used to check a
    trained posterior against a set of true values of theta.

    Args:
        posterior_samples: The predicted parameter samples to compute the coverage of,
                these samples are expected to have shape ``(num_samples,
                num_tarp_samples, num_dims)``. These are obtained by sampling a trained
                posterior `num_samples` times. Multiple (posterior) samples for one
                observation are encouraged.
        theta: The true parameter value theta. Theta is expected to
                 have shape ``(num_tarp_samples, num_dims)``.
        references: the reference points to use for the coverage regions, with
                shape ``(num_tarp_samples, num_dims)``, e.g. from
                ``get_tarp_references``.
        distance: the distance metric to use when computing the distance.
                Should be a callable function that accepts two tensors and
                computes the distance between them, e.g. given two tensors
                of shape ``(batch, 3)`` and ``(batch,3)``, this function should
                return ``(batch,1)`` distance values.
                Possible values: ``sbi.utils.metrics.l1`` or
                ``sbi.utils.metrics.l2``. ``l2`` is the default.
        num_bins: number of bins to use for the credibility values.
                If ``None``, one bin per simulation.
        z_score_theta : whether to rescale each parameter dimension to ``[0, 1]``,
                using the minimum and maximum over thetas and posterior samples.

    Returns:
        ecp: Expected coverage probability (``ecp``), see equation 4 of the paper,
            at each credibility value in ``alpha``, computed from noisy coverage
            values as in ``run_tarp``.
        alpha: grid of credibility values, ``num_bins + 1`` points evenly spaced over
            ``[0, 1]``, see equation 2 of the paper

    """
    num_posterior_samples, num_tarp_samples, _ = posterior_samples.shape
    input_device = posterior_samples.device

    assert references.shape == thetas.shape, (
        "references must have the same shape as thetas"
    )

    if num_bins is None:
        num_bins = num_tarp_samples

    if z_score_theta:
        lo, hi = _get_tarp_bounds(thetas, posterior_samples)
        posterior_samples = (posterior_samples - lo) / (hi - lo + 1e-10)
        thetas = (thetas - lo) / (hi - lo + 1e-10)
        references = (references - lo) / (hi - lo + 1e-10)

    # distances between references and samples
    sample_dists = distance(references, posterior_samples)

    # distances between references and true values
    theta_dists = distance(references, thetas)

    # number of posterior samples closer to the reference than theta (f in algorithm 2
    # is this count divided by num_posterior_samples); theta takes a random position
    # among posterior samples at the same distance
    ties = torch.sum(sample_dists == theta_dists, dim=0)
    counts = torch.sum(sample_dists < theta_dists, dim=0) + torch.floor(
        torch.rand_like(ties, dtype=torch.float32) * (ties + 1)
    )

    # Under calibration the counts are uniform on 0, ..., num_posterior_samples, so
    # the noisy counts below are uniform on [0, 1) for any num_posterior_samples.
    coverage_values = (counts + torch.rand_like(counts)) / (num_posterior_samples + 1)

    # enforce execution on the CPU due to
    # https://github.com/pytorch/pytorch/issues/69519
    # fixed range: the bin edges are the alpha grid and must span [0, 1]
    hist, alpha_grid = torch.histogram(
        coverage_values.cpu(), density=True, bins=num_bins, range=(0.0, 1.0)
    )

    # return all tensors to input_device to keep contract valid
    hist, alpha_grid = hist.to(input_device), alpha_grid.to(input_device)

    # calculate empirical CDF via cumsum and normalize
    ecp = torch.cumsum(hist, dim=0) / hist.sum()
    # add 0 to the beginning of the ecp curve to match the alpha grid
    ecp = torch.cat([torch.zeros((1,), device=input_device), ecp])

    return ecp, alpha_grid


def get_tarp_references(thetas: Tensor, posterior_samples: Tensor) -> Tensor:
    """Returns one TARP reference point per theta, uniform on a box.

    Args:
        thetas: ground-truth parameters, of shape ``(num_tarp_samples, num_dims)``.
        posterior_samples: posterior samples for each theta, of shape
            ``(num_posterior_samples, num_tarp_samples, num_dims)``. The box spans
            thetas and posterior samples.

    Returns:
        Reference points of shape ``(num_tarp_samples, num_dims)``.
    """
    lo, hi = _get_tarp_bounds(thetas, posterior_samples)
    refpdf = torch.distributions.Uniform(low=lo, high=hi)

    return refpdf.sample(torch.Size([thetas.shape[0]]))


def _get_tarp_bounds(
    thetas: Tensor, posterior_samples: Tensor
) -> Tuple[Tensor, Tensor]:
    """Returns the per-dimension minimum and maximum over thetas and posterior samples.

    Bounds that depend on each theta and its posterior samples alike keep them
    exchangeable, so the coverage values stay uniform under calibration.
    """
    lo = torch.minimum(thetas.amin(dim=0), posterior_samples.amin(dim=(0, 1)))
    hi = torch.maximum(thetas.amax(dim=0), posterior_samples.amax(dim=(0, 1)))
    return lo, hi


def check_tarp(
    ecp: Tensor,
    alpha: Tensor,
    num_tarp_samples: Optional[int] = None,
) -> Tuple[float, float]:
    r"""Return the area to curve and a calibration test for the TARP results.

    Args:
        ecp: expected coverage probabilities computed with the TARP method,
            i.e. first output of ``run_tarp``.
        alpha: credibility levels $\alpha$, i.e. second output of ``run_tarp``.
        num_tarp_samples: number of simulations ``run_tarp`` used: the number of
            ``thetas`` minus those it dropped because their ``xs`` contain NaN or
            inf (it warns when it drops any). Needed for the p-value.

    Returns:
        atc: area to curve, the area between the ecp curve and the diagonal for
            alpha values larger than 0.5. Values larger than ``0`` suggest an
            overdispersed posterior (too wide), values smaller than ``0`` an
            underdispersed one (too narrow). The ATC is a signed summary: it can be
            close to ``0`` for a biased posterior or when deviations from the
            diagonal cancel, so read it together with ``ks_pval`` and ``plot_tarp``.
        ks_pval: p-value of a Kolmogorov-Smirnov test of the null hypothesis that
            the coverage values are uniform on ``[0, 1]``, which holds for a
            calibrated posterior. Small values (commonly below 0.05) indicate
            miscalibration. The KS distance is taken on the ``alpha`` grid only, so
            the test is conservative and loses power with fewer bins than
            simulations. The test assumes independent posterior samples: for MCMC
            posteriors, thin the chains until the samples are close to independent.
            NaN if ``num_tarp_samples`` is not given.
    """

    # get the index of the middle of the alpha grid
    midindex = alpha.shape[0] // 2
    dalpha = alpha[1] - alpha[0]
    atc = ((ecp[midindex:] - alpha[midindex:]) * dalpha).sum().item()

    if num_tarp_samples is None:
        warnings.warn(
            "check_tarp needs num_tarp_samples, the number of simulations used by "
            "run_tarp, to compute the KS p-value, and returns NaN without it. "
            "Earlier sbi versions returned a p-value that compared the ecp values "
            "with the alpha grid and did not test calibration. num_tarp_samples "
            "will become a required argument.",
            FutureWarning,
            stacklevel=2,
        )
        return atc, float("nan")
    if num_tarp_samples < 1:
        raise ValueError(f"num_tarp_samples must be positive, got {num_tarp_samples}.")

    # ecp at each alpha is the empirical CDF of the coverage values just below
    # alpha, so this is the KS distance on the grid, a lower bound of the full one.
    ks_distance = (ecp - alpha).abs().max().item()
    ks_pval = float(kstwo(num_tarp_samples).sf(ks_distance))

    return atc, ks_pval
